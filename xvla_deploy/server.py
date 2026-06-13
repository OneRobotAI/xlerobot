#!/usr/bin/env python3
"""
X-VLA Cloud Inference Server

A FastAPI-based inference server for X-VLA models.
Designed for deployment on GPU cloud instances.

Usage:
  # Start server (default)
  python server.py
  
  # With custom model and port
  python server.py --model-path your/xvla-xlerobot-fold --port 8000

API Endpoint:
  POST /act  -- Run policy inference
  GET  /health -- Health check
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xvla_server")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    logger.error("fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")
    raise

# ============================================================
# X-VLA Model Loader
# ============================================================


class XVLAModelServer:
    """Wraps an X-VLA model with a simple inference API."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.policy = None
        self.processor = None

    def load(self):
        """Load the X-VLA model from HuggingFace Hub or local path."""
        logger.info(f"Loading X-VLA model from: {self.model_path}")
        start = time.perf_counter()

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Option 1: Load via LeRobot (recommended for fine-tuned policies)
        try:
            from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
            self.policy = XVLAPolicy.from_pretrained(self.model_path)
            self.policy = self.policy.to(self.device)
            logger.info(f"✅ Loaded via LeRobot XVLAPolicy on {self.device}")
        except (ImportError, Exception) as e:
            logger.warning(f"LeRobot loading failed ({e}), trying Transformers AutoModel...")
            # Option 2: Load via Transformers (original X-VLA repo format)
            from transformers import AutoModel, AutoProcessor
            self.policy = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            logger.info(f"✅ Loaded via Transformers AutoModel on {self.device}")

        elapsed = time.perf_counter() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

    def infer(self, proprio: np.ndarray, images: list[np.ndarray],
              instruction: str, domain_id: int = 0, steps: int = 10) -> np.ndarray:
        """Run policy inference.
        
        Args:
            proprio: Robot joint positions (12D for XLeRobot)
            images: List of camera images [(H,W,3), ...]
            instruction: Language task instruction
            domain_id: Domain/embodiment identifier
            steps: Number of denoising steps
            
        Returns:
            action_chunk: Predicted action sequence, shape (chunk_size, action_dim)
        """
        if self.policy is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build observation dict for XVLAPolicy
        obs = self._build_observation(proprio, images, instruction, domain_id)

        with torch.no_grad():
            # Prepare input tensors
            if hasattr(self.policy, "select_action"):
                # LeRobot XVLAPolicy interface
                action_chunk = self.policy.select_action(obs)
                if isinstance(action_chunk, torch.Tensor):
                    action_chunk = action_chunk.cpu().numpy()
            else:
                # Transformers AutoModel interface (original X-VLA)
                action_chunk = self._transformers_infer(obs, steps)

        return np.asarray(action_chunk, dtype=np.float32)

    def _build_observation(self, proprio, images, instruction, domain_id):
        """Build observation dict matching XVLAPolicy expectations."""
        obs = {
            "observation.state": torch.from_numpy(proprio).float(),
            "observation.task": instruction,
        }
        # Add camera images
        for i, img in enumerate(images):
            obs[f"observation.images.cam_{i}"] = torch.from_numpy(img).float()
        # Add domain_id if supported
        if self.processor is not None:
            obs["domain_id"] = domain_id
        return obs

    def _transformers_infer(self, obs, steps):
        """Inference using original X-VLA Transformers interface."""
        # This path is used when loaded via AutoModel
        if self.processor is None:
            raise RuntimeError("Processor not available for Transformers inference")

        # Process inputs
        inputs = self.processor(
            images=[obs.get(f"observation.images.cam_{i}", None) for i in range(3)],
            text=obs.get("observation.task", ""),
            proprio=obs.get("observation.state"),
            domain_id=obs.get("domain_id", 0),
            return_tensors="pt",
        ).to(self.device)

        # Run model
        outputs = self.policy.generate(
            **inputs,
            num_denoising_steps=steps,
        )
        return outputs.actions.cpu().numpy()


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="X-VLA Inference Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_server: XVLAModelServer | None = None


@app.on_event("startup")
async def startup():
    """Load model on server start."""
    global model_server
    args = app.state.args
    model_server = XVLAModelServer(args.model_path, args.device)
    model_server.load()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": model_server.model_path if model_server else None,
        "device": model_server.device if model_server else None,
    }


@app.post("/act")
async def act(data: dict):
    """Run policy inference on submitted observation.
    
    Request format:
    {
        "proprio": "[...]",       # JSON-serialized numpy array (12D joint positions)
        "language_instruction": "fold the towel on the table",
        "image0": "[...]",       # JSON-serialized numpy array (H,W,3) uint8
        "image1": "[...]",       # Optional second camera
        "image2": "[...]",       # Optional third camera
        "domain_id": 0,          # Domain/embodiment ID
        "steps": 10              # Denoising steps
    }
    """
    if model_server is None or model_server.policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Parse inputs
        proprio = np.array(json.loads(data["proprio"]), dtype=np.float32)
        instruction = data.get("language_instruction", "")
        domain_id = int(data.get("domain_id", 0))
        steps = int(data.get("steps", 10))

        # Parse images (up to 3)
        images = []
        for i in range(3):
            img_key = f"image{i}"
            if img_key in data:
                img = np.array(json.loads(data[img_key]), dtype=np.uint8)
                images.append(img)
            else:
                # Create a dummy image if not provided
                images.append(np.zeros((256, 256, 3), dtype=np.uint8))

        logger.info(f"Received request: proprio_dim={proprio.shape[0]}, "
                     f"images={len(images)}, instruction='{instruction[:50]}...'")

        # Run inference
        infer_start = time.perf_counter()
        action_chunk = model_server.infer(proprio, images, instruction, domain_id, steps)
        infer_time = time.perf_counter() - infer_start

        logger.info(f"Inference done: {action_chunk.shape}, time={infer_time*1000:.0f}ms")

        # Serialize response
        return {
            "action": json.loads(json.dumps(action_chunk.tolist())),
            "shape": list(action_chunk.shape),
            "inference_time_ms": round(infer_time * 1000, 1),
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CLI & Entry Point
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="X-VLA Cloud Inference Server")
    parser.add_argument("--model-path", default="lerobot/xvla-folding",
                        help="Model path or HF hub ID (default: lerobot/xvla-folding)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--device", default="auto",
                        help='Device: "cuda", "cpu", or "auto"')
    return parser.parse_args()


def main():
    args = parse_args()
    app.state.args = args

    logger.info(f"Starting X-VLA server...")
    logger.info(f"  Model: {args.model_path}")
    logger.info(f"  Listen: {args.host}:{args.port}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  API: POST http://{args.host}:{args.port}/act")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
