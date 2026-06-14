#!/usr/bin/env python3
"""
SmolVLA Cloud Inference Server for XLeRobot

SmolVLA (450M params) is HuggingFace's lightweight Vision-Language-Action model.
This server serves the same API as xvla_deploy/server.py (POST /act, GET /health),
so the same client.py can connect to either backend interchangeably.

Architecture:
  Client (XLeRobot hardware)  ──HTTP POST /act──>  Server (SmolVLA on GPU)
       │                                                │
       │  {proprio, image0, image1, image2,             │
       │   language_instruction, steps}                  │
       │                                                │
       │  <──  {action: chunk, inference_time_ms}  ─────│

Usage:
  python server.py --model-path lerobot/smolvla_base --port 8000

  After fine-tuning:
  python server.py --model-path ./outputs/checkpoints/best --port 8000
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smolvla_server")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    logger.error("fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")
    raise

import torch.nn.functional as F


# ============================================================
# Constants
# ============================================================
# Key names SmolVLA expects in the batch dict
OBS_STATE = "observation.state"
OBS_IMAGE_PREFIX = "observation.images."
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


# ============================================================
# SmolVLA Model Server
# ============================================================

class SmolVLAModelServer:
    """Wraps SmolVLAPolicy with preprocessing/postprocessing for HTTP inference."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.image_keys = []           # Model's camera key names (e.g. camera1, camera2, camera3)
        self.image_shapes = {}         # Expected (C, H, W) per camera
        self.state_dim = 0             # Expected state dimension
        self.action_dim = 0            # Expected action dimension
        self.chunk_size = 50           # Number of actions in a chunk
        self.policy_config = None

    def load(self):
        """Load SmolVLA model and pre/post-processors."""
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        logger.info(f"Loading SmolVLA model from: {self.model_path}")
        start = time.perf_counter()

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(self.device)

        # Load policy (always float32 for inference stability)
        # Convert to absolute path to avoid HF Hub repo ID validation issues
        model_path = Path(self.model_path).resolve()
        self.policy = SmolVLAPolicy.from_pretrained(str(model_path))
        self.policy = self.policy.to(device).eval()
        self.policy_config = self.policy.config

        # Read model dimensions from config
        for k, v in self.policy_config.input_features.items():
            if v.type == "VISUAL":
                self.image_keys.append(k)
                self.image_shapes[k] = v.shape  # (C, H, W)
        state_feat = self.policy_config.input_features.get(OBS_STATE)
        self.state_dim = state_feat.shape[0] if state_feat else 0
        action_feat = self.policy_config.output_features.get("action")
        self.action_dim = action_feat.shape[0] if action_feat else 0
        self.chunk_size = getattr(self.policy_config, "chunk_size", 50)

        logger.info(f"✅ Model loaded on {self.device} ({time.perf_counter() - start:.1f}s)")
        logger.info(f"   State dim: {self.state_dim}, Action dim: {self.action_dim}")
        logger.info(f"   Image keys: {[k.split('.')[-1] for k in self.image_keys]}")
        logger.info(f"   Chunk size: {self.chunk_size}")

        # Load pre/post-processors from model checkpoint
        # These handle: renaming, batching, tokenization, normalization, denormalization
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor_overrides = {"device_processor": {"device": str(device)}}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy_config,
            self.model_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={},
        )
        logger.info(f"✅ Pre/post-processors loaded")

        elapsed = time.perf_counter() - start
        logger.info(f"✅ Server ready ({elapsed:.1f}s)")

    def _get_image_key_short(self, full_key: str) -> str:
        """Extract short image key (e.g. 'camera1') from full key (e.g. 'observation.images.camera1')."""
        return full_key.replace(OBS_IMAGE_PREFIX, "")

    def infer(
        self,
        proprio: np.ndarray,
        images: list[np.ndarray],
        instruction: str,
    ) -> np.ndarray:
        """Run SmolVLA inference and return a full action chunk.

        Args:
            proprio: (state_dim,) float32 joint position array
            images: List of (H, W, 3) uint8 images in model's camera order
            instruction: Language instruction string

        Returns:
            (chunk_size, action_dim) float32 action chunk
        """
        if self.policy is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        device = next(self.policy.parameters()).device

        # 1. Build raw frame (unnormalized, with task string)
        #    The preprocessor expects:
        #      - observation.state: float32 tensor (D,)
        #      - observation.images.<key>: uint8 tensor (H, W, 3)
        #      - task: raw instruction string
        frame = {
            OBS_STATE: torch.from_numpy(proprio).float(),
        }

        # Add images using model's expected key names
        for idx, key in enumerate(self.image_keys):
            if idx < len(images):
                # Convert uint8 to float32 in [0,1], then to (C, H, W) for SmolVLA
                img = torch.from_numpy(images[idx].astype(np.float32) / 255.0)
                frame[key] = img.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
            else:
                # Fill missing cameras with zeros
                c, h, w = self.image_shapes.get(key, (3, 256, 256))
                frame[key] = torch.zeros((c, h, w), dtype=torch.float32)

        # Add language instruction as raw string (preprocessor handles tokenization)
        frame["task"] = instruction

        # 2. Preprocess: normalize state, tokenize instruction, add batch dim, move to device
        batch = self.preprocessor(frame)

        # 3. Run inference - predict full action chunk
        with torch.no_grad():
            # predict_action_chunk returns (batch_size, chunk_size, action_dim)
            action_chunk = self.policy.predict_action_chunk(batch)

        # 4. Postprocess: denormalize actions, move to CPU
        action_out = self.postprocessor(action_chunk)
        # action_out is a dict with key "action": (1, chunk_size, action_dim)
        if isinstance(action_out, dict):
            action_chunk = action_out.get("action", action_chunk)
        else:
            action_chunk = action_out

        # Remove batch dim: (1, chunk_size, action_dim) -> (chunk_size, action_dim)
        action_chunk = np.asarray(action_chunk.cpu().float())
        if action_chunk.ndim == 3:
            action_chunk = action_chunk[0]

        return action_chunk


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="SmolVLA Inference Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_server: SmolVLAModelServer | None = None


@app.on_event("startup")
async def startup():
    global model_server
    args = app.state.args
    model_server = SmolVLAModelServer(args.model_path, args.device)
    model_server.load()


@app.get("/health")
async def health():
    if model_server is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model": model_server.model_path,
        "device": model_server.device,
        "state_dim": model_server.state_dim,
        "action_dim": model_server.action_dim,
        "chunk_size": model_server.chunk_size,
        "cameras": [k.split(".")[-1] for k in (model_server.image_keys or [])],
    }


@app.post("/act")
async def act(data: dict):
    """Accept observation, return predicted action chunk.

    Compatible with xvla_deploy/client.py request format:
      proprio:    JSON array of joint angles (state_dim floats)
      image0-2:   JSON array of (H, W, 3) uint8 images
      language_instruction: str
      steps:      int (denoising steps, passed to model)

    Returns:
      action:           (chunk_size, action_dim) float32 array
      inference_time_ms: float
    """
    global model_server
    if model_server is None or model_server.policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Parse proprioception
        proprio = np.array(json.loads(data["proprio"]), dtype=np.float32)
        instruction = data.get("language_instruction", "")

        # Parse camera images
        images = []
        for i in range(len(model_server.image_keys)):
            img_key = f"image{i}"
            if img_key in data:
                img = np.array(json.loads(data[img_key]), dtype=np.uint8)
                images.append(img)
            else:
                # Placeholder if camera missing
                c, h, w = list(model_server.image_shapes.values())[0]
                images.append(np.zeros((h, w, 3), dtype=np.uint8))

        logger.info(
            f"Request: proprio={proprio.shape[0]}d, "
            f"images={len(images)}, instruction='{instruction[:40]}...'"
        )

        infer_start = time.perf_counter()
        action_chunk = model_server.infer(proprio, images, instruction)
        infer_time = time.perf_counter() - infer_start

        # Truncate/pad to match whatever the client expects
        logger.info(f"Inference done: {action_chunk.shape}, {infer_time*1000:.0f}ms")

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
    parser = argparse.ArgumentParser(description="SmolVLA Cloud Inference Server")
    parser.add_argument(
        "--model-path",
        default="lerobot/smolvla_base",
        help="Model path or HF hub ID (default: lerobot/smolvla_base)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device: "cuda", "cpu", or "auto" (default: auto)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app.state.args = args

    logger.info("Starting SmolVLA server...")
    logger.info(f"  Model: {args.model_path}")
    logger.info(f"  Listen: {args.host}:{args.port}")
    logger.info(f"  API: POST http://{args.host}:{args.port}/act")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
