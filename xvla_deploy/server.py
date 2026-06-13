#!/usr/bin/env python3
"""
X-VLA Cloud Inference Server

统一处理 XLerobot（16D）和双臂（12D）模型。
自动读取模型 config 确定输入维度，无需手动区分。

Usage:
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
logger = logging.getLogger("xvla_server")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    logger.error("fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")
    raise

import torch.nn.functional as F


# ============================================================
# X-VLA Model Server
# ============================================================

class XVLAModelServer:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.policy = None
        self.post_processor = None
        self.tokenizer = None
        self.image_keys = []          # 模型期望的摄像头键名
        self.image_shapes = {}        # 每路摄像头的期望尺寸 (C,H,W)
        self.state_dim = 0            # 状态维度
        self.action_dim = 0           # 动作维度
        self.max_token_length = 64    # tokenizer 最大长度

    def load(self):
        logger.info(f"Loading X-VLA model from: {self.model_path}")
        start = time.perf_counter()

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载模型（强制 float32，训练用 bfloat16 省显存但推理兼容性更好）
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.configs import PreTrainedConfig
        config = PreTrainedConfig.from_pretrained(self.model_path)
        config.dtype = "float32"
        self.policy = XVLAPolicy.from_pretrained(self.model_path, config=config)
        self.policy = self.policy.to(self.device)
        self.policy.eval()

        # 加载后处理器（反归一化模型输出的 action）
        from lerobot.processor import PolicyProcessorPipeline
        self.post_processor = PolicyProcessorPipeline.from_pretrained(
            self.model_path, "policy_postprocessor.json"
        )
        logger.info(f"✅ Post-processor loaded (denormalizes actions)")

        # 扩展位置编码（训练模型是 512，推理需要 2048 容纳 3 路摄像头 token）
        pos = self.policy.model.transformer.pos_emb
        TARGET_LEN = 2048
        if pos.shape[1] < TARGET_LEN:
            new_pos = torch.nn.functional.interpolate(
                pos.permute(0, 2, 1), size=TARGET_LEN, mode="linear"
            ).permute(0, 2, 1)
            self.policy.model.transformer.pos_emb = torch.nn.Parameter(new_pos)
            logger.info(f"📐 pos_emb 扩展: {pos.shape[1]} → {TARGET_LEN}")
        logger.info(f"✅ Model loaded on {self.device}")

        # 读取模型配置，自动确定输入输出格式
        cfg = self.policy.config
        self.image_keys = []
        self.image_shapes = {}
        for k, v in cfg.input_features.items():
            if v.type == "VISUAL":
                self.image_keys.append(k)
                self.image_shapes[k] = v.shape  # (C, H, W)
        state_feat = cfg.input_features.get("observation.state")
        self.state_dim = state_feat.shape[0] if state_feat else 0
        action_feat = cfg.output_features.get("action")
        self.action_dim = action_feat.shape[0] if action_feat else 20
        self.max_token_length = getattr(cfg, "tokenizer_max_length", 64)

        # 加载 tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            getattr(cfg, "tokenizer_name", "facebook/bart-large"),
            model_max_length=self.max_token_length,
        )

        elapsed = time.perf_counter() - start
        logger.info(f"✅ Model loaded on {self.device} ({elapsed:.1f}s)")
        logger.info(f"   State dim: {self.state_dim}, Action dim: {self.action_dim}")
        logger.info(f"   Camera keys: {self.image_keys}")

    def infer(self, proprio: np.ndarray, images: list[np.ndarray],
              instruction: str, domain_id: int = 0) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("Model not loaded.")

        device = next(self.policy.parameters()).device
        dtype = torch.float32  # 推理统一用 float32

        # 1. Tokenize instruction
        tokens = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
        )
        # 2. 构建 batch（全部移到模型所在设备并转为模型的数据类型）
        batch = {
            "observation.state": torch.from_numpy(proprio).float().unsqueeze(0).to(device=device, dtype=dtype),
            "observation.language.tokens": tokens["input_ids"].to(device),
            "observation.language.attention_mask": tokens["attention_mask"].to(device),
        }
        # 3. 添加摄像头图像（缩放至模型期望的尺寸，应用 ImageNet 归一化）
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, -1, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, -1, 1, 1)
        for idx, key in enumerate(self.image_keys):
            target_c, target_h, target_w = self.image_shapes.get(key, (3, 256, 256))
            if idx < len(images):
                img = images[idx].astype(np.float32) / 255.0
                img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
                if img_t.shape[2] != target_h or img_t.shape[3] != target_w:
                    img_t = F.interpolate(img_t, size=(target_h, target_w), mode="bilinear")
                img_t = (img_t - imagenet_mean) / imagenet_std
                batch[key] = img_t
            else:
                batch[key] = torch.zeros(1, target_c, target_h, target_w, device=device, dtype=dtype)

        # 4. 推理（用 predict_action_chunk 一次性返回多步动作，减少 HTTP 请求）
        with torch.no_grad():
            action_chunk = self.policy.predict_action_chunk(batch).cpu()

        action_chunk = np.asarray(action_chunk, dtype=np.float32)

        # 5. 反归一化：将标准化后的 action 还原为原始机器人关节角度范围
        if self.post_processor is not None:
            action_tensor = torch.from_numpy(action_chunk)
            action_tensor = self.post_processor.process_action(action_tensor)
            action_chunk = action_tensor.numpy()

        # 去掉 batch 维度: (1, 30, 16) → (30, 16)
        if action_chunk.ndim == 3:
            action_chunk = action_chunk[0]

        return action_chunk


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="X-VLA Inference Server", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_server: XVLAModelServer | None = None


@app.on_event("startup")
async def startup():
    global model_server
    args = app.state.args
    model_server = XVLAModelServer(args.model_path, args.device)
    model_server.load()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": model_server.model_path if model_server else None,
        "device": model_server.device if model_server else None,
        "state_dim": model_server.state_dim if model_server else None,
        "action_dim": model_server.action_dim if model_server else None,
        "cameras": model_server.image_keys if model_server else None,
    }


@app.post("/act")
async def act(data: dict):
    if model_server is None or model_server.policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 解析输入
        proprio = np.array(json.loads(data["proprio"]), dtype=np.float32)
        instruction = data.get("language_instruction", "")

        # 解析摄像头（自动读取模型期望的路数）
        images = []
        for i in range(len(model_server.image_keys)):
            img_key = f"image{i}"
            if img_key in data:
                img = np.array(json.loads(data[img_key]), dtype=np.uint8)
                images.append(img)
            else:
                images.append(np.zeros((256, 256, 3), dtype=np.uint8))

        logger.info(f"Request: proprio={proprio.shape[0]}d, "
                     f"images={len(images)}, instruction='{instruction[:40]}...'")

        infer_start = time.perf_counter()
        action_chunk = model_server.infer(proprio, images, instruction)
        infer_time = time.perf_counter() - infer_start

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
    logger.info(f"  API: POST http://{args.host}:{args.port}/act")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
