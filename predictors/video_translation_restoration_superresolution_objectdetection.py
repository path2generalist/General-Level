"""
Unified evaluator for four video–vision tasks and their metrics

    • Video Translation                →  Frame-Acc  (CLIP-based)
    • Video Restoration (去噪/去模糊/…)  →  PSNR
    • Video Super-Resolution           →  MUSIQ      (no-reference IQA)
    • Video (Salient / Camouflaged) Object Detection →  Structure-measure

"""

from __future__ import annotations

import os
import sys
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

# ───────────────────────── third-party imports ────────────────────────────
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T

import open_clip                # Frame-Acc

import pyiqa                    # MUSIQ


# Accepted image extensions (case-insensitive)
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

# ─────────────────────────────  dataclass  ────────────────────────────────
@dataclass
class Instance:
    """Single sample inside the JSON"""
    input: Dict[str, Any]
    output: Dict[str, Any]
    id: str

# ──────────────────────────────  abstract  ────────────────────────────────
class BaseTask(ABC):
    def __init__(self, task_data: Dict[str, Any]):
        self.task_data = task_data
        self.data: List[Instance] = self._parse_data(task_data)

    # --- implement in subclass ------------------------------------------------
    @abstractmethod
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        ...

    @abstractmethod
    def run_inference(self) -> None:
        """collect paths & meta ⇒ self.records   (does *not* run a model)"""
        ...

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        ...

# ════════════════════════════════════════════════════════════════════════════
#  1.  Video Translation  –  Frame-Acc
# ════════════════════════════════════════════════════════════════════════════
class VideoTranslationTask(BaseTask):
    def _parse_data(self, task_data):
        return [Instance(**d) for d in task_data["data"]]

    def run_inference(self):
        """gather [(frame_paths, src_prompt, tgt_prompt), …]"""
        self.records: List[Tuple[List[str], str, str]] = []
        for inst in tqdm(self.data, desc="collect-frames"):
            frame_dir = inst.output["frame_dir"]
            frames = sorted(
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.lower().endswith(IMG_EXTS)
            )
            self.records.append((frames,
                                 inst.input["source_prompt"],
                                 inst.input["target_prompt"]))

    @torch.no_grad()
    def evaluate(self, batch_size: int = 32):
        if open_clip is None:
            raise ImportError("open_clip_torch not installed.  pip install open_clip_torch")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        model.eval()
        tokenizer = open_clip.tokenize

        total, correct = 0, 0
        for frame_paths, src_prompt, tgt_prompt in tqdm(self.records, desc="Frame-Acc eval"):
            text_feat = model.encode_text(
                tokenizer([src_prompt, tgt_prompt]).to(device)
            ).float()
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)   # (2,D)

            for i in range(0, len(frame_paths), batch_size):
                batch_files = frame_paths[i:i + batch_size]
                imgs = torch.stack([
                    preprocess(Image.open(p).convert("RGB")) for p in batch_files
                ]).to(device)
                img_feat = model.encode_image(imgs).float()
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)   # (B,D)
                sim = img_feat @ text_feat.T                                 # (B,2)
                correct += (sim[:, 1] > sim[:, 0]).sum().item()
                total += sim.size(0)

        return {"Frame-Acc": 100.0 * correct / total if total else 0.0}


# ════════════════════════════════════════════════════════════════════════════
#  2.  Video Restoration suite  –  PSNR
# ════════════════════════════════════════════════════════════════════════════
def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    mse = np.mean((img1 - img2) ** 2, dtype=np.float64)
    if mse == 0:
        return math.inf
    return 10.0 * math.log10((max_val ** 2) / mse)


class VideoRestorationTask(BaseTask):
    def _parse_data(self, task_data):
        return [Instance(**d) for d in task_data["data"]]

    def run_inference(self):
        """gather [(pred_paths, gt_paths), …]"""
        self.records: List[Tuple[List[str], List[str]]] = []
        for inst in tqdm(self.data, desc="collect-frames"):
            pred_dir = inst.input["pred_dir"]
            gt_dir = inst.input["gt_dir"]

            frame_names = sorted(
                f for f in os.listdir(gt_dir) if f.lower().endswith(IMG_EXTS)
            )
            pred_paths, gt_paths = [], []
            for fname in frame_names:
                p_path = os.path.join(pred_dir, fname)
                g_path = os.path.join(gt_dir,   fname)
                if not os.path.exists(p_path):
                    raise FileNotFoundError(f"Missing prediction frame: {p_path}")
                pred_paths.append(p_path)
                gt_paths.append(g_path)
            self.records.append((pred_paths, gt_paths))

    def evaluate(self):
        psnr_sum, valid_frames = 0.0, 0

        for preds, gts in tqdm(self.records, desc="PSNR eval"):
            for p, g in zip(preds, gts):
                img1 = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
                img2 = np.array(Image.open(g).convert("RGB"), dtype=np.float32)

                if img1.shape != img2.shape:
                    raise ValueError(f"Shape mismatch: {p} vs {g}")

                val = compute_psnr(img1, img2)
                if math.isfinite(val):
                    psnr_sum += val
                    valid_frames += 1

        return {"PSNR": psnr_sum / valid_frames if valid_frames else 0.0}

# ════════════════════════════════════════════════════════════════════════════
#  3.  Video Super-Resolution  –  MUSIQ
# ════════════════════════════════════════════════════════════════════════════
class VideoSuperResolutionTask(BaseTask):
    def _parse_data(self, task_data):
        return [Instance(**d) for d in task_data["data"]]

    def run_inference(self):
        self.records: List[List[str]] = []
        for inst in tqdm(self.data, desc="collect-frames"):
            pred_dir = inst.input["pred_dir"]
            frames = sorted(
                os.path.join(pred_dir, f)
                for f in os.listdir(pred_dir)
                if f.lower().endswith(IMG_EXTS)
            )
            if not frames:
                raise RuntimeError(f"No prediction frames found in {pred_dir}")
            self.records.append(frames)

    @torch.no_grad()
    def evaluate(self, batch_size: int = 8):
        if pyiqa is None:
            raise ImportError("pyiqa not installed.  pip install pyiqa")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = pyiqa.create_metric('musiq', device=device, as_loss=False)
        model.eval()
        transform = T.ToTensor()

        total_sum, total_frames = 0.0, 0
        for frames in tqdm(self.records, desc="MUSIQ eval"):
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                imgs = torch.stack([
                    transform(Image.open(p).convert("RGB")) for p in batch
                ]).to(device)
                scores = model(imgs)                        # (B,)
                total_sum += scores.sum().item()
                total_frames += scores.numel()

        return {"MUSIQ": total_sum / total_frames if total_frames else 0.0}


# ════════════════════════════════════════════════════════════════════════════
#  4.  Video (Salient / Camouflaged) Object Detection  –  Structure-measure
# ════════════════════════════════════════════════════════════════════════════
def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mp, mg = pred.mean(), gt.mean()
    var_p, var_g = pred.var(), gt.var()
    cov = ((pred - mp) * (gt - mg)).mean()
    return ((2 * mp * mg + C1) * (2 * cov + C2)) / (
        (mp ** 2 + mg ** 2 + C1) * (var_p + var_g + C2) + 1e-8)


def _object_score(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    mu, sigma = x.mean(), x.std()
    return 2 * mu / (mu * mu + 1 + sigma + 1e-8)


def structure_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    """pred in [0,1] float32, gt binary uint8 (0/1)"""
    y = gt.mean()
    if y == 0:   # GT 全黑
        return 1.0 - pred.mean()
    if y == 1:   # GT 全白
        return pred.mean()

    # ─── object-aware term ─────────────────────────────────────────────────
    S_fg = _object_score(pred[gt > 0.5])
    S_bg = _object_score(1 - pred[gt <= 0.5])
    s_object = y * S_fg + (1 - y) * S_bg

    # ─── region-aware term ────────────────────────────────────────────────
    h, w = gt.shape
    rows, cols = np.where(gt > 0.5)
    cx = int(np.round(cols.mean())) if cols.size else w // 2
    cy = int(np.round(rows.mean())) if rows.size else h // 2

    def split(img):
        return [img[:cy, :cx], img[:cy, cx:], img[cy:, :cx], img[cy:, cx:]]

    regions_p = split(pred)
    regions_g = split(gt.astype(np.float32))

    weights = [r.size / (h * w) for r in regions_g]
    ssim_scores = [_ssim(p_r, g_r) for p_r, g_r in zip(regions_p, regions_g)]
    s_region = sum(w * s for w, s in zip(weights, ssim_scores))

    score = alpha * s_object + (1 - alpha) * s_region
    return max(score, 0.0)


class VideoObjectDetectionTask(BaseTask):
    def _parse_data(self, task_data):
        return [Instance(**d) for d in task_data["data"]]

    def run_inference(self):
        self.records: List[Tuple[List[str], List[str]]] = []
        for inst in tqdm(self.data, desc="collect-frames"):
            pred_dir = inst.input["pred_dir"]
            gt_dir = inst.input["gt_dir"]

            frame_names = sorted(
                f for f in os.listdir(gt_dir) if f.lower().endswith(IMG_EXTS)
            )
            preds, gts = [], []
            for fname in frame_names:
                p_path = os.path.join(pred_dir, fname)
                g_path = os.path.join(gt_dir,   fname)
                if not os.path.exists(p_path):
                    raise FileNotFoundError(f"Missing prediction frame: {p_path}")
                preds.append(p_path)
                gts.append(g_path)
            self.records.append((preds, gts))

    def evaluate(self):
        total_sum, total_frames = 0.0, 0

        for preds, gts in tqdm(self.records, desc="S-measure eval"):
            for p, g in zip(preds, gts):
                pred = np.array(Image.open(p).convert('L'), dtype=np.float32)
                if pred.max() > 1.0:
                    pred /= 255.0
                gt = (np.array(Image.open(g).convert('L')) > 128).astype(np.uint8)

                if pred.shape != gt.shape:
                    raise ValueError(f"Shape mismatch: {p} vs {g}")

                total_sum += structure_measure(pred, gt)
                total_frames += 1

        return {"S-measure": total_sum / total_frames if total_frames else 0.0}


# ════════════════════════════════════════════════════════════════════════════
#  unified runner
# ═════════════════
TASK_MAPPING = {
    "VideoTranslation":     VideoTranslationTask,
    "VideoRestoration":     VideoRestorationTask,
    "VideoSuperResolution": VideoSuperResolutionTask,
    "VideoObjectDetection": VideoObjectDetectionTask,
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python integrated_eval.py <task_json>")
        sys.exit(1)

    task_json_path = sys.argv[1]
    with open(task_json_path, 'r', encoding='utf-8') as f:
        task_data = json.load(f)

    task_type = task_data.get("type")
    TaskCls = TASK_MAPPING.get(task_type)
    if TaskCls is None:
        raise NotImplementedError(f"Unsupported task type: {task_type}")

    task = TaskCls(task_data)
    task.run_inference()
    metrics = task.evaluate()
    print(f"[{task_type}] Evaluation Results → {metrics}")


if __name__ == "__main__":
    main()