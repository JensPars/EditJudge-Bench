"""Supervised judges: EditReward and EditScore (upstream code vendored in third_party/)."""
import gc
import json
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"


def run(cfg: dict, examples: list) -> list:
    return _editscore(cfg, examples) if cfg["backend"] == "editscore" else _editreward(cfg, examples)


def _editreward(cfg, examples):
    """TIGER-Lab/EditReward-Qwen2.5-VL-7B. Needs its own environment (see README)."""
    import torch
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(THIRD_PARTY / "editreward"))
    from EditReward import EditRewardInferencer

    ckpt = os.environ.get("EDITREWARD_CHECKPOINT") or snapshot_download(cfg["model"])
    inf = EditRewardInferencer(config_path=str(THIRD_PARTY / "editreward/EditReward/config/EditReward-Qwen2.5-7B-VL.yaml"),
                               checkpoint_path=ckpt, device="cuda", reward_dim="dim1",
                               rm_head_type="ranknet_multi_head")
    rows = []
    for i in tqdm(range(0, len(examples), 32), desc="EditReward"):
        batch = examples[i:i + 32]
        with torch.no_grad():
            rewards = inf.reward([e["instruction"] for e in batch], [e["before"] for e in batch],
                                 [e["after"] for e in batch])
        for r in rewards:   # [mean, log_sigma]; the reward is the mean
            rows.append({"raw_output": f"logit={r[0].item():.4f}", "score": r[0].item()})
    del inf
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def _editscore(cfg, examples):
    """EditScore/EditScore-Qwen3-VL-4B-Instruct with upstream settings (transformers backend)."""
    import torch

    sys.path.insert(0, str(THIRD_PARTY / "editscore"))
    from editscore import EditScore

    scorer = EditScore(backbone="qwen3vl", model_name_or_path=cfg["base_model"], lora_path=cfg["model"],
                       score_range=25, temperature=0.7, num_pass=1, seed=42)
    keys = ("overall", "prompt_following", "consistency", "perceptual_quality")
    rows = []
    for ex in tqdm(examples, desc="EditScore"):
        with torch.no_grad(), Image.open(ex["before"]) as b, Image.open(ex["after"]) as a:
            res = scorer.evaluate([b.convert("RGB"), a.convert("RGB")], ex["instruction"])
        raw = json.dumps({k: float(res[k]) if k in keys else res[k] for k in keys + ("SC_reasoning", "PQ_reasoning")},
                         sort_keys=True)
        rows.append({"raw_output": raw, "score": float(res["overall"])})
    del scorer
    torch.cuda.empty_cache()
    gc.collect()
    return rows
