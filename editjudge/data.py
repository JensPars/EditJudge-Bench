"""Paths, downloads and loaders for the benchmark, the saved predictions and the bridge benchmarks."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("EJB_DATA", REPO / "data"))

# EditJudge-Bench on Hugging Face, pinned to the version used in the paper
DATASET_REPO, DATASET_REVISION = "JensParslov/EditJudge-Bench", "v1.1"
BENCHMARK = REPO / "data" / "benchmark.parquet"   # byte-identical copy of the dataset's benchmark.parquet
BENCHMARK_SHA256 = "ebb449ae7ab3296df2320c9b0cb6f5d927ecac513838e3edd53b3553d7923833"
IMAGES = DATA / "editjudge-bench"                   # images/<sample_id>/{before,after}.png

# Saved predictions behind every number in the paper
PREDICTIONS_REPO = "JensParslov/EditJudge-Bench-predictions"
PREDICTIONS = DATA / "predictions"                  # {ejb,gab,erb}/<config name>.parquet

# Real-image human-preference benchmarks (Tables 3-4)
GAB_PARQUET = "hf://datasets/TIGER-Lab/GenAI-Bench/image_edition/test-00000-of-00001.parquet"
ERB_PARQUETS = ["2pair/train-00000-of-00001.parquet", "3pair/train-00000-of-00002.parquet",
                "3pair/train-00001-of-00002.parquet"] + [f"4pair/train-0000{i}-of-00004.parquet" for i in range(4)]
ERB_ROOT = "hf://datasets/TIGER-Lab/EditReward-Bench/"
BRIDGE = DATA / "bridge"                            # {gab,erb}_labels.parquet, {gab,erb}/ candidates + images


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# EditJudge-Bench
# ---------------------------------------------------------------------------

def load_benchmark(with_images: bool = False) -> pd.DataFrame:
    """The 1,382 benchmark pairs. ``with_images`` makes before/after absolute paths."""
    df = pd.read_parquet(BENCHMARK)
    if with_images:
        if not (IMAGES / "images").is_dir():
            raise SystemExit("Images not found. Run: uv run editjudge download images")
        for col in ("before", "after"):
            df[col] = str(IMAGES) + "/" + df[col]
    return df


def download_images():
    from huggingface_hub import snapshot_download
    print(f"Downloading {DATASET_REPO}@{DATASET_REVISION} (~8 GB) -> {IMAGES}")
    snapshot_download(DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION, local_dir=IMAGES)
    if sha256(IMAGES / "benchmark.parquet") != BENCHMARK_SHA256:
        raise SystemExit("benchmark.parquet does not match the pinned version")
    n = sum(1 for _ in (IMAGES / "images").glob("*/*.png"))
    print(f"OK: {n} images")


# ---------------------------------------------------------------------------
# Saved predictions
# ---------------------------------------------------------------------------

def download_predictions():
    from huggingface_hub import snapshot_download
    print(f"Downloading {PREDICTIONS_REPO} -> {PREDICTIONS}")
    snapshot_download(PREDICTIONS_REPO, repo_type="dataset", local_dir=PREDICTIONS)
    bridge_labels()


def saved_predictions(benchmark: str, name: str) -> Path:
    path = PREDICTIONS / benchmark / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: uv run editjudge download predictions")
    return path


# ---------------------------------------------------------------------------
# GenAI-Bench (gab) and EditReward-Bench (erb)
# ---------------------------------------------------------------------------

def bridge_labels() -> dict:
    """Human labels per (sample_id, candidate_idx); read from the source datasets without images."""
    BRIDGE.mkdir(parents=True, exist_ok=True)
    out = {}
    path = BRIDGE / "gab_labels.parquet"
    if not path.exists():
        print("Fetching GenAI-Bench labels ...")
        votes = pd.read_parquet(GAB_PARQUET, columns=["vote_type"])["vote_type"]
        pd.DataFrame([{"sample_id": i, "candidate_idx": c, "vote_type": v}
                      for i, v in enumerate(votes) for c in (0, 1)]).to_parquet(path, index=False)
    out["gab"] = pd.read_parquet(path)
    path = BRIDGE / "erb_labels.parquet"
    if not path.exists():
        print("Fetching EditReward-Bench labels ...")
        t = pd.concat([pd.read_parquet(ERB_ROOT + f, columns=["score_1", "score_2"]) for f in ERB_PARQUETS],
                      ignore_index=True)
        pd.DataFrame([{"sample_id": i, "candidate_idx": c, "human_score": s}
                      for i, (s1, s2) in enumerate(zip(t["score_1"], t["score_2"]))
                      for c, s in ((0, s1), (1, s2))]).to_parquet(path, index=False)
    out["erb"] = pd.read_parquet(path)
    return out


def download_bridge(benchmark: str):
    """Images + one row per candidate (sample_id, candidate_idx, before, after, instruction)."""
    from datasets import load_dataset
    from tqdm import tqdm
    root = BRIDGE / benchmark
    (root / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    if benchmark == "gab":
        ds = load_dataset("TIGER-Lab/GenAI-Bench", "image_edition", split="test")
        keys, instr = ("left_output_image", "right_output_image"), "instruct_prompt"
        src = "source_image"
    else:
        ds = load_dataset("TIGER-Lab/EditReward-Bench", split="train")
        keys, instr, src = ("candidate_1", "candidate_2"), "instruction", "source_image"
    for i, s in enumerate(tqdm(ds, desc=f"Saving {benchmark} images")):
        before = root / "images" / f"{i:04d}_source.png"
        s[src].save(before)
        for c, k in enumerate(keys):
            after = root / "images" / f"{i:04d}_{c}.png"
            s[k].save(after)
            rows.append({"sample_id": i, "candidate_idx": c, "before": str(before), "after": str(after),
                         "instruction": s[instr]})
    pd.DataFrame(rows).to_parquet(root / "candidates.parquet", index=False)
    print(f"OK: {len(rows)} candidates -> {root}")


def load_bridge(benchmark: str) -> pd.DataFrame:
    path = BRIDGE / benchmark / "candidates.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: uv run editjudge download {benchmark}")
    return pd.read_parquet(path)


def as_list(x) -> list:
    return x.tolist() if isinstance(x, np.ndarray) else list(x)
