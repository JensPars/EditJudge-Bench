"""Run one judge config (configs/<benchmark>/<name>.toml) and save its predictions."""
import tomllib
from pathlib import Path

import pandas as pd

from editjudge import data, metrics
from editjudge.prompts import build_prompt, parse

OUTPUTS = data.REPO / "outputs"


def load_config(path) -> dict:
    path = Path(path)
    cfg = tomllib.loads(path.read_text())
    cfg["name"], cfg["benchmark"] = path.stem, path.parent.name
    if cfg["benchmark"] not in ("ejb", "gab", "erb"):
        raise SystemExit(f"{path}: configs live in configs/ejb, configs/gab or configs/erb")
    return cfg


def output_path(cfg: dict) -> Path:
    return OUTPUTS / cfg["benchmark"] / f"{cfg['name']}.parquet"


def build_examples(cfg: dict, limit: int = None) -> list:
    """One example per (image pair, instruction)."""
    if cfg["benchmark"] == "ejb":
        df = data.load_benchmark(with_images=True)
        if limit:
            df = df.iloc[:limit]
        rows = []
        for r in df.itertuples():
            rows.append(dict(sample_id=r.sample_id, edit_type=r.edit_type, example_type="positive",
                             negative_type=None, ground_truth=True, instruction=r.instruction_pos,
                             before=r.before, after=r.after))
            for inst, typ in zip(data.as_list(r.instruction_neg_list), data.as_list(r.instruction_neg_types)):
                rows.append(dict(sample_id=r.sample_id, edit_type=r.edit_type, example_type="negative",
                                 negative_type=typ, ground_truth=False, instruction=inst,
                                 before=r.before, after=r.after))
    else:
        df = data.load_bridge(cfg["benchmark"])
        rows = df.iloc[:2 * limit if limit else None].to_dict("records")
    if "template" in cfg:   # reranker and reward models build their own inputs
        for ex in rows:
            ex["system"], ex["parts"] = build_prompt(cfg["template"], ex["instruction"], cfg.get("image_position"))
    return rows


def run(config_path, limit: int = None) -> Path:
    cfg = load_config(config_path)
    if cfg.get("model", "").startswith("datasets/"):   # weights stored inside a dataset repo
        from huggingface_hub import snapshot_download
        _, owner, repo, sub = cfg["model"].split("/", 3)
        root = snapshot_download(f"{owner}/{repo}", repo_type="dataset", allow_patterns=f"{sub}/*")
        cfg["model"] = f"{root}/{sub}"
    examples = build_examples(cfg, limit)
    out = output_path(cfg) if not limit else output_path(cfg).with_suffix(f".first{limit}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{cfg['name']}: {len(examples)} examples, backend={cfg['backend']}, model={cfg['model']}")

    backend = cfg["backend"]
    if backend == "vllm":
        from editjudge.backends import vllm
        res = vllm.run(cfg, examples)
    elif backend in ("openai", "anthropic", "gemini"):
        from editjudge.backends import api
        res = api.run(cfg, examples, out.with_suffix(".jsonl"))
    elif backend == "reranker":
        from editjudge.backends import reranker
        res = reranker.run(cfg, examples)
    elif backend in ("editscore", "editreward"):
        from editjudge.backends import reward
        res = reward.run(cfg, examples)
    else:
        raise SystemExit(f"unknown backend {backend!r}")

    keep = ["sample_id", "edit_type", "example_type", "negative_type", "ground_truth", "instruction"] \
        if cfg["benchmark"] == "ejb" else ["sample_id", "candidate_idx", "instruction"]
    pred = pd.DataFrame([{k: ex[k] for k in keep} for ex in examples])
    pred["score_norm"] = [r["score"] if "score" in r else parse(cfg["template"], r["answer"]) for r in res]
    pred["raw_output"] = [r["raw_output"] for r in res]
    pred.to_parquet(out, index=False)
    print(f"Saved {out}  (parsed {pred['score_norm'].notna().mean():.1%})")
    if not limit:
        metrics.compare(cfg, out)
    return out

