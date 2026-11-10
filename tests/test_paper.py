"""Saved predictions reproduce every paper value in configs/ (needs: editjudge download predictions)."""
from pathlib import Path

import pytest

from editjudge import data, metrics
from editjudge.run import load_config

CONFIGS = sorted(Path(__file__).resolve().parents[1].glob("configs/*/*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_paper_value(path):
    cfg = load_config(path)
    saved = data.PREDICTIONS / cfg["benchmark"] / f"{cfg['name']}.parquet"
    if not saved.exists():
        pytest.skip("run: uv run editjudge download predictions")
    assert round(metrics.headline(cfg["benchmark"], saved), 1) == pytest.approx(cfg["paper"], abs=0.051)
