"""Rebuild the paper's tables and Fig. 5 from predictions.

By default the saved paper predictions are used (data/predictions/). With
``--mine`` your own runs (outputs/) are used where they exist.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from editjudge import data, metrics
from editjudge.run import OUTPUTS

# (method, base model, config name)
TABLE1 = [
    ("VIE-score", "LLaVA-OV-7B", "vie__llava-ov-7b"),
    ("VIE-score", "Qwen3-VL-4B", "vie__qwen3-vl-4b"),
    ("VIE-score", "Qwen3-VL-8B", "vie__qwen3-vl-8b"),
    ("VIE-score", "Qwen3-VL-8B + Think", "vie__qwen3-vl-8b-thinking"),
    ("VIE-score", "Qwen3-VL-32B", "vie__qwen3-vl-32b"),
    ("VIE-score", "Qwen3-VL-32B + Think", "vie__qwen3-vl-32b-thinking"),
    ("VIE-score", "Gemma-4-31B", "vie__gemma-4-31b"),
    ("Kris", "Qwen3-VL-8B", "kris__qwen3-vl-8b"),
    ("Kris", "Gemma-4-31B", "kris__gemma-4-31b"),
    ("ImgEdit", "Qwen3-VL-8B", "imgedit__qwen3-vl-8b"),
    ("ImgEdit", "Gemma-4-31B", "imgedit__gemma-4-31b"),
    ("VIE-score", "Gemini-2.5-flash", "vie__gemini-2.5-flash"),
    ("VIE-score", "GPT-4.1-mini", "vie__gpt-4.1-mini"),
    ("VIE-score", "GPT-4o", "vie__gpt-4o"),
    ("VIE-score", "GPT-5.4", "vie__gpt-5.4"),
    ("VIE-score", "GPT-5.6-sol", "vie__gpt-5.6-sol"),
    ("VIE-score", "Opus 5", "vie__claude-opus-5"),
    ("Qwen-Reranker", "Qwen3-VL-Reranker-8B", "reranker__qwen3-vl-reranker-8b"),
    ("Log-prob", "Qwen3-VL-4B", "logprob__qwen3-vl-4b"),
    ("Log-prob", "Qwen3-VL-8B", "logprob__qwen3-vl-8b"),
    ("Log-prob", "Qwen2.5-VL-7B", "logprob__qwen2.5-vl-7b"),
    ("Log-prob", "Gemma-4-31B", "logprob__gemma-4-31b"),
    ("EditScore", "Qwen3-VL-4B", "editscore__qwen3-vl-4b"),
    ("ImgEdit-Judge", "Qwen2.5-VL-7B", "imgedit-judge__qwen2.5-vl-7b"),
    ("EditReward", "Qwen2.5-VL-7B", "editreward__qwen2.5-vl-7b"),
]
# (method, base model, EditJudge-Bench config, GenAI-Bench / EditReward-Bench config)
TABLE3 = [
    ("VIE-score", "LLaVA-OV-7B", "vie__llava-ov-7b", "vie__llava-ov-7b"),
    ("VIE-score", "Qwen2.5-VL-7B", "vie__qwen2.5-vl-7b", "vie__qwen2.5-vl-7b"),
    ("VIE-score", "Qwen3-VL-4B", "vie__qwen3-vl-4b", "vie__qwen3-vl-4b"),
    ("VIE-score", "Qwen3-VL-8B", "vie__qwen3-vl-8b", "vie__qwen3-vl-8b"),
    ("VIE-score", "Qwen3-VL-32B", "vie__qwen3-vl-32b", "vie__qwen3-vl-32b"),
    ("VIE-score", "Gemma-4-31B", "vie__gemma-4-31b", "vie__gemma-4-31b"),
    ("Qwen-Reranker", "Qwen3-VL-Reranker-8B", "reranker__qwen3-vl-reranker-8b", "reranker__qwen3-vl-reranker-8b"),
    ("Log-prob", "Qwen2.5-VL-7B", "logprob-simple__qwen2.5-vl-7b", "logprob-simple__qwen2.5-vl-7b"),
    ("Log-prob", "Qwen3-VL-8B", "logprob-noq__qwen3-vl-8b", "logprob-noq__qwen3-vl-8b"),
    ("Log-prob", "Gemma-4-31B", "logprob__gemma-4-31b", "logprob-noq__gemma-4-31b"),
    ("EditReward", "Qwen2.5-VL-7B", "editreward__qwen2.5-vl-7b", "editreward__qwen2.5-vl-7b"),
]
# (model, images before text, images after text)
TABLE5 = [
    ("Gemini-2.5-flash", "vie__gemini-2.5-flash__images-first", "vie__gemini-2.5-flash"),
    ("GPT-4.1-mini", "vie__gpt-4.1-mini__images-first", "vie__gpt-4.1-mini"),
]

COLS = {"removal": "Rem", "material": "Mat", "lighting": "Lgt", "camera": "Cam", "movement": "Mov",
        "scale": "Scl", "shape": "Shp", "articulation": "Art", "rotation": "Rot", "overall": "Overall"}


def _path(bench: str, name: str, mine: bool) -> Path:
    own = OUTPUTS / bench / f"{name}.parquet"
    return own if mine and own.exists() else data.saved_predictions(bench, name)


def table1(mine=False):
    rows = []
    for method, base, name in TABLE1:
        df = pd.read_parquet(_path("ejb", name, mine))
        rows.append({"Method": method, "Base model": base, **metrics.per_edit_type(df),
                     "MC Acc.": metrics.mc_accuracy(df)})
    return pd.DataFrame(rows).rename(columns=COLS)


def table2(mine=False):
    rows = []
    for method, base, name in TABLE1:
        df = pd.read_parquet(_path("ejb", name, mine))
        rows.append({"Method": method, "Base model": base, **metrics.per_negative_type(df)})
    return pd.DataFrame(rows).rename(columns={"dir": "Dir.", "cross": "Cross", "wrong_obj": "Wrong obj.",
                                              "no_edit": "No edit", "sem": "Sem."})


def table3_4(mine=False):
    labels = data.bridge_labels()
    rows = []
    for method, base, ejb, bridge in TABLE3:
        row = {"Method": method, "Base model": base,
               "EJB": metrics.per_edit_type(pd.read_parquet(_path("ejb", ejb, mine)))["overall"]}
        for b, col in (("gab", "GenAI-Bench"), ("erb", "EditRewardBench")):
            row[col] = metrics.pairwise_accuracy(pd.read_parquet(_path(b, bridge, mine)), labels[b], b)
        rows.append(row)
    t3 = pd.DataFrame(rows)
    for c in ("EJB", "GenAI-Bench", "EditRewardBench"):
        t3[f"# {c}"] = t3[c].rank(ascending=False)
    pairs = [("EJB", "GenAI-Bench"), ("EJB", "EditRewardBench"), ("GenAI-Bench", "EditRewardBench")]
    t4 = pd.DataFrame([{"Benchmark pair": f"{a} vs. {b}", "Spearman rho": spearmanr(t3[a], t3[b])[0]}
                       for a, b in pairs])
    return t3, t4


def table5(mine=False):
    rows = []
    for model, first, last in TABLE5:
        for where, name in (("Before text", first), ("After text", last)):
            rows.append({"Model": model, "Images": where,
                         **metrics.per_edit_type(pd.read_parquet(_path("ejb", name, mine)))})
    return pd.DataFrame(rows).rename(columns=COLS)


# ---------------------------------------------------------------------------
# Fig. 5: no-edit counterfactuals are the main failure mode, driven by small edits
# ---------------------------------------------------------------------------

NEG_LABEL = {"counterfactual": "Directional", "wrong_object": "Wrong-\nObject",
             "cross_type": "Cross-Edit-\nType", "no_edit": "No\nedit"}
FOCUS = {"scale": "#2a78d6", "shape": "#eb6834", "articulation": "#1baf7a"}
MAG = {"scale": (100, "%", "Scale change |s − 1|"), "shape": (100, "%", "Parameter change"),
       "articulation": (1, "°", "Door rotation")}
SHORT = {"removal": "Rem.", "material": "Mat.", "lighting": "Light.", "camera": "Cam.", "movement": "Move.",
         "scale": "Scale", "shape": "Shape", "articulation": "Artic.", "rotation": "Rot."}


def _auroc_vs(df, neg_type, ids=None):
    if ids is not None:
        df = df[df["sample_id"].isin(ids)]
    m = (df["example_type"] == "positive") | (df["negative_type"] == neg_type)
    return metrics.auroc(df.loc[m, "ground_truth"], df.loc[m, "score_norm"])


def _magnitude() -> pd.Series:
    ref = data.load_benchmark().set_index("sample_id")
    mag = pd.Series(np.nan, index=ref.index)
    s = ref["edit_type"] == "scale"
    mag[s] = (ref.loc[s, "meta.factor"].astype(float) - 1).abs()
    s = ref["edit_type"] == "shape"
    old, new = ref.loc[s, "params.original_value"].astype(float), ref.loc[s, "params.new_value"].astype(float)
    mag[s] = (new - old).abs() / old.abs()
    s = ref["edit_type"] == "articulation"
    mag[s] = np.degrees(ref.loc[s, "meta.delta"].astype(float).abs())
    return mag, ref["edit_type"]


def figure5(out: Path, mine=False):
    import matplotlib.pyplot as plt
    runs = [pd.read_parquet(_path("ejb", name, mine)) for _, _, name in TABLE1]
    a = pd.DataFrame([{nt: _auroc_vs(df, nt) for nt in NEG_LABEL} for df in runs])
    b_rows = []
    for df in runs:
        per = {et: _auroc_vs(g, "no_edit") for et, g in df.groupby("edit_type")}
        per["material"] = np.nanmean([per.get(t, np.nan) for t in metrics.MATERIAL])
        b_rows.append({c: per.get(c, np.nan) for c in metrics.EDIT_TYPES})
    b = pd.DataFrame(b_rows)
    mag, etype = _magnitude()
    c_rows = []
    for et in FOCUS:
        m = mag[etype == et]
        bins = pd.qcut(m, 4, labels=False)
        for q in range(4):
            ids = set(bins[bins == q].index)
            vals = [_auroc_vs(df[df["edit_type"] == et], "no_edit", ids) for df in runs]
            c_rows.append({"edit_type": et, "bin": q, "lo": m[bins == q].min(), "hi": m[bins == q].max(),
                           "mean": np.nanmean(vals), "p25": np.nanpercentile(vals, 25),
                           "p75": np.nanpercentile(vals, 75)})
    c = pd.DataFrame(c_rows)

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
    fig = plt.figure(figsize=(5.5, 3.6), layout="constrained")
    gs = fig.add_gridspec(2, 1)
    top, bot = gs[0].subgridspec(1, 2, width_ratios=[1.2, 2.0]), gs[1].subgridspec(1, 3)
    ax_a = fig.add_subplot(top[0])
    ax_b = fig.add_subplot(top[1], sharey=ax_a)
    ax_c = [fig.add_subplot(bot[k], sharey=ax_a) for k in range(3)]
    rng = np.random.default_rng(0)

    def boxes(ax, d, colors):
        vals = [d[col].dropna().values for col in d.columns]
        bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "k"})
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.35)
        for i, (v, col) in enumerate(zip(vals, colors), start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=7, color=col, zorder=3)
        ax.axhline(50, color="grey", lw=0.8, ls="--")

    boxes(ax_a, a, ["#e34948" if nt == "no_edit" else "#a3a29d" for nt in a.columns])
    ax_a.set_xticks(range(1, 5), [NEG_LABEL[nt] for nt in a.columns], fontsize=6.5)
    ax_a.set_ylabel("AUROC")
    ax_a.set_title("(a) By negative type", fontsize=8.5, loc="left")
    order = list(b.median().sort_values().index)
    boxes(ax_b, b[order], [FOCUS.get(et, "#a3a29d") for et in order])
    ax_b.set_xticks(range(1, 10), [SHORT[et] for et in order], fontsize=6.5)
    ax_b.set_title("(b) No-edit only, by edit type", fontsize=8.5, loc="left")
    for k, (ax, et) in enumerate(zip(ax_c, [et for et in order if et in FOCUS])):
        g = c[c["edit_type"] == et].sort_values("bin")
        x = g["bin"].values + 1
        ax.fill_between(x, g["p25"], g["p75"], color=FOCUS[et], alpha=0.18, lw=0)
        ax.plot(x, g["mean"], color=FOCUS[et], lw=1.6, marker="o", ms=4)
        scale, unit, label = MAG[et]
        ax.set_xticks(x, [f"{scale * lo:.0f}–{scale * hi:.0f}{unit}" for lo, hi in zip(g["lo"], g["hi"])],
                      fontsize=6)
        ax.set_xlabel(label, fontsize=7)
        ax.axhline(50, color="grey", lw=0.8, ls="--")
        ax.text(0.04, 0.95, et.capitalize(), transform=ax.transAxes, va="top", color=FOCUS[et], fontweight="bold")
        if k == 0:
            ax.set_title("(c) No-edit only, by edit size", fontsize=8.5, loc="left")
    ax_a.set_ylim(0, 100)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    return a, b, c


def main(mine=False):
    out = OUTPUTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    fmt = lambda t: t.to_string(index=False, float_format="{:.1f}".format)  # noqa: E731
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        t1 = table1(mine)
        print("Table 1: AUROC per edit type, Overall, MC accuracy\n" + fmt(t1))
        t2 = table2(mine)
        print("\nTable 2: AUROC per counterfactual type\n" + fmt(t2))
        t3, t4 = table3_4(mine)
        print("\nTable 3: EditJudge-Bench AUROC vs. human-preference accuracy (# = rank)\n" + fmt(t3))
        print("\nTable 4: Spearman rank correlation\n" + t4.to_string(index=False, float_format="{:.2f}".format))
        t5 = table5(mine)
        print("\nTable 5: image placement (VIE-score)\n" + fmt(t5))
        a, b, _ = figure5(out / "figure5.pdf", mine)
    for name, t in (("table1", t1), ("table2", t2), ("table3", t3), ("table4", t4), ("table5", t5)):
        t.to_csv(out / f"{name}.csv", index=False)
    print(f"\nFig. 5 median AUROC by counterfactual type: "
          + ", ".join(f"{k}={v:.1f}" for k, v in a.median().items()))
    print(f"Wrote {out}/table{{1..5}}.csv and figure5.pdf")
