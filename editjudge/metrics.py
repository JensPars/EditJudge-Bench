"""Paper metrics.

EditJudge-Bench (1,382 pairs)
    AUROC per edit type, over all counterfactuals. "Material" is the mean of
    material_slot and object_material; "Overall" is the mean of the nine edit types.
    Per counterfactual type: positives vs one counterfactual type. "Sem." is the
    mean of directional, cross-type and wrong-object.
    MC accuracy: per pair, the factual instruction must score highest among its
    candidates; a tie for first among k candidates gives 1/k.
GenAI-Bench / EditReward-Bench
    Pairwise accuracy on pairs with a strict human preference; a judge tie or an
    unparsed score counts 1/2.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EDIT_TYPES = ["removal", "material", "lighting", "camera", "movement", "scale", "shape", "articulation", "rotation"]
MATERIAL = ("material_slot", "object_material")
NEG_TYPES = {"dir": "counterfactual", "cross": "cross_type", "wrong_obj": "wrong_object", "no_edit": "no_edit"}


def auroc(y, s) -> float:
    y, s = np.asarray(y, int), np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    return 100 * roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def per_edit_type(df: pd.DataFrame) -> dict:
    per = {t: auroc(g["ground_truth"], g["score_norm"]) for t, g in df.groupby("edit_type")}
    per["material"] = np.nanmean([per.get(t, np.nan) for t in MATERIAL])
    row = {t: per.get(t, np.nan) for t in EDIT_TYPES}
    row["overall"] = np.nanmean(list(row.values()))
    return row


def per_negative_type(df: pd.DataFrame) -> dict:
    pos = df["example_type"] == "positive"
    row = {k: auroc(df.loc[pos | (df["negative_type"] == v), "ground_truth"],
                    df.loc[pos | (df["negative_type"] == v), "score_norm"]) for k, v in NEG_TYPES.items()}
    row["sem"] = np.mean([row["dir"], row["cross"], row["wrong_obj"]])
    return row


def mc_accuracy(df: pd.DataFrame) -> float:
    s = df["score_norm"].astype(float)
    is_pos = df["example_type"] == "positive"
    pos = s[is_pos].groupby(df.loc[is_pos, "sample_id"]).first()
    neg = df[~is_pos & np.isfinite(s)]
    neg_s = neg["score_norm"].astype(float)
    neg_max = neg_s.groupby(neg["sample_id"]).max().reindex(pos.index).fillna(-np.inf)
    ties = pd.Series(neg_s.values == pos.reindex(neg["sample_id"]).values, index=neg["sample_id"].values)
    n_ties = ties.groupby(level=0).sum().reindex(pos.index).fillna(0)
    return 100 * float(np.where(pos >= neg_max, 1 / (1 + n_ties), 0.0).mean())


def pairwise_accuracy(scores: pd.DataFrame, labels: pd.DataFrame, benchmark: str) -> float:
    d = scores[["sample_id", "candidate_idx", "score_norm"]].merge(labels, on=["sample_id", "candidate_idx"],
                                                                   validate="one_to_one")
    s = d.pivot(index="sample_id", columns="candidate_idx", values="score_norm").astype(float)
    model = np.sign(s[0] - s[1]).fillna(0)
    if benchmark == "gab":
        vote = d.groupby("sample_id")["vote_type"].first()
        human, keep = vote.map({"leftvote": 1, "rightvote": -1}), vote.isin(["leftvote", "rightvote"])
    else:
        h = d.pivot(index="sample_id", columns="candidate_idx", values="human_score")
        human = np.sign(h[0] - h[1])
        keep = human != 0
    m, h = model[keep], human[keep]
    return 100 * float(((m == h) + 0.5 * (m == 0)).mean())


def headline(benchmark: str, path: Path) -> float:
    """Overall AUROC (EditJudge-Bench) or pairwise accuracy (GenAI-Bench, EditReward-Bench)."""
    df = pd.read_parquet(path)
    if benchmark == "ejb":
        return per_edit_type(df)["overall"]
    from editjudge.data import bridge_labels
    return pairwise_accuracy(df, bridge_labels()[benchmark], benchmark)


def compare(cfg: dict, out: Path):
    """Print the re-run next to the saved paper run and the paper value."""
    from editjudge.data import PREDICTIONS
    saved = PREDICTIONS / cfg["benchmark"] / f"{cfg['name']}.parquet"
    metric = "Overall AUROC" if cfg["benchmark"] == "ejb" else "pairwise accuracy"
    print(f"\n{cfg['name']} ({cfg['benchmark']}, {metric})")
    print(f"  paper     {cfg['paper']:.1f}")
    if saved.exists():
        print(f"  saved run {headline(cfg['benchmark'], saved):.1f}")
    print(f"  your run  {headline(cfg['benchmark'], out):.1f}")
