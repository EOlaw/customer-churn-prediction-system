"""Pure metric functions: thresholded classification, review capacity, calibration, subgroups.

Everything here is a deterministic function of (labels, scores, fixed parameters). Thresholds are
inputs, never derived inside these functions from the data being evaluated, except for
``best_f1_threshold`` which callers must run on the calibration partition only.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

BOOTSTRAP_RESAMPLES = 1000
MIN_SUBGROUP_ROWS = 50
MIN_SUBGROUP_CLASS = 10  # positives and negatives, each
MIN_BIN_ROWS = 20


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flagged = p >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "flagged_share": float(flagged.mean()) if len(y) else None}


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing F1 over the observed scores; ties go to the higher threshold."""
    best, best_f1 = None, -1.0
    for candidate in np.unique(p):
        f1 = threshold_metrics(y, p, float(candidate))["f1"] or 0.0
        if f1 >= best_f1:
            best, best_f1 = float(candidate), f1
    if best is None:
        raise ValueError("No scores to choose a threshold from")
    return best


def capacity_metrics(y: np.ndarray, p: np.ndarray, capacity: float,
                     tie_break: np.ndarray | None = None) -> dict:
    """Review the top ceil(capacity * n) customers by score. Ties are broken by ``tie_break``
    (descending), then by row position, so the result is deterministic."""
    n = len(y)
    k = max(1, math.ceil(capacity * n))
    secondary = tie_break if tie_break is not None else np.zeros(n)
    order = np.lexsort((np.arange(n), -secondary, -p))
    hits = int(y[order[:k]].sum())
    positives = int(y.sum())
    prevalence = positives / n
    precision = hits / k
    return {"capacity_share": capacity, "reviewed": k, "true_positives_found": hits,
            "recall": hits / positives if positives else None, "precision": precision,
            "lift": precision / prevalence if prevalence else None}


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    index = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        positives = int(y[mask].sum())
        interval = wilson(positives, n)
        rows.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "rows": n,
                     "mean_predicted": float(p[mask].mean()) if n else None,
                     "observed_rate": positives / n if n else None,
                     "observed_ci95_low": interval[0] if interval else None,
                     "observed_ci95_high": interval[1] if interval else None,
                     "sparse": n < MIN_BIN_ROWS})
    return rows


def calibration_summary(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, p))
    baseline = prevalence * (1 - prevalence)  # Brier of always predicting the base rate
    table = reliability_table(y, p, bins)
    ece = sum(r["rows"] * abs(r["mean_predicted"] - r["observed_rate"])
              for r in table if r["rows"]) / len(y)
    slope = None
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    if np.ptp(logit) > 1e-9:
        slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(
            logit.reshape(-1, 1), y).coef_[0, 0])
    return {"brier": brier, "brier_prevalence_baseline": baseline,
            "brier_skill_score": 1 - brier / baseline if baseline else None,
            "expected_calibration_error": float(ece), "mean_predicted": float(p.mean()),
            "observed_prevalence": prevalence,
            "calibration_in_the_large": float(p.mean() - prevalence),
            "calibration_slope": slope, "bins": bins, "reliability": table}


def auc_score(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def pr_auc_score(y, p):
    return float(average_precision_score(y, p)) if y.sum() else None


def bootstrap_ci(y: np.ndarray, p: np.ndarray, statistic, seed: int,
                 resamples: int = BOOTSTRAP_RESAMPLES) -> list[float] | None:
    """Percentile 95% interval by resampling rows. Resamples missing a class are skipped."""
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(resamples):
        idx = rng.randint(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            value = statistic(y[idx], p[idx])
            if value is not None:
                values.append(value)
    if len(values) < resamples // 2:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def paired_bootstrap_difference(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, statistic,
                                seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> dict | None:
    """statistic(a) - statistic(b) on the same resampled rows, with a percentile 95% interval."""
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(resamples):
        idx = rng.randint(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            a, b = statistic(y[idx], p_a[idx]), statistic(y[idx], p_b[idx])
            if a is not None and b is not None:
                diffs.append(a - b)
    if len(diffs) < resamples // 2:
        return None
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(statistic(y, p_a) - statistic(y, p_b)),
            "ci95": [float(low), float(high)]}


def heldout_metrics(y: np.ndarray, p: np.ndarray, threshold: float, capacity: float,
                    tie_break: np.ndarray, seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Headline held-out metrics with bootstrap intervals; threshold and capacity are fixed inputs."""
    at_threshold = threshold_metrics(y, p, threshold)
    at_capacity = capacity_metrics(y, p, capacity, tie_break)
    statistics = {
        "roc_auc": auc_score, "pr_auc": pr_auc_score,
        "brier": lambda yy, pp: float(brier_score_loss(yy, pp)),
        "precision": lambda yy, pp: threshold_metrics(yy, pp, threshold)["precision"],
        "recall": lambda yy, pp: threshold_metrics(yy, pp, threshold)["recall"],
        "f1": lambda yy, pp: threshold_metrics(yy, pp, threshold)["f1"],
        "recall_at_capacity": lambda yy, pp: capacity_metrics(yy, pp, capacity)["recall"],
    }
    intervals = {name: bootstrap_ci(y, p, fn, seed, resamples) for name, fn in statistics.items()}
    return {
        "rows": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "roc_auc": auc_score(y, p), "pr_auc": pr_auc_score(y, p),
        "reference": {"roc_auc": 0.5, "pr_auc": float(y.mean()),
                      "note": "no-skill baselines computed from test labels only, no model fitted"},
        "at_threshold": {**at_threshold, "confusion_matrix": {
            "true_negative": at_threshold["tn"], "false_positive": at_threshold["fp"],
            "false_negative": at_threshold["fn"], "true_positive": at_threshold["tp"]}},
        "at_review_capacity": at_capacity,
        "calibration": calibration_summary(y, p),
        "confidence_intervals_95": intervals,
        "interval_method": f"percentile bootstrap, {resamples} row resamples, seed {seed}",
    }


def subgroup_report(y: np.ndarray, p: np.ndarray, groups: np.ndarray, threshold: float,
                    seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> list[dict]:
    """Per-level metrics at the fixed threshold. A level with fewer than MIN_SUBGROUP_ROWS rows
    or fewer than MIN_SUBGROUP_CLASS of either class reports counts only: 'insufficient_data'."""
    out = []
    for level in sorted(set(groups.tolist()), key=str):
        mask = groups == level
        yy, pp = y[mask], p[mask]
        n, positives = len(yy), int(yy.sum())
        negatives = n - positives
        base = threshold_metrics(yy, pp, threshold)
        entry = {"group": str(level), "rows": n, "positives": positives,
                 "prevalence": positives / n, "selection_rate": base["flagged_share"]}
        reasons = []
        if n < MIN_SUBGROUP_ROWS:
            reasons.append(f"fewer than {MIN_SUBGROUP_ROWS} rows")
        if positives < MIN_SUBGROUP_CLASS:
            reasons.append(f"fewer than {MIN_SUBGROUP_CLASS} positives")
        if negatives < MIN_SUBGROUP_CLASS:
            reasons.append(f"fewer than {MIN_SUBGROUP_CLASS} negatives")
        if reasons:
            entry.update(status="insufficient_data", reason="; ".join(reasons), metrics=None)
        else:
            entry.update(status="ok", reason=None, metrics={
                "roc_auc": auc_score(yy, pp), "pr_auc": pr_auc_score(yy, pp),
                "brier": float(brier_score_loss(yy, pp)),
                "precision": base["precision"], "recall": base["recall"], "f1": base["f1"],
                "confidence_intervals_95": {
                    "roc_auc": bootstrap_ci(yy, pp, auc_score, seed + 1, resamples),
                    "recall": bootstrap_ci(
                        yy, pp, lambda a, b: threshold_metrics(a, b, threshold)["recall"],
                        seed + 1, resamples)}})
        out.append(entry)
    return out
