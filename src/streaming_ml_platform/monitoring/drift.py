from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected_bins = pd.qcut(expected.rank(method="first"), q=bins, duplicates="drop")
    actual_bins = pd.qcut(actual.rank(method="first"), q=bins, duplicates="drop")
    e = expected_bins.value_counts(normalize=True).sort_index()
    a = actual_bins.value_counts(normalize=True).sort_index()
    aligned = e.to_frame("e").join(a.to_frame("a"), how="outer").fillna(1e-6)
    return float(((aligned["a"] - aligned["e"]) * (aligned["a"] / aligned["e"]).apply(lambda x: np.log(x))).sum())


def jensen_shannon_divergence(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e_hist, bin_edges = np.histogram(expected, bins=bins, density=True)
    a_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
    e = np.clip(e_hist / (e_hist.sum() + 1e-8), 1e-8, None)
    a = np.clip(a_hist / (a_hist.sum() + 1e-8), 1e-8, None)
    m = 0.5 * (e + a)
    kl_em = np.sum(e * np.log(e / m))
    kl_am = np.sum(a * np.log(a / m))
    return float(0.5 * (kl_em + kl_am))


def compute_drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue
        ref = reference[column].dropna()
        cur = current[column].dropna()
        if ref.empty or cur.empty:
            continue
        report[column] = {
            "psi": population_stability_index(ref, cur),
            "jsd": jensen_shannon_divergence(ref, cur),
            "reference_mean": float(ref.mean()),
            "current_mean": float(cur.mean()),
        }
    return report
