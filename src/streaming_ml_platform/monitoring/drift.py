import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected_bins = pd.qcut(expected.rank(method="first"), q=bins, duplicates="drop")
    actual_bins = pd.qcut(actual.rank(method="first"), q=bins, duplicates="drop")
    e = expected_bins.value_counts(normalize=True).sort_index()
    a = actual_bins.value_counts(normalize=True).sort_index()
    aligned = e.to_frame("e").join(a.to_frame("a"), how="outer").fillna(1e-6)
    return float(((aligned["a"] - aligned["e"]) * (aligned["a"] / aligned["e"]).apply(lambda x: np.log(x))).sum())
