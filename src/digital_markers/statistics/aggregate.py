import numpy as np
import pandas as pd

def summarize_series(values: pd.Series, *, prefix: str, stats: tuple[str, ...] = ("mean", "std", "ac1", "slope")) -> dict[str, float]:
    clean = pd.Series(values, dtype="float64").dropna()
    result: dict[str, float] = {}
    if "mean" in stats:
        result[f"mean_{prefix}"] = float(clean.mean()) if len(clean) else np.nan
    if "std" in stats:
        result[f"std_{prefix}"] = float(clean.std()) if len(clean) > 1 else np.nan
    if "ac1" in stats:
        result[f"ac1_{prefix}"] = float(clean.autocorr(lag=1)) if len(clean) > 2 else np.nan
    if "slope" in stats:
        if len(clean) > 1:
            x = np.arange(len(clean), dtype=float)
            result[f"slope_{prefix}"] = float(np.polyfit(x, clean.to_numpy(), 1)[0])
        else:
            result[f"slope_{prefix}"] = np.nan
    return result
