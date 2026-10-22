"""Replay a station's real record with different sampling policies.

The station always measures at the top of each hour. During an "intensive" hour it also measures at :15, :30
and :45. A policy decides, at each top of the hour and from the hourly measurements only, whether the coming
hour is intensive. The truth is the source series on a 5-minute grid.
"""
import numpy as np

STEP = 300                 # 5-minute grid, in seconds
PER_HOUR = 12              # grid points per hour
INTENSIVE = (0, 3, 6, 9)   # offsets measured during an intensive hour: :00, :15, :30, :45
MAX_GAP = 12 * 3600        # the source stores a value only when the level changes; longer gaps are missing data
MERGE = 24 * PER_HOUR      # two exceedances less than 24 h apart are one flood
QUANTILE = 0.98            # alert threshold: the daily maximum exceeded on 2 % of days


def to_grid(t: np.ndarray, h: np.ndarray) -> tuple[int, np.ndarray]:
    """Start time and heights on a 5-minute grid starting on the hour, linear between source points."""
    t0 = int(-(-t[0] // 3600) * 3600)
    n = (int(t[-1]) - t0) // 3600 * PER_HOUR
    g = t0 + STEP * np.arange(n)
    x = np.interp(g, t, h).astype(np.float32)
    after = np.searchsorted(t, g)  # index of the first source point at or after each grid time
    gap = t[np.clip(after, 0, len(t) - 1)] - t[np.clip(after - 1, 0, len(t) - 1)]
    x[gap > MAX_GAP] = np.nan
    return t0, x


def threshold(x: np.ndarray) -> tuple[float, float]:
    """(median level, alert threshold) of a station, from its whole record."""
    days = x[: len(x) // 288 * 288].reshape(-1, 288)
    full = ~np.isnan(days).any(1)
    return float(np.nanmedian(x)), float(np.quantile(days[full].max(1), QUANTILE))


def hourly(x: np.ndarray) -> np.ndarray:
    return x[::PER_HOUR]


def measured(intensive: np.ndarray) -> np.ndarray:
    """Grid indices the station measures, given the intensive flag of each hour."""
    idx = PER_HOUR * np.arange(len(intensive))
    extra = (PER_HOUR * np.flatnonzero(intensive))[:, None] + np.array(INTENSIVE[1:])
    return np.sort(np.concatenate([idx, extra.ravel()]))


def events(x: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """(first index at or above thr, last index) of each flood, from the 5-minute truth."""
    above = np.flatnonzero(x >= thr)
    if len(above) == 0:
        return []
    cut = np.flatnonzero(np.diff(above) > MERGE)
    starts, ends = above[np.r_[0, cut + 1]], above[np.r_[cut, len(above) - 1]]
    # A flood must start after at least 6 h of valid data below the threshold, so the policy had a chance.
    return [(int(s), int(e)) for s, e in zip(starts, ends)
            if s >= 6 * PER_HOUR and not np.isnan(x[s - 6 * PER_HOUR:s]).any()]


def score(x: np.ndarray, thr: float, floods: list[tuple[int, int]], intensive: np.ndarray) -> dict:
    """Measurements per year, and for each flood the alert delay and the part of the peak the station missed."""
    m = measured(intensive)
    hours = len(intensive)
    rows = []
    for s, e in floods:
        inside = m[(m >= s) & (m <= e)]
        hit = inside[x[inside] >= thr]
        peak = float(np.nanmax(x[s:e + 1]))
        seen = float(np.nanmax(x[inside])) if len(inside) else float("nan")
        rows.append({"delay_min": None if len(hit) == 0 else int((hit[0] - s) * STEP // 60),
                     "peak_miss_cm": round(100 * (peak - seen), 1) if len(inside) else None})
    return {"per_year": len(m) / (hours / 8766), "intensive_share": float(intensive.mean()), "floods": rows}


def features(h1: np.ndarray, base: float, thr: float, lags: int = 6) -> np.ndarray:
    """Last `lags` hourly measurements, scaled so 0 = median level and 1 = alert threshold. Row k = hour k."""
    z = (h1 - base) / (thr - base)
    pad = np.concatenate([np.full(lags - 1, np.nan, np.float32), z])
    return np.stack([pad[lags - 1 - j: len(pad) - j] for j in range(lags)], 1).astype(np.float32)


def target(x: np.ndarray, thr: float, n_hours: int, ahead: int = 3) -> np.ndarray:
    """1 if the truth reaches the threshold within the next `ahead` hours."""
    above = np.r_[x >= thr, np.zeros(ahead * PER_HOUR, bool)]
    run = np.convolve(above, np.ones(ahead * PER_HOUR, int))[ahead * PER_HOUR - 1:]  # run[i] = sum of above[i:i+w]
    return (run[PER_HOUR * np.arange(n_hours) + 1] > 0).astype(np.float32)


# Rule policies. Each returns the intensive flag of every hour from the hourly measurements h1.
def fixed(h1: np.ndarray, every_15: bool) -> np.ndarray:
    return np.full(len(h1), every_15)


def level(h1: np.ndarray, base: float, thr: float, beta: float) -> np.ndarray:
    """Intensive once the level passes base + beta x (thr - base). beta = 1: at the threshold itself."""
    return np.nan_to_num(h1, nan=-np.inf) >= base + beta * (thr - base)


def rise(h1: np.ndarray, base: float, thr: float, alpha: float) -> np.ndarray:
    """Intensive at the threshold, or when the last hour rose by more than alpha x (thr - base)."""
    d = np.r_[0, np.diff(h1)]
    return level(h1, base, thr, 1.0) | (np.nan_to_num(d, nan=0) >= alpha * (thr - base))
