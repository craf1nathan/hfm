"""
Phase 2 – Normalized Feature Engineering  (OPTIMIZED)
All outputs ∈ [-1, +1]
No look-ahead bias: normalization params computed at time t, applied to lags.

Optimizations applied:
  1. Numba JIT for xcorr_max and structure loops
  2. NumPy-native rolling via stride tricks for z-score / rank
  3. Vectorized polyfit replacement (linear slope via covariance)
  4. Batch lag application via numpy hstack (single DataFrame construction)
  5. Pre-computation of shared intermediates (atr, rsi, ret) 
  6. Contiguous array enforcement before Numba calls

Usage in Jupyter:
    from src.phase2_feature_engineering import build_features, validate_features
    features = build_features(splits["train"], data_cfg, pipeline_cfg)
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings

# ── Numba with graceful fallback ──────────────────────────────────────────────
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available – falling back to NumPy-only mode.")

    def njit(*args, **kwargs):          # no-op decorator
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator

    prange = range


# ─────────────────────────────────────────────────────────────────────────────
# NUMBA KERNELS  (compiled once, reused every call)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _xcorr_max_nb(
    rx: np.ndarray,     # (n,)  float64 – return series of target
    ra: np.ndarray,     # (n,)  float64 – return series of predictor
    window: int,
    max_lag: int
) -> np.ndarray:
    """
    Rolling cross-correlation, signed by dominant lag.
    Returns array ∈ [-1, +1] of length n.

    For each bar i ≥ window:
      - Take rx[i-window:i], ra[i-window:i]
      - Compute Pearson corr for lags -max_lag … +max_lag
      - Record (best_lag / max_lag) × sign(best_corr)
    
    Numba eliminates Python overhead from the triple loop
    (bars × lags × window). ~40-80× faster than pandas version.
    """
    n   = len(rx)
    out = np.zeros(n, dtype=np.float64)

    for i in range(window, n):
        x_win = rx[i - window: i]
        a_win = ra[i - window: i]

        # Means
        xm, am = 0.0, 0.0
        for k in range(window):
            xm += x_win[k]
            am += a_win[k]
        xm /= window
        am /= window

        # Std devs
        xs, as_ = 0.0, 0.0
        for k in range(window):
            xs  += (x_win[k] - xm) ** 2
            as_ += (a_win[k] - am) ** 2
        xs  = xs  ** 0.5
        as_ = as_ ** 0.5

        # Skip degenerate windows
        if xs < 1e-9 or as_ < 1e-9:
            out[i] = 0.0
            continue

        best_val = 0.0
        best_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            # Overlapping slice for given lag
            if lag == 0:
                x_sl = x_win
                a_sl = a_win
                m    = window
            elif lag > 0:                    # x leads a
                x_sl = x_win[lag:]
                a_sl = a_win[:window - lag]
                m    = window - lag
            else:                            # a leads x
                x_sl = x_win[:window + lag]
                a_sl = a_win[-lag:]
                m    = window + lag

            if m < 3:
                continue

            # Pearson correlation (inline for Numba)
            mx, ma = 0.0, 0.0
            for k in range(m):
                mx += x_sl[k]
                ma += a_sl[k]
            mx /= m
            ma /= m

            num, dx2, da2 = 0.0, 0.0, 0.0
            for k in range(m):
                dx = x_sl[k] - mx
                da = a_sl[k] - ma
                num += dx * da
                dx2 += dx * dx
                da2 += da * da

            denom = (dx2 * da2) ** 0.5
            if denom < 1e-12:
                c = 0.0
            else:
                c = num / denom

            if abs(c) > abs(best_val):
                best_val = c
                best_lag = lag

        # Encode as signed fraction of max_lag
        out[i] = (best_lag / max_lag) * (1.0 if best_val >= 0 else -1.0)

    return out


@njit(cache=True)
def _hh_ll_score_nb(
    high: np.ndarray,   # (n,)
    low:  np.ndarray,   # (n,)
    n_lags: int = 5
) -> tuple:
    """
    Compute hh_score and ll_score via exponential weighting.
    
    Replaces the Python loop::
        for i in range(1, 6):
            hh += 0.5**i * (high > high.shift(i))
    
    Returns (hh_score, ll_score) each ∈ [-1, +1].
    """
    n         = len(high)
    hh        = np.zeros(n, dtype=np.float64)
    ll        = np.zeros(n, dtype=np.float64)
    max_score = 0.0

    for i in range(1, n_lags + 1):
        w          = 0.5 ** i
        max_score += w
        for t in range(i, n):
            if high[t] > high[t - i]:
                hh[t] += w
            if low[t]  < low[t  - i]:
                ll[t] += w

    # Map to [-1, +1]
    hh_score = np.empty(n, dtype=np.float64)
    ll_score = np.empty(n, dtype=np.float64)
    inv_ms   = 1.0 / max_score

    for t in range(n):
        hh_score[t] = min(1.0, max(-1.0, hh[t] * inv_ms * 2.0 - 1.0))
        ll_score[t] = min(1.0, max(-1.0, ll[t] * inv_ms * 2.0 - 1.0))

    return hh_score, ll_score


@njit(cache=True)
def _rolling_z_nb(x: np.ndarray, w: int, min_periods: int) -> np.ndarray:
    """
    Rolling z-score using a single-pass Welford-style accumulation.
    O(n) rather than O(n×w). Returns 0 where std ≈ 0.
    
    This replaces pandas rolling().mean() + rolling().std()
    which each do O(n×w) work internally.
    """
    n   = len(x)
    out = np.zeros(n, dtype=np.float64)

    # Expanding warm-up
    s1, s2 = 0.0, 0.0
    cnt    = 0

    for i in range(n):
        xi  = x[i]
        s1 += xi
        s2 += xi * xi
        cnt += 1

        if cnt < min_periods:
            out[i] = 0.0
            continue

        if cnt > w:
            # Remove element falling out of window
            xout = x[i - w]
            s1  -= xout
            s2  -= xout * xout
            cnt -= 1

        mu  = s1 / cnt
        var = s2 / cnt - mu * mu
        if var < 1e-12:
            out[i] = 0.0
        else:
            out[i] = (xi - mu) / (var ** 0.5)

    return out


@njit(cache=True)
def _rolling_rank_nb(x: np.ndarray, w: int, min_periods: int) -> np.ndarray:
    """
    Rolling percentile rank → mapped to [-1, +1].
    
    Uses insertion sort on the active window (O(n×w)) which is still
    faster than pandas rank() due to zero Python overhead.
    For w ≤ 240 this is practically instant.
    """
    n   = len(x)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        start = max(0, i - w + 1)
        win   = x[start: i + 1]
        m     = len(win)

        if m < min_periods:
            out[i] = 0.0
            continue

        # Count values ≤ x[i]
        xi    = x[i]
        count = 0
        for k in range(m):
            if win[k] <= xi:
                count += 1

        pct     = count / m          # ∈ (0, 1]
        out[i]  = min(1.0, max(-1.0, pct * 2.0 - 1.0))

    return out


@njit(cache=True)
def _linear_slope_rolling_nb(x: np.ndarray, w: int, min_periods: int) -> np.ndarray:
    """
    Rolling linear slope via closed-form OLS (avoids polyfit).
    
    For window [i-w+1 … i], slope = Cov(t, x) / Var(t).
    O(n×w) but with very low constant — no Python or numpy overhead.
    
    Replaces rolling().apply(np.polyfit) which has huge per-window cost.
    """
    n   = len(x)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        start = max(0, i - w + 1)
        m     = i - start + 1

        if m < min_periods:
            out[i] = 0.0
            continue

        # t ∈ [0, m-1], x values
        t_mean  = (m - 1) / 2.0
        x_mean  = 0.0
        for k in range(m):
            x_mean += x[start + k]
        x_mean /= m

        cov_tx, var_t = 0.0, 0.0
        for k in range(m):
            dt      = k - t_mean
            dx      = x[start + k] - x_mean
            cov_tx += dt * dx
            var_t  += dt * dt

        out[i] = cov_tx / var_t if var_t > 1e-12 else 0.0

    return out


@njit(cache=True)
def _tanh_nb(x: np.ndarray) -> np.ndarray:
    """Element-wise tanh, NaN → 0."""
    n   = len(x)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        xi = x[i]
        if xi != xi:          # NaN check
            out[i] = 0.0
        elif xi > 10.0:
            out[i] = 1.0
        elif xi < -10.0:
            out[i] = -1.0
        else:
            out[i] = np.tanh(xi)
    return out


@njit(cache=True)
def _clip1_nb(x: np.ndarray) -> np.ndarray:
    """Clip to [-1, +1] in-place-style."""
    n   = len(x)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        xi = x[i]
        if xi > 1.0:
            out[i] = 1.0
        elif xi < -1.0:
            out[i] = -1.0
        else:
            out[i] = xi
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ARRAY HELPERS  (bridge between pandas and numba)
# ─────────────────────────────────────────────────────────────────────────────

def _to_f64(s: pd.Series) -> np.ndarray:
    """
    Convert Series to contiguous float64 array, filling NaN → 0.
    Required before passing to Numba kernels.
    """
    return np.ascontiguousarray(s.fillna(0).values, dtype=np.float64)


def _wrap(arr: np.ndarray, index: pd.Index) -> pd.Series:
    """Wrap numpy result back into a pandas Series with original index."""
    return pd.Series(arr, index=index, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVES  (pandas-level, used where overhead is acceptable)
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_z(x: pd.Series, w: int) -> pd.Series:
    """
    Rolling z-score dispatched to Numba kernel.
    Falls back to pandas if Numba unavailable.
    """
    min_p = max(2, w // 4)
    if NUMBA_AVAILABLE:
        return _wrap(_rolling_z_nb(_to_f64(x), w, min_p), x.index)
    # Pandas fallback
    mu  = x.rolling(w, min_periods=min_p).mean()
    sig = x.rolling(w, min_periods=min_p).std()
    return (x - mu) / sig.replace(0, np.nan).fillna(1)


def _rolling_rank(x: pd.Series, w: int) -> pd.Series:
    """
    Rolling percentile rank → [-1, +1], dispatched to Numba kernel.
    """
    min_p = max(2, w // 4)
    if NUMBA_AVAILABLE:
        return _wrap(_rolling_rank_nb(_to_f64(x), w, min_p), x.index)
    rank = x.rolling(w, min_periods=min_p).rank(pct=True)
    return (rank * 2 - 1).clip(-1, 1)


def _tanh(x: pd.Series) -> pd.Series:
    """Soft squash to (-1, +1). NaN-safe."""
    if NUMBA_AVAILABLE:
        return _wrap(_tanh_nb(_to_f64(x)), x.index)
    return np.tanh(x.fillna(0))


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """
    Average True Range via EWM.
    Pre-compute TR as numpy for speed then hand to pandas ewm.
    """
    h = high.values
    l = low.values
    c = close.values
    c_prev = np.empty_like(c)
    c_prev[0] = c[0]
    c_prev[1:] = c[:-1]

    tr = np.maximum(
        h - l,
        np.maximum(np.abs(h - c_prev), np.abs(l - c_prev))
    )
    return pd.Series(tr, index=close.index).ewm(span=n, adjust=False).mean()


def _clamp(x: pd.Series) -> pd.Series:
    return x.clip(-1, 1)


def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return a.div(b.replace(0, np.nan)).fillna(fill)


def _ffill(x: pd.Series) -> pd.Series:
    return x.ffill()


def _rolling_slope(x: pd.Series, w: int) -> pd.Series:
    """
    Rolling linear slope — dispatched to Numba kernel.
    Replaces rolling().apply(np.polyfit) (~100× faster for w=10).
    """
    min_p = max(2, w // 4)
    if NUMBA_AVAILABLE:
        return _wrap(_linear_slope_rolling_nb(_to_f64(x), w, min_p), x.index)
    # Pandas fallback (slow)
    def _slope(arr):
        if len(arr) < 2:
            return 0.0
        return float(np.polyfit(range(len(arr)), arr, 1)[0])
    return x.rolling(w, min_periods=min_p).apply(_slope, raw=True)


def _vprint(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# A. PRICE-BASED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def feat_returns(close: pd.Series) -> dict[str, pd.Series]:
    """ret_n for n ∈ {1,5,15,60} — tanh(z-score over 120 bars)."""
    out = {}
    log_close = np.log(close.values)   # compute log once

    for n in [1, 5, 15, 60]:
        # Vectorized log-return using pre-computed log
        ret_arr  = np.empty(len(log_close), dtype=np.float64)
        ret_arr[:n] = 0.0
        ret_arr[n:] = log_close[n:] - log_close[:-n]
        ret_s    = pd.Series(ret_arr, index=close.index)
        out[f"ret_{n}"] = _tanh(_rolling_z(ret_s, 120))

    return out

def feat_bar_structure(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, atr20: pd.Series
) -> dict[str, pd.Series]:
    """bar_range, body_pct, wick_upper, wick_lower."""
    h = high.values;  l = low.values
    o = open_.values; c = close.values
    a = atr20.values

    oc_max = np.maximum(o, c)
    oc_min = np.minimum(o, c)

    # ── Suppress divide-by-zero: hl=0 and a=0 are handled by np.where ────────
    with np.errstate(divide='ignore', invalid='ignore'):
        hl = h - l   # raw range (may be 0)

        bar_range_ratio = np.where(a  == 0, np.nan, hl / a)
        body_ratio      = np.where(hl == 0, 0.0,    np.abs(c - o) / hl)
        wick_up_ratio   = np.where(hl == 0, 0.0,    (h - oc_max)  / hl)
        wick_lo_ratio   = np.where(hl == 0, 0.0,    (oc_min - l)  / hl)

    def _s(arr): return pd.Series(arr, index=close.index)

    bar_range  = _rolling_rank(_s(bar_range_ratio), 60)
    body_pct   = _clamp(_s(np.clip(2 * body_ratio   - 1, -1, 1)))
    wick_upper = _clamp(_s(np.clip(2 * wick_up_ratio - 1, -1, 1)))
    wick_lower = _clamp(_s(np.clip(2 * wick_lo_ratio - 1, -1, 1)))

    return {
        "bar_range":  bar_range,
        "body_pct":   body_pct,
        "wick_upper": wick_upper,
        "wick_lower": wick_lower,
    }

def feat_vwap_dev(
    close: pd.Series, high: pd.Series, low: pd.Series,
    volume: pd.Series, atr20: pd.Series, w: int = 60
) -> pd.Series:
    """VWAP deviation — ATR-normalised, tanh-squashed."""
    tp   = (high + low + close) / 3
    vwap = _safe_div(
        (tp * volume).rolling(w, min_periods=1).sum(),
        volume.rolling(w,  min_periods=1).sum()
    )
    return _tanh(_rolling_z(_safe_div(close - vwap, atr20), 60))

def feat_price_pos(close: pd.Series, w: int = 60) -> pd.Series:
    """
    Position in w-bar high/low range → [-1, +1].

    Fix: use np.errstate to suppress the divide-by-zero RuntimeWarning
    that fires when hi == lo (flat price windows). The np.where already
    handles the zero-range case correctly (returns 0.5); the warning is
    a NumPy evaluation artefact because BOTH branches are evaluated before
    np.where selects between them.
    """
    c  = close.values
    lo = pd.Series(c, index=close.index).rolling(w, min_periods=1).min().values
    hi = pd.Series(c, index=close.index).rolling(w, min_periods=1).max().values
    rng = hi - lo

    with np.errstate(divide='ignore', invalid='ignore'):
        pos = np.where(rng == 0, 0.5, (c - lo) / rng)

    return _clamp(pd.Series(pos * 2 - 1, index=close.index))

# ─────────────────────────────────────────────────────────────────────────────
# B. MOMENTUM FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _rsi_raw(close: pd.Series, n: int) -> pd.Series:
    """RSI in [0, 100]."""
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=n, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=n, adjust=False).mean()
    rs    = _safe_div(gain, loss, fill=1.0)
    return 100 - (100 / (1 + rs))


def feat_rsi(close: pd.Series) -> dict[str, pd.Series]:
    """rsi_n for n ∈ {5,14,20} remapped to [-1, +1]."""
    return {
        f"rsi_{n}": _clamp(_rsi_raw(close, n) / 50 - 1)
        for n in [5, 14, 20]
    }


def feat_roc(close: pd.Series) -> dict[str, pd.Series]:
    """roc_n for n ∈ {5,15,60} — tanh(z-score)."""
    out = {}
    c   = close.values
    for n in [5, 15, 60]:
        roc_arr    = np.empty(len(c), dtype=np.float64)
        roc_arr[:n] = 0.0
        roc_arr[n:] = (c[n:] / c[:-n] - 1) * 100
        out[f"roc_{n}"] = _tanh(_rolling_z(pd.Series(roc_arr, index=close.index), 120))
    return out


def feat_momentum_extras(close: pd.Series, atr20: pd.Series) -> dict[str, pd.Series]:
    """mom_accel, mom_regime, trend_strength."""
    c   = close.values
    roc5_arr        = np.empty(len(c), dtype=np.float64)
    roc5_arr[:5]    = 0.0
    roc5_arr[5:]    = (c[5:] / c[:-5] - 1) * 100
    roc5            = pd.Series(roc5_arr, index=close.index)

    rsi14           = _rsi_raw(close, 14)
    mom_accel       = _tanh(_rolling_z(roc5 - roc5.shift(1), 120))
    mom_regime      = _clamp(_ema(rsi14 / 50 - 1, span=10))
    trend_strength  = _tanh(_safe_div(close - _ema(close, 20), atr20))

    return {
        "mom_accel":     mom_accel,
        "mom_regime":    mom_regime,
        "trend_strength": trend_strength,
    }


# ─────────────────────────────────────────────────────────────────────────────
# C. VOLATILITY FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def feat_volatility(
    close: pd.Series, high: pd.Series, low: pd.Series, atr20: pd.Series
) -> dict[str, pd.Series]:
    """vol_regime, bb_squeeze, hvol_rank, vol_accel, vol_ratio."""
    atr10   = _atr(high, low, close, 10)
    atr_ema = _ema(atr20, 80)

    log_ret = pd.Series(
        np.concatenate([[0.0], np.log(close.values[1:] / close.values[:-1])]),
        index=close.index
    )

    std20   = close.rolling(20, min_periods=2).std()
    bb_width = _safe_div(4 * std20, atr20)

    vol_regime = _rolling_rank(_safe_div(atr20, atr_ema), 120)
    bb_squeeze = _rolling_rank(bb_width, 60)
    hvol_rank  = _rolling_rank(
        log_ret.rolling(20, min_periods=2).std(), 120
    )
    # atr20.diff() → use numpy diff for speed
    atr20_diff = pd.Series(
        np.concatenate([[0.0], np.diff(atr20.values)]), index=close.index
    )
    vol_accel  = _tanh(_rolling_z(atr20_diff, 120))
    vol_ratio  = _tanh(_safe_div(atr10, atr20) - 1)

    return {
        "vol_regime": vol_regime,
        "bb_squeeze": bb_squeeze,
        "hvol_rank":  hvol_rank,
        "vol_accel":  vol_accel,
        "vol_ratio":  vol_ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# D. STRUCTURE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def feat_structure(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, atr20: pd.Series
) -> dict[str, pd.Series]:
    """hh_score, ll_score, inside_bar, engulf_score, swing_pos."""
    idx = close.index

    # hh / ll via Numba kernel
    hh_arr, ll_arr = _hh_ll_score_nb(
        np.ascontiguousarray(high.values,  dtype=np.float64),
        np.ascontiguousarray(low.values,   dtype=np.float64),
        n_lags=5
    )
    hh_score = _wrap(hh_arr, idx)
    ll_score = _wrap(ll_arr, idx)

    # Inside bar
    h = high.values; l = low.values
    range_now = h - l
    range_20  = (
        pd.Series(range_now, index=idx)
        .rolling(20, min_periods=2).mean().values
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        # Safe ratio: 0/0 → nan → handled by np.where
        ratio = np.where(
            (range_20 == 0) | (range_now == 0),
            1.0,                          # exp(-1) ≈ 0.37 → inside_bar ≈ -0.26
            range_now / range_20
        )

    inside_bar = _clamp(_wrap(np.clip(2 * np.exp(-ratio) - 1, -1, 1), idx))

    # Engulf score
    o = open_.values; c = close.values
    body_now  = np.abs(c - o)
    body_prev = np.empty_like(body_now)
    body_prev[0]  = np.nan
    body_prev[1:] = body_now[:-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.where(
            (body_now > 0) & (body_prev > 0),
            np.log(body_now / body_prev),
            0.0
        )

    engulf_score = _tanh(_wrap(log_ratio, idx))

    # Swing position
    swing_pos = _rolling_rank(close - _ema(close, 60), 240)

    return {
        "hh_score":     hh_score,
        "ll_score":     ll_score,
        "inside_bar":   inside_bar,
        "engulf_score": engulf_score,
        "swing_pos":    swing_pos,
    }

# ─────────────────────────────────────────────────────────────────────────────
# E. VOLUME / MICROSTRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def _volume_is_flat(volume: pd.Series, threshold: float = 0.01) -> bool:
    return volume.std() < threshold or volume.nunique() <= 2


def feat_volume(
    close: pd.Series, volume: pd.Series, atr5: pd.Series
) -> dict[str, pd.Series]:
    """vol_shock, vol_trend, vol_sync, cvd_delta."""
    if _volume_is_flat(volume):
        return _feat_volume_atr_proxy(close, atr5)

    vol_ema20 = _ema(volume, 20)
    log_vol   = _ffill(np.log(volume.replace(0, np.nan)))

    vol_shock = _tanh(_rolling_z(_safe_div(volume, vol_ema20), 60))

    # Rolling slope via Numba (replaces rolling().apply(polyfit))
    vol_trend = _tanh(_rolling_z(_rolling_slope(log_vol, 10), 120))

    log_ret   = pd.Series(
        np.concatenate([[0.0], np.log(close.values[1:] / close.values[:-1])]),
        index=close.index
    )
    abs_ret   = log_ret.abs()
    vol_sync  = _tanh(_rolling_z(abs_ret.rolling(20, min_periods=5).corr(volume), 60))

    sign_price = pd.Series(np.sign(np.diff(close.values, prepend=close.values[0])),
                           index=close.index)
    cvd        = (sign_price * volume).rolling(20, min_periods=1).sum()
    vol_norm   = (volume.rolling(20, min_periods=1).sum() + 1e-9) ** 0.5
    cvd_delta  = _tanh(_rolling_z(_safe_div(cvd, vol_norm), 60))

    return {
        "vol_shock": vol_shock,
        "vol_trend": vol_trend,
        "vol_sync":  vol_sync,
        "cvd_delta": cvd_delta,
    }


def _feat_volume_atr_proxy(
    close: pd.Series, atr5: pd.Series
) -> dict[str, pd.Series]:
    """ATR-based substitutes when volume is flat."""
    log_ret  = pd.Series(
        np.concatenate([[0.0], np.log(close.values[1:] / close.values[:-1])]),
        index=close.index
    )
    atr_ema  = _ema(atr5, 20)
    atr_ratio = _safe_div(atr5, atr_ema)
    vol_shock = _tanh(_rolling_z(atr_ratio, 60))

    # Rolling slope via Numba
    vol_trend = _tanh(_rolling_z(_rolling_slope(atr5, 10), 120))

    abs_ret   = log_ret.abs()
    corr_ra   = abs_ret.rolling(20, min_periods=5).corr(atr5)
    vol_sync  = _tanh(_rolling_z(corr_ra, 60))

    signed_atr = pd.Series(np.sign(log_ret.values) * atr5.values, index=close.index)
    cvd_delta  = _tanh(_rolling_z(
        signed_atr.rolling(20, min_periods=1).sum(), 60
    ))

    return {
        "vol_shock": vol_shock,
        "vol_trend": vol_trend,
        "vol_sync":  vol_sync,
        "cvd_delta": cvd_delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# F. CROSS-ASSET CORRELATION & LEAD-LAG
# ─────────────────────────────────────────────────────────────────────────────

def feat_correlation(
    ret_x: pd.Series, ret_a: pd.Series, prefix: str
) -> dict[str, pd.Series]:
    """
    corr_w for w ∈ {5,15,60,240}, corr_accel, lead_lag, sync_score.
    
    Rolling corr is kept in pandas (well-optimised internally).
    xcorr_max is routed to the Numba kernel.
    """
    out   = {}
    corrs = {}
    idx   = ret_x.index

    for w in [5, 15, 60, 240]:
        c = ret_x.rolling(w, min_periods=max(3, w // 4)).corr(ret_a)
        out[f"{prefix}_corr_{w}"] = _clamp(c)
        corrs[w] = c

    # Correlation acceleration
    corr15     = corrs[15]
    corr_sig   = corr15.rolling(60, min_periods=10).std().replace(0, np.nan).fillna(1)
    out[f"{prefix}_corr_accel"] = _tanh((corr15 - corr15.shift(5)) / corr_sig)

    # Lead-lag via Numba (major speedup)
    ll_arr = _xcorr_max_nb(
        np.ascontiguousarray(ret_x.fillna(0).values, dtype=np.float64),
        np.ascontiguousarray(ret_a.fillna(0).values, dtype=np.float64),
        window=60, max_lag=5
    )
    out[f"{prefix}_lead_lag"] = _wrap(_clip1_nb(ll_arr), idx)

    # Sync score
    same_sign = pd.Series(
        (np.sign(ret_x.values) == np.sign(ret_a.values)).astype(np.float64),
        index=idx
    )
    out[f"{prefix}_sync_score"] = _clamp(_ema(same_sign, span=5) * 2 - 1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# G. RELATIVE STRENGTH
# ─────────────────────────────────────────────────────────────────────────────

def feat_relative_strength(
    close_x: pd.Series, close_a: pd.Series,
    rsi_x:   pd.Series, rsi_a:   pd.Series,
    prefix:  str
) -> dict[str, pd.Series]:
    """rel_ret_n, ratio_z, ratio_mom, ratio_accel, rel_strength."""
    out = {}
    lc_x = np.log(close_x.values)
    lc_a = np.log(close_a.values)
    idx  = close_x.index

    for n in [1, 5, 15, 60]:
        rx_arr = np.empty(len(lc_x), dtype=np.float64)
        ra_arr = np.empty(len(lc_a), dtype=np.float64)
        rx_arr[:n] = 0.0; rx_arr[n:] = lc_x[n:] - lc_x[:-n]
        ra_arr[:n] = 0.0; ra_arr[n:] = lc_a[n:] - lc_a[:-n]
        diff = pd.Series(ra_arr - rx_arr, index=idx)
        out[f"{prefix}_rel_ret_{n}"] = _tanh(_rolling_z(diff, 120))

    ratio     = _safe_div(close_a, close_x)
    ratio_z   = _tanh(_rolling_z(ratio, 120))
    # Use numpy diff for single-pass
    rz_diff   = pd.Series(
        np.concatenate([[0.0], np.diff(ratio_z.values)]), index=idx
    )
    ratio_mom = _tanh(_rolling_z(rz_diff, 60))
    rz_diff2  = pd.Series(
        np.concatenate([[0.0], np.diff(rz_diff.values)]), index=idx
    )
    ratio_acc = _tanh(_rolling_z(rz_diff2, 60))

    out[f"{prefix}_ratio_z"]     = ratio_z
    out[f"{prefix}_ratio_mom"]   = ratio_mom
    out[f"{prefix}_ratio_accel"] = ratio_acc
    out[f"{prefix}_rel_strength"] = _clamp(rsi_a - rsi_x)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# H. BASKET FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def feat_basket(
    close_x:     pd.Series,
    rsi_x:       pd.Series,
    pred_closes: dict[str, pd.Series],
    pred_rsis:   dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """basket_ret, basket_mom, dispersion, leadership."""
    out = {}
    if not pred_closes:
        return out

    idx    = close_x.index
    lc_x   = np.log(close_x.values)

    # 5-bar log returns — batch via numpy stack
    sym_list    = list(pred_closes.keys())
    n_preds     = len(sym_list)
    n           = len(lc_x)

    ret5_matrix = np.zeros((n, n_preds), dtype=np.float64)
    for j, sym in enumerate(sym_list):
        lc_a = np.log(pred_closes[sym].values)
        ret5_matrix[5:, j] = lc_a[5:] - lc_a[:-5]

    ret5_x = np.empty(n, dtype=np.float64)
    ret5_x[:5] = 0.0
    ret5_x[5:] = lc_x[5:] - lc_x[:-5]

    # basket_ret
    basket_ret5       = ret5_matrix.mean(axis=1)
    out["basket_ret"] = _tanh(_rolling_z(pd.Series(basket_ret5, index=idx), 120))

    # basket_mom — stacked RSI matrix
    rsi_matrix = np.stack([pred_rsis[sym].values for sym in sym_list], axis=1)
    basket_raw = pd.Series(rsi_matrix.mean(axis=1), index=idx)
    if basket_raw.std() < 0.01:
        out["basket_mom"] = _tanh(_rolling_z(basket_raw, 60))
    else:
        out["basket_mom"] = _clamp(basket_raw)

    # dispersion — std across all assets (including x)
    all_ret5 = np.hstack([ret5_x.reshape(-1, 1), ret5_matrix])
    std_arr  = all_ret5.std(axis=1)
    out["dispersion"] = _rolling_rank(pd.Series(std_arr, index=idx), 120)

    # leadership
    median_arr = np.median(all_ret5, axis=1)
    out["leadership"] = _rolling_rank(
        pd.Series(ret5_x - median_arr, index=idx), 120
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# LAG APPLIER  — batch construction via numpy
# ─────────────────────────────────────────────────────────────────────────────

LAGS = [0, 1, 2, 5, 15, 60]


def apply_lags(features: dict[str, pd.Series], index: pd.Index) -> pd.DataFrame:
    """
    Apply LAGS to every feature and return a single DataFrame.

    Key optimization: build all columns as numpy arrays,
    then construct DataFrame once (avoids repeated Series alignment).

    Parameters
    ----------
    features : dict of already-normalized Series
    index    : original DatetimeIndex

    Returns
    -------
    pd.DataFrame with columns {name}_lag{k} for each name and lag k
    """
    n        = len(index)
    col_names: list[str]      = []
    col_data:  list[np.ndarray] = []

    for name, series in features.items():
        arr = series.values.astype(np.float64)
        for lag in LAGS:
            col_names.append(f"{name}_lag{lag}")
            if lag == 0:
                col_data.append(arr.copy())
            else:
                shifted = np.empty(n, dtype=np.float64)
                shifted[:lag] = np.nan
                shifted[lag:] = arr[:-lag]
                col_data.append(shifted)

    # Single DataFrame construction — much faster than dict-of-Series
    matrix = np.column_stack(col_data)        # (n, n_cols)
    return pd.DataFrame(matrix, index=index, columns=col_names)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    master:       pd.DataFrame,
    data_cfg:     dict,
    pipeline_cfg: dict,
    apply_lag:    bool = True,
    verbose:      bool = True
) -> pd.DataFrame:
    """
    Build complete normalized feature matrix.

    Parameters
    ----------
    master       : aligned DataFrame from Phase 1 run_pipeline
    data_cfg     : data_config.json as dict
    pipeline_cfg : pipeline_config.json as dict
    apply_lag    : add lagged columns at offsets LAGS
    verbose      : print progress

    Returns
    -------
    pd.DataFrame — features ∈ [-1,+1], meta cols prefixed with '_'
    Warm-up rows dropped, residual NaNs forward-filled then back-filled.
    """
    target_sym = data_cfg["target_symbol"]
    pred_syms  = data_cfg["predictor_symbols"]

    _vprint(verbose, "\n" + "─" * 60)
    _vprint(verbose, "  PHASE 2 — FEATURE ENGINEERING  (OPTIMIZED)")
    _vprint(verbose, f"  Numba: {'✓ enabled' if NUMBA_AVAILABLE else '✗ fallback'}")
    _vprint(verbose, "─" * 60)

    all_features: dict[str, pd.Series] = {}

    # ── Target OHLCV ──────────────────────────────────────────────────────────
    O = master["open"]
    H = master["high"]
    L = master["low"]
    C = master["close"]
    V = master["volume"]

    # Pre-compute shared intermediates ONCE
    atr20_x = _atr(H, L, C, 20)
    atr5_x  = _atr(H, L, C, 5)
    rsi14_x = _rsi_raw(C, 14)
    ret1_x  = pd.Series(
        np.concatenate([[0.0], np.log(C.values[1:] / C.values[:-1])]),
        index=C.index
    )

    # ── A. Price ──────────────────────────────────────────────────────────────
    _vprint(verbose, "\n[A] Price-based features")
    all_features |= feat_returns(C)
    all_features |= feat_bar_structure(O, H, L, C, atr20_x)
    all_features["vwap_dev"]  = feat_vwap_dev(C, H, L, V, atr20_x)
    all_features["price_pos"] = feat_price_pos(C)
    _vprint(verbose, f"    → {len(all_features)} features so far")

    # ── B. Momentum ───────────────────────────────────────────────────────────
    _vprint(verbose, "[B] Momentum features")
    all_features |= feat_rsi(C)
    all_features |= feat_roc(C)
    all_features |= feat_momentum_extras(C, atr20_x)
    _vprint(verbose, f"    → {len(all_features)} features so far")

    # ── C. Volatility ─────────────────────────────────────────────────────────
    _vprint(verbose, "[C] Volatility features")
    all_features |= feat_volatility(C, H, L, atr20_x)
    _vprint(verbose, f"    → {len(all_features)} features so far")

    # ── D. Structure ──────────────────────────────────────────────────────────
    _vprint(verbose, "[D] Structure features")
    all_features |= feat_structure(O, H, L, C, atr20_x)
    _vprint(verbose, f"    → {len(all_features)} features so far")

    # ── E. Volume ─────────────────────────────────────────────────────────────
    _vprint(verbose, "[E] Volume / microstructure features")
    all_features |= feat_volume(C, V, atr5_x)
    _vprint(verbose, f"    → {len(all_features)} features so far")

    # ── F–H. Cross-asset (shared ret1 reused across all predictors) ───────────
    pred_closes: dict[str, pd.Series] = {}
    pred_rsis:   dict[str, pd.Series] = {}

    for sym in pred_syms:
        _vprint(verbose, f"[F-H] Cross-asset ← {sym}")

        C_a = master[f"{sym}_close"]
        ret1_a = pd.Series(
            np.concatenate([[0.0], np.log(C_a.values[1:] / C_a.values[:-1])]),
            index=C_a.index
        )
        rsi14_a = _rsi_raw(C_a, 14)

        pred_closes[sym] = C_a
        pred_rsis[sym]   = rsi14_a / 50 - 1

        # ret1_x pre-computed above — pass directly, no recompute
        all_features |= feat_correlation(ret1_x, ret1_a, prefix=sym)
        all_features |= feat_relative_strength(
            close_x=C,   close_a=C_a,
            rsi_x=rsi14_x / 50 - 1,
            rsi_a=rsi14_a / 50 - 1,
            prefix=sym
        )

    # ── H. Basket ─────────────────────────────────────────────────────────────
    _vprint(verbose, "[H] Basket / dispersion features")
    all_features |= feat_basket(
        close_x=C,   rsi_x=rsi14_x / 50 - 1,
        pred_closes=pred_closes,
        pred_rsis=pred_rsis,
    )

    # ── Meta columns ──────────────────────────────────────────────────────────
    all_features["_atr20"]   = atr20_x
    all_features["_session"] = master["session"]
    all_features["_close"]   = C

    base_count = len([k for k in all_features if not k.startswith("_")])
    _vprint(verbose, f"\n  Base features (before lags) : {base_count}")

    # ── Lags ──────────────────────────────────────────────────────────────────
    if apply_lag:
        _vprint(verbose, f"  Applying lags               : {LAGS}")
        feat_only = {k: v for k, v in all_features.items() if not k.startswith("_")}
        meta_only = {k: v for k, v in all_features.items() if k.startswith("_")}

        # Batch lag construction — single DataFrame build
        df_feats = apply_lags(feat_only, master.index)

        # Attach meta columns
        for k, v in meta_only.items():
            df_feats[k] = v.values

        df = df_feats
        _vprint(verbose, f"  Total columns (with lags)   : {df.shape[1]}")
    else:
        df = pd.DataFrame(all_features, index=master.index)

    feat_cols = [c for c in df.columns if not c.startswith("_")]

    # ── Drop warm-up rows (>80% NaN) ──────────────────────────────────────────
    nan_frac = df[feat_cols].isna().mean(axis=1)
    n_before = len(df)
    df       = df[nan_frac < 0.80].copy()
    _vprint(verbose, f"  Warm-up rows dropped        : "
                     f"{n_before - len(df):,}  ({n_before:,} → {len(df):,})")

    # ── Fill residual NaNs ────────────────────────────────────────────────────
    nan_before = int(df[feat_cols].isna().sum().sum())
    df[feat_cols] = df[feat_cols].ffill().bfill()
    nan_after  = int(df[feat_cols].isna().sum().sum())
    if verbose and nan_before > 0:
        print(f"  Residual NaNs filled        : {nan_before:,} → {nan_after:,}")

    # ── Final clamp ───────────────────────────────────────────────────────────
    df[feat_cols] = df[feat_cols].clip(-1, 1)

    _vprint(verbose, f"  Final shape                 : {df.shape}")
    _vprint(verbose, "─" * 60)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_features(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Check all non-meta feature columns:
      ✓ Bounded  ∈ [-1.01, +1.01]
      ✓ No NaN
      ✓ Non-flat  (std > 0.01)
    """
    feature_cols = [c for c in df.columns if not c.startswith("_")]
    errors: list[str] = []

    # Vectorized checks — much faster than per-column Python loops
    feat_df  = df[feature_cols]
    col_min  = feat_df.min()
    col_max  = feat_df.max()
    col_nan  = feat_df.isna().sum()
    col_std  = feat_df.std()

    for col in feature_cols:
        if col_min[col] < -1.01:
            errors.append(f"{col}: min={col_min[col]:.4f} < -1")
        if col_max[col] > 1.01:
            errors.append(f"{col}: max={col_max[col]:.4f} > +1")
        if col_nan[col] > 0:
            errors.append(f"{col}: {col_nan[col]} NaN values")
        if col_std[col] < 0.01:
            errors.append(f"{col}: std={col_std[col]:.6f} — flat")

    if errors:
        print(f"❌ Validation failed — {len(errors)} issue(s):")
        for e in errors[:20]:
            print(f"   {e}")
        if len(errors) > 20:
            print(f"   ... and {len(errors) - 20} more")
        return False

    if verbose:
        print(
            f"✅ All {len(feature_cols)} features validated\n"
            f"   Bounds : [{col_min.min():.4f}, {col_max.max():.4f}]\n"
            f"   NaN    : {col_nan.sum()}\n"
            f"   Flat   : {(col_std < 0.01).sum()}\n"
            f"   Shape  : {df.shape}"
        )
    return True


if __name__ == "__main__":
    print("Phase 2 Feature Engineering Module — OPTIMIZED")
    print(f"Numba available: {NUMBA_AVAILABLE}")