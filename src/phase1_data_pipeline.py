"""
Phase 1 – Data Pipeline
Load · Clean · Align · Label · Split
Designed for Jupyter import: from src.phase1_data_pipeline import *
"""

import json
from pathlib import Path
import pandas as pd


# ── Config ──────────────────────────────────────────────────────────────────

def load_configs(
    data_cfg_path: str = "config/data_config.json",
    pipeline_cfg_path: str = "config/pipeline_config.json"
) -> tuple[dict, dict]:
    """Load both config files. Returns (data_cfg, pipeline_cfg)."""
    with open(data_cfg_path)     as f: data_cfg     = json.load(f)
    with open(pipeline_cfg_path) as f: pipeline_cfg  = json.load(f)

    n = len(data_cfg["predictor_symbols"])
    assert data_cfg["min_predictors"] <= n <= data_cfg["max_predictors"], (
        f"predictor count {n} outside [{data_cfg['min_predictors']}, "
        f"{data_cfg['max_predictors']}]"
    )
    return data_cfg, pipeline_cfg


# ── Load ─────────────────────────────────────────────────────────────────────

def load_symbol(symbol: str, data_cfg: dict) -> pd.DataFrame:
    """
    Load one symbol CSV → clean DatetimeIndex (UTC).
    Expected columns: timestamp, open, high, low, close, volume
    """
    path = Path(data_cfg["data_folder"]) / f"{symbol}{data_cfg['file_suffix']}"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")

    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.sort_index(inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    return df


# ── Clean ────────────────────────────────────────────────────────────────────

def clean_symbol(df: pd.DataFrame, symbol: str = "") -> tuple[pd.DataFrame, dict]:
    """
    Per-symbol cleaning:
      - Drop duplicate timestamps (keep last)
      - Drop weekends (Saturday=5, Sunday=6)
      - Drop rows with non-positive OHLC
    Returns cleaned df + report dict.
    """
    report = {"symbol": symbol, "raw_bars": len(df)}

    # Duplicates
    dupes = df.index.duplicated(keep="last")
    report["duplicates_removed"] = int(dupes.sum())
    df = df[~dupes]

    # Weekends
    is_weekend = df.index.dayofweek >= 5
    report["weekend_bars_removed"] = int(is_weekend.sum())
    df = df[~is_weekend]

    # Non-positive prices
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    report["bad_price_bars_removed"] = int(bad.sum())
    df = df[~bad]

    report["clean_bars"] = len(df)
    return df, report


# ── Align ────────────────────────────────────────────────────────────────────

def align_predictors(
    target: pd.DataFrame,
    predictors: dict[str, pd.DataFrame],
    max_fill_gap: int = 5
) -> pd.DataFrame:
    """
    Align all predictors to target's M1 index.
    - Forward-fill gaps up to max_fill_gap bars (flag with is_filled=1)
    - Gaps > max_fill_gap remain NaN (session breaks)
    - Drop rows where target close is NaN
    Returns single wide DataFrame.
    """
    master_idx = target.index

    frames = {"": target.copy()}           # target cols: open, high, low, close, volume
    frames[""]["is_filled"] = 0

    for sym, df in predictors.items():
        prefix = f"{sym}_"
        df_r   = df.reindex(master_idx)    # reindex to master clock

        was_nan = df_r["close"].isna()

        # Forward-fill with hard limit
        df_r = df_r.ffill(limit=max_fill_gap)

        # Flag filled bars
        df_r["is_filled"] = (was_nan & df_r["close"].notna()).astype(int)

        df_r.columns = [f"{prefix}{c}" for c in df_r.columns]
        frames[sym] = df_r

    master = pd.concat(frames.values(), axis=1)

    # Drop bars where target has no data
    master = master.dropna(subset=["close"])
    return master


# ── Session Labels ───────────────────────────────────────────────────────────

def add_session_labels(df: pd.DataFrame, sessions: dict) -> pd.DataFrame:
    """
    Add integer session column based on UTC hour.
    sessions: dict from pipeline_config["sessions"]
    """
    hour = df.index.hour
    df   = df.copy()
    df["session"] = 0                      # default: Dead_zone

    for label, info in sessions.items():
        mask = (hour >= info["start_hour"]) & (hour < info["end_hour"])
        df.loc[mask, "session"] = int(label)

    return df


# ── Split ────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    splits_cfg: dict
) -> dict[str, pd.DataFrame]:
    """
    Slice master DataFrame into train / validation / test.
    Dates come from pipeline_config["splits"].
    Returns dict with keys: train, validation, test.
    """
    result = {}
    for name, bounds in splits_cfg.items():
        mask = (df.index >= bounds["start"]) & (df.index <= bounds["end"])
        result[name] = df[mask].copy()
    return result


# ── Master Runner ────────────────────────────────────────────────────────────

def run_pipeline(
    data_cfg_path:     str = "config/data_config.json",
    pipeline_cfg_path: str = "config/pipeline_config.json",
    verbose:           bool = True
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """
    Full Phase 1 pipeline.

    Returns
    -------
    splits  : dict  – {"train": df, "validation": df, "test": df}
    master  : df    – full aligned + labelled DataFrame
    reports : dict  – cleaning reports per symbol
    """
    data_cfg, pipeline_cfg = load_configs(data_cfg_path, pipeline_cfg_path)

    # ── Load & clean target ──────────────────────────────────────────────────
    target_sym  = data_cfg["target_symbol"]
    target_raw  = load_symbol(target_sym, data_cfg)
    target, rpt = clean_symbol(target_raw, target_sym)
    reports     = {target_sym: rpt}

    if verbose:
        print(f"[TARGET]  {target_sym}: {rpt['clean_bars']:,} bars "
              f"| dupes removed: {rpt['duplicates_removed']} "
              f"| weekends: {rpt['weekend_bars_removed']}")

    # ── Load & clean predictors ──────────────────────────────────────────────
    predictors = {}
    for sym in data_cfg["predictor_symbols"]:
        raw, rpt_p   = clean_symbol(load_symbol(sym, data_cfg), sym)
        predictors[sym] = raw
        reports[sym]    = rpt_p
        if verbose:
            print(f"[PRED]    {sym}: {rpt_p['clean_bars']:,} bars "
                  f"| dupes: {rpt_p['duplicates_removed']} "
                  f"| weekends: {rpt_p['weekend_bars_removed']}")

    # ── Align ────────────────────────────────────────────────────────────────
    master = align_predictors(target, predictors, pipeline_cfg["max_fill_gap"])
    if verbose:
        print(f"\n[ALIGN]   Master shape: {master.shape} "
              f"| {master.index[0]} → {master.index[-1]}")

    # ── Session labels ───────────────────────────────────────────────────────
    master = add_session_labels(master, pipeline_cfg["sessions"])
    if verbose:
        print(f"[SESSION] Distribution:\n"
              f"{master['session'].value_counts().sort_index().to_string()}\n")

    # ── Split ────────────────────────────────────────────────────────────────
    splits = split_data(master, pipeline_cfg["splits"])
    if verbose:
        for name, df in splits.items():
            sacred = " ⚠️  SACRED — do not touch until final go/no-go" if name == "test" else ""
            print(f"[SPLIT]   {name:<12}: {len(df):>8,} bars "
                  f"| {df.index[0]} → {df.index[-1]}{sacred}")

    return splits, master, reports