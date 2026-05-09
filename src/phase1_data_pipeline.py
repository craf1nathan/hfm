"""
Phase 1 – Data Pipeline
Load · Clean · Align · Session Label · Split

Usage in Jupyter:
    import sys; sys.path.append("..")
    from src.phase1_data_pipeline import run_pipeline
    splits, master, reports = run_pipeline()
"""

import json
from pathlib import Path
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_configs(
    data_cfg_path:     str = "config/data_config.json",
    pipeline_cfg_path: str = "config/pipeline_config.json"
) -> tuple[dict, dict]:
    """
    Load and validate both config files.
    Injects resolved absolute 'data_folder' so all downstream
    functions are path-agnostic regardless of working directory.
    Returns (data_cfg, pipeline_cfg).
    """
    data_cfg_path     = Path(data_cfg_path).resolve()
    pipeline_cfg_path = Path(pipeline_cfg_path).resolve()

    with open(data_cfg_path)     as f: data_cfg     = json.load(f)
    with open(pipeline_cfg_path) as f: pipeline_cfg = json.load(f)

    # ── Resolve data_folder relative to data_config.json location ────────────
    raw_folder = data_cfg["data_folder"]          # e.g.  "./data/"
    if not Path(raw_folder).is_absolute():
        # anchor to the directory that contains data_config.json
        resolved = (data_cfg_path.parent / raw_folder).resolve()
    else:
        resolved = Path(raw_folder).resolve()

    data_cfg["data_folder"] = str(resolved)       # overwrite with absolute path

    # ── Predictor count guard ─────────────────────────────────────────────────
    n  = len(data_cfg["predictor_symbols"])
    mn = data_cfg["min_predictors"]
    mx = data_cfg["max_predictors"]
    assert mn <= n <= mx, (
        f"predictor count={n} must be between {mn} and {mx}. "
        f"Edit predictor_symbols in data_config.json."
    )

    return data_cfg, pipeline_cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_symbol(symbol: str, data_cfg: dict) -> pd.DataFrame:
    """
    Read one <SYMBOL>_M1.csv → DataFrame with UTC DatetimeIndex.
    Expected CSV columns: timestamp, open, high, low, close, volume
    data_cfg["data_folder"] is always an absolute path (set by load_configs).
    """
    path = Path(data_cfg["data_folder"]) / f"{symbol}{data_cfg['file_suffix']}"

    if not path.exists():
        raise FileNotFoundError(
            f"\n  File not found : {path}"
            f"\n  data_folder    : {data_cfg['data_folder']}"
            f"\n  symbol         : {symbol}"
            f"\n  Check data_config.json → data_folder and that the CSV exists."
        )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
        index_col="timestamp"
    )

    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"
    df.sort_index(inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def clean_symbol(df: pd.DataFrame, symbol: str = "") -> tuple[pd.DataFrame, dict]:
    """
    Clean one symbol DataFrame:
      • Duplicate timestamps → keep last
      • Weekend bars         → drop  (dayofweek 5=Sat, 6=Sun)
      • Non-positive OHLC    → drop
    Returns (cleaned_df, report_dict).
    """
    report = {
        "symbol":                 symbol,
        "raw_bars":               len(df),
        "duplicates_removed":     0,
        "weekend_bars_removed":   0,
        "bad_price_bars_removed": 0,
        "clean_bars":             0,
    }

    # Duplicates
    mask_dupe = df.index.duplicated(keep="last")
    report["duplicates_removed"] = int(mask_dupe.sum())
    df = df[~mask_dupe]

    # Weekends
    mask_wknd = df.index.dayofweek >= 5
    report["weekend_bars_removed"] = int(mask_wknd.sum())
    df = df[~mask_wknd]

    # Non-positive prices
    mask_bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    report["bad_price_bars_removed"] = int(mask_bad.sum())
    df = df[~mask_bad]

    report["clean_bars"] = len(df)
    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# 4. ALIGN
# ─────────────────────────────────────────────────────────────────────────────

def align_predictors(
    target:       pd.DataFrame,
    predictors:   dict[str, pd.DataFrame],
    max_fill_gap: int = 5
) -> pd.DataFrame:
    """
    Reindex all predictors to target's M1 timestamp index.
      • Forward-fill up to max_fill_gap consecutive missing bars.
      • Filled bars flagged: <SYMBOL>_is_filled = 1
      • Gaps > max_fill_gap stay NaN  (session breaks).
      • Rows where target close is NaN are dropped.
    Returns one wide DataFrame (target cols unprefixed,
    predictor cols prefixed with <SYMBOL>_).
    """
    master_idx = target.index

    # ── Target block ──────────────────────────────────────────────────────────
    tgt              = target.copy()
    tgt["is_filled"] = 0
    frames           = [tgt]

    # ── Predictor blocks ──────────────────────────────────────────────────────
    for sym, df in predictors.items():
        df_r    = df.reindex(master_idx)
        was_nan = df_r["close"].isna()

        df_r    = df_r.ffill(limit=max_fill_gap)

        df_r["is_filled"] = (was_nan & df_r["close"].notna()).astype(int)
        df_r.columns      = [f"{sym}_{c}" for c in df_r.columns]
        frames.append(df_r)

    master = pd.concat(frames, axis=1)
    master.dropna(subset=["close"], inplace=True)
    return master


# ─────────────────────────────────────────────────────────────────────────────
# 5. SESSION LABELS
# ─────────────────────────────────────────────────────────────────────────────

def add_session_labels(df: pd.DataFrame, sessions: dict) -> pd.DataFrame:
    """
    Add integer 'session' column based on UTC hour of each bar.
    sessions: pipeline_config["sessions"]
    """
    df            = df.copy()
    hour          = df.index.hour
    df["session"] = 0                      # default → Dead_zone

    for label, info in sessions.items():
        mask = (hour >= info["start_hour"]) & (hour < info["end_hour"])
        df.loc[mask, "session"] = int(label)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    df:         pd.DataFrame,
    splits_cfg: dict
) -> dict[str, pd.DataFrame]:
    """
    Slice master DataFrame into named windows.
    splits_cfg: pipeline_config["splits"]
      keys  : train | validation | test
      values: {"start": "...", "end": "..."}
    Returns dict[split_name → DataFrame].
    """
    result = {}
    for name, bounds in splits_cfg.items():
        mask         = (df.index >= bounds["start"]) & (df.index <= bounds["end"])
        result[name] = df.loc[mask].copy()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_cfg_path:     str  = "config/data_config.json",
    pipeline_cfg_path: str  = "config/pipeline_config.json",
    verbose:           bool = True
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """
    Execute the full Phase 1 pipeline.

    Parameters
    ----------
    data_cfg_path     : path to data_config.json  (absolute or relative)
    pipeline_cfg_path : path to pipeline_config.json
    verbose           : print progress to stdout

    Returns
    -------
    splits  : dict {"train": df, "validation": df, "test": df}
    master  : pd.DataFrame — full aligned + labelled dataset
    reports : dict — per-symbol cleaning report
    """

    # ── Config ────────────────────────────────────────────────────────────────
    data_cfg, pipeline_cfg = load_configs(data_cfg_path, pipeline_cfg_path)
    target_sym             = data_cfg["target_symbol"]
    predictor_syms         = data_cfg["predictor_symbols"]

    _hdr("PHASE 1 — DATA PIPELINE", verbose)
    if verbose:
        print(f"  data_folder : {data_cfg['data_folder']}")

    # ── Load + clean target ───────────────────────────────────────────────────
    _log("Loading & cleaning target", verbose)
    target_raw, rpt_t = clean_symbol(load_symbol(target_sym, data_cfg), target_sym)
    reports           = {target_sym: rpt_t}
    _print_report(rpt_t, tag="TARGET", verbose=verbose)

    # ── Load + clean predictors ───────────────────────────────────────────────
    _log("Loading & cleaning predictors", verbose)
    predictors: dict[str, pd.DataFrame] = {}
    for sym in predictor_syms:
        df_clean, rpt   = clean_symbol(load_symbol(sym, data_cfg), sym)
        predictors[sym] = df_clean
        reports[sym]    = rpt
        _print_report(rpt, tag="PRED  ", verbose=verbose)

    # ── Align ─────────────────────────────────────────────────────────────────
    _log("Aligning to master clock (target index)", verbose)
    master = align_predictors(target_raw, predictors, pipeline_cfg["max_fill_gap"])
    if verbose:
        print(f"         shape : {master.shape}")
        print(f"         range : {master.index[0]}  →  {master.index[-1]}")

    # ── Session labels ────────────────────────────────────────────────────────
    _log("Assigning session labels", verbose)
    master = add_session_labels(master, pipeline_cfg["sessions"])
    if verbose:
        dist  = master["session"].value_counts().sort_index()
        names = {int(k): v["name"] for k, v in pipeline_cfg["sessions"].items()}
        for tag, count in dist.items():
            print(f"         [{tag}] {names[tag]:<12} : {count:>10,} bars")

    # ── Split ─────────────────────────────────────────────────────────────────
    _log("Splitting data", verbose)
    splits = split_data(master, pipeline_cfg["splits"])
    if verbose:
        for name, df in splits.items():
            sacred = "  ⚠️  SACRED" if name == "test" else ""
            print(
                f"         {name:<12} : {len(df):>8,} bars"
                f"  {df.index[0]}  →  {df.index[-1]}{sacred}"
            )

    _hdr("PHASE 1 COMPLETE", verbose)
    return splits, master, reports


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (internal)
# ─────────────────────────────────────────────────────────────────────────────

def _hdr(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  {msg}")
        print(f"{'─' * 60}")

def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"\n[·] {msg}")

def _print_report(rpt: dict, tag: str, verbose: bool) -> None:
    if verbose:
        print(
            f"    [{tag}] {rpt['symbol']:<10} "
            f"raw={rpt['raw_bars']:>9,}  "
            f"clean={rpt['clean_bars']:>9,}  "
            f"dupes={rpt['duplicates_removed']:>4}  "
            f"wknd={rpt['weekend_bars_removed']:>6,}  "
            f"bad_px={rpt['bad_price_bars_removed']:>3}"
        )