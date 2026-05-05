# main.py
"""
ORCHESTRATEUR PRINCIPAL — Lance tout le pipeline.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, 'src')

from src.utils import log_section, log_result, ProgressBar
from src.swing_detection import run_swing_detection, SwingDetector
from src.sequence_building import run_sequence_building
from src.feature_engineering import run_feature_engineering
from src.label_engineering import run_label_engineering
from src.regime_detection import run_regime_detection
from src.quantile_regression import run_quantile_regression
from src.backtesting import run_backtest

def main():
    log_section("SWING HEDGE FUND MANAGER — FULL RESEARCH PIPELINE")
    
    # Setup
    ohlcv_path = 'data/XAUUSD_1H_sample.csv'
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────
    # STEP 1: Load OHLCV
    # ─────────────────────────────────────────────────────────────
    print("[1/9] Loading OHLCV data...")
    df = pd.read_csv(ohlcv_path)
    df['time'] = pd.to_datetime(df['time'])
    log_result(f"Bars loaded", len(df))
    log_result(f"Date range", f"{df['time'].min()} to {df['time'].max()}")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 2: Calculate ATR
    # ─────────────────────────────────────────────────────────────
    print("\n[2/9] Calculating ATR (14-period)...")
    detector = SwingDetector()
    atr = detector.calculate_atr(df)
    log_result("ATR mean", f"{atr.mean():.4f}")
    log_result("ATR range", f"{atr.min():.4f} - {atr.max():.4f}")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 3: Swing Detection
    # ─────────────────────────────────────────────────────────────
    print("\n[3/9] Detecting swings (lookback=5)...")
    swings_df = run_swing_detection(
        ohlcv_path,
        str(output_dir / 'swings.csv')
    )
    
    # ─────────────────────────────────────────────────────────────
    # STEP 4: Sequence Building
    # ─────────────────────────────────────────────────────────────
    print("\n[4/9] Building leg sequences...")
    seqs_df, sequences = run_sequence_building(
        str(output_dir / 'swings.csv'),
        atr,
        str(output_dir / 'sequences.csv')
    )
    
    # ─────────────────────────────────────────────────────────────
    # STEP 5: Feature Engineering
    # ─────────────────────────────────────────────────────────────
    print("\n[5/9] Feature engineering (causal only)...")
    features_df = run_feature_engineering(
        ohlcv_path,
        str(output_dir / 'sequences.csv'),
        atr,
        str(output_dir / 'features.csv')
    )
    
    # ─────────────────────────────────────────────────────────────
    # STEP 6: Label Engineering
    # ─────────────────────────────────────────────────────────────
    print("\n[6/9] Label engineering (simulating on future OHLCV)...")
    dataset = run_label_engineering(
        str(output_dir / 'features.csv'),
        ohlcv_path,
        atr,
        str(output_dir / 'dataset_ml.csv')
    )
    
    # ─────────────────────────────────────────────────────────────
    # STEP 7: Regime Detection
    # ─────────────────────────────────────────────────────────────
    print("\n[7/9] Detecting market regimes (HMM with 4 states)...")
    dataset_regimes = run_regime_detection(
        str(output_dir / 'dataset_ml.csv'),
        str(output_dir / 'dataset_regimes.csv')
    )
    
    # ─────────────────────────────────────────────────────────────
    # STEP 8: Quantile Regression
    # ─────────────────────────────────────────────────────────────
    print("\n[8/9] Training quantile regression per regime...")
    dataset_final = run_quantile_regression(
        str(output_dir / 'dataset_regimes.csv'),
        str(output_dir / 'dataset_final.csv')
    )
    
    # ─────────────────────────────────────────────────────────────
    # STEP 9: Backtesting
    # ─────────────────────────────────────────────────────────────
    print("\n[9/9] Running backtest with Kelly sizing...")
    stats, trades, equity_curve = run_backtest(
        str(output_dir / 'dataset_final.csv'),
        str(output_dir / 'backtest_trades.csv')
    )
    
    # ─────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────
    log_section("RESEARCH COMPLETE — SUMMARY")
    
    print("Output files generated:")
    for f in sorted(output_dir.glob('*.csv')):
        size = f.stat().st_size / 1024
        print(f"  ✓ {f.name:<40s} ({size:>6.1f} KB)")
    
    print("\nKey Results:")
    log_result("Total Return", f"{stats['total_return_pct']:.2f}%")
    log_result("Win Rate", f"{stats['win_rate']:.1%}")
    log_result("Max Drawdown", f"{stats['max_drawdown_pct']:.2f}%")
    log_result("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
    log_result("Total Trades", f"{stats['total_trades']:.0f}")
    
    return dataset_final, stats

if __name__ == '__main__':
    try:
        dataset, stats = main()
        print("\n✅ Pipeline executed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
