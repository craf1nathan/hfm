# src/08_backtesting.py
"""
Backtest complet avec sizing Kelly fractionnel.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Trade:
    bar_index: int
    entry_price: float
    entry_regime: str
    exit_price: float
    position_size: float
    pnl_pct: float
    pnl_atr: float
    exit_reason: str      # 'P75', 'P50', 'SL', 'TIMEOUT'
    bars_held: int
    
    @property
    def r_multiple(self):
        # Simple R calculation
        if self.pnl_atr >= 0:
            return self.pnl_atr / 0.5  # Assuming SL at 0.5 ATR
        else:
            return self.pnl_atr / 0.5

class BacktestEngine:
    def __init__(self, initial_equity: float = 100000.0):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.trades: List[Trade] = []
    
    def calculate_kelly_sizing(self, 
                              regime_stats: dict,
                              regime: str,
                              max_risk_pct: float = 0.02) -> float:
        """
        Kelly fractionnel par régime.
        
        Kelly = (WR * RR - (1 - WR)) / RR
        Fraction = 25% de Kelly (conservative)
        """
        wr = regime_stats.get('win_rate', 0.5)
        rr = regime_stats.get('avg_rr', 2.0)
        
        if rr <= 0 or wr == 0:
            return 0.01  # 1% min
        
        kelly = (wr * rr - (1 - wr)) / rr
        kelly_fraction = 0.25 * max(0, kelly)
        
        size = max(0.01, min(kelly_fraction, max_risk_pct))
        return size
    
    def run(self, df: pd.DataFrame) -> dict:
        """
        Lance le backtest.
        """
        df = df.sort_values('bar_index')
        
        # Calcul des stats par régime
        regime_stats = {}
        for regime in df['regime_viterbi'].unique():
            regime_df = df[df['regime_viterbi'] == regime]
            regime_stats[regime] = {
                'win_rate': (regime_df['label_win'] == 1).mean(),
                'avg_pnl': regime_df['label_pnl_atr'].mean(),
                'avg_rr': 3.0,  # simplified
                'n_samples': len(regime_df),
            }
        
        print("Regime Statistics:")
        for regime, stats in regime_stats.items():
            print(f"  {regime:15s}: WR={stats['win_rate']:.1%}, "
                 f"AvgPnL={stats['avg_pnl']:.2f} ATR, N={stats['n_samples']}")
        
        # Simuler les trades
        equity_curve = [self.initial_equity]
        
        for _, row in df.iterrows():
            regime = row['regime_viterbi']
            
            # Entrée
            entry_price = row['close_price']
            entry_regime = regime
            
            # Sizing
            size = self.calculate_kelly_sizing(regime_stats[regime], regime)
            
            # Cibles (quantiles)
            p50 = row.get('pred_p50', entry_price)
            p75 = row.get('pred_p75', entry_price)
            
            # Simuler la sortie
            actual_pnl_atr = row['label_pnl_atr']
            exit_reason = row['label_outcome']
            
            exit_price = entry_price + (actual_pnl_atr * row['atr_at_event'])
            pnl_pct = (exit_price - entry_price) / entry_price
            
            # PnL portefeuille
            pnl_amount = self.equity * size * pnl_pct
            self.equity += pnl_amount
            equity_curve.append(self.equity)
            
            # Enregistrer le trade
            trade = Trade(
                bar_index=int(row['bar_index']),
                entry_price=entry_price,
                entry_regime=entry_regime,
                exit_price=exit_price,
                position_size=size,
                pnl_pct=pnl_pct,
                pnl_atr=actual_pnl_atr,
                exit_reason=exit_reason,
                bars_held=int(row.get('label_bars_to_outcome', 10)),
            )
            self.trades.append(trade)
        
        # Calcul des stats
        trades_df = pd.DataFrame([
            {
                'bar_index': t.bar_index,
                'entry_regime': t.entry_regime,
                'pnl_pct': t.pnl_pct,
                'r_multiple': t.r_multiple,
                'exit_reason': t.exit_reason,
            }
            for t in self.trades
        ])
        
        total_return = (self.equity - self.initial_equity) / self.initial_equity
        wins = sum(1 for t in self.trades if t.pnl_pct > 0)
        win_rate = wins / len(self.trades) if self.trades else 0.0
        
        # Drawdown
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Sharpe
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        
        stats = {
            'total_return_pct': total_return * 100,
            'final_equity': self.equity,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'avg_pnl_per_trade': np.mean([t.pnl_pct for t in self.trades]) * 100 if self.trades else 0.0,
        }
        
        return stats, trades_df, equity_curve

def run_backtest(dataset_path: str, output_path: str) -> tuple:
    """
    Lance le backtest complet.
    """
    df = pd.read_csv(dataset_path)
    
    print("\n" + "="*80)
    print("BACKTEST SIMULATION")
    print("="*80)
    
    engine = BacktestEngine(initial_equity=100000.0)
    stats, trades_df, equity_curve = engine.run(df)
    
    # Sauvegarder
    trades_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"\nTotal Return:       {stats['total_return_pct']:>8.2f}%")
    print(f"Final Equity:       ${stats['final_equity']:>12,.0f}")
    print(f"Win Rate:           {stats['win_rate']:>8.1%}")
    print(f"Total Trades:       {stats['total_trades']:>8.0f}")
    print(f"Max Drawdown:       {stats['max_drawdown_pct']:>8.2f}%")
    print(f"Sharpe Ratio:       {stats['sharpe_ratio']:>8.2f}")
    print(f"Avg PnL/Trade:      {stats['avg_pnl_per_trade']:>8.2f}%")
    
    return stats, trades_df, equity_curve
