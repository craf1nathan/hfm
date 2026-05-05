# src/04_label_engineering.py
"""
Calcule les labels en simulant sur barres OHLCV futures.
ZÉRO lookahead — utilise uniquement (T+1):end
"""

import pandas as pd
import numpy as np

class LabelEngineering:
    def __init__(self, ohlcv_df: pd.DataFrame, atr: np.ndarray):
        self.df = ohlcv_df
        self.atr = atr
    
    def simulate_trade(self, 
                      entry_idx: int,
                      direction: str,        # 'up' | 'down'
                      sl_atr: float = 0.5,
                      tp_atr: float = 3.0,
                      max_bars: int = 50) -> dict:
        """
        Simule un trade à partir de entry_idx
        """
        if entry_idx + max_bars >= len(self.df):
            return {'outcome': 'TIMEOUT', 'pnl_atr': 0.0, 'bars': max_bars}
        
        entry_price = self.df.iloc[entry_idx]['open']
        entry_atr = self.atr[entry_idx]
        
        future_high = self.df.iloc[entry_idx:entry_idx + max_bars]['high'].max()
        future_low = self.df.iloc[entry_idx:entry_idx + max_bars]['low'].min()
        
        if direction == 'up':
            sl_price = entry_price - (sl_atr * entry_atr)
            tp_price = entry_price + (tp_atr * entry_atr)
            
            if future_low <= sl_price:
                return {'outcome': 'LOSS', 'pnl_atr': -sl_atr, 
                       'bars': int(np.where(self.df.iloc[entry_idx:entry_idx + max_bars]['low'].values <= sl_price)[0][0])}
            elif future_high >= tp_price:
                return {'outcome': 'WIN', 'pnl_atr': tp_atr,
                       'bars': int(np.where(self.df.iloc[entry_idx:entry_idx + max_bars]['high'].values >= tp_price)[0][0])}
            else:
                return {'outcome': 'TIMEOUT', 'pnl_atr': (future_high - entry_price) / entry_atr,
                       'bars': max_bars}
        
        else:  # down
            sl_price = entry_price + (sl_atr * entry_atr)
            tp_price = entry_price - (tp_atr * entry_atr)
            
            if future_high >= sl_price:
                return {'outcome': 'LOSS', 'pnl_atr': -sl_atr,
                       'bars': int(np.where(self.df.iloc[entry_idx:entry_idx + max_bars]['high'].values >= sl_price)[0][0])}
            elif future_low <= tp_price:
                return {'outcome': 'WIN', 'pnl_atr': tp_atr,
                       'bars': int(np.where(self.df.iloc[entry_idx:entry_idx + max_bars]['low'].values <= tp_price)[0][0])}
            else:
                return {'outcome': 'TIMEOUT', 'pnl_atr': (entry_price - future_low) / entry_atr,
                       'bars': max_bars}
    
    def compute(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les labels
        """
        labels = []
        
        for _, row in features_df.iterrows():
            bar_idx = int(row['bar_index'])
            trend = row['trend']
            
            direction = 'up' if trend == 'ascending' else 'down'
            
            result = self.simulate_trade(bar_idx + 1, direction)
            
            labels.append({
                'bar_index': bar_idx,
                'label_outcome': result['outcome'],
                'label_pnl_atr': round(result['pnl_atr'], 4),
                'label_bars_to_outcome': result['bars'],
                'label_win': 1 if result['outcome'] == 'WIN' else 0,
            })
        
        labels_df = pd.DataFrame(labels)
        return pd.merge(features_df, labels_df, on='bar_index')

def run_label_engineering(features_path: str, ohlcv_path: str,
                         atr: np.ndarray, output_path: str) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    ohlcv = pd.read_csv(ohlcv_path)
    
    le = LabelEngineering(ohlcv, atr)
    dataset = le.compute(features)
    
    dataset.to_csv(output_path, index=False)
    print(f"✓ Added labels to {len(dataset)} samples")
    return dataset
