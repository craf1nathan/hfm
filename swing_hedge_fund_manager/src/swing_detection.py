# src/01_swing_detection.py
"""
Détecte les swings confirmés via lookback delay.
ZÉRO lookahead — utilise uniquement (t-lookback):t
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Swing:
    bar_index: int
    price: float
    direction: str          # 'high' | 'low'
    strength: int           # lookback
    atr_at_swing: float
    timestamp: datetime
    
    def to_dict(self):
        return {
            'bar_index': self.bar_index,
            'price': round(self.price, 5),
            'direction': self.direction,
            'strength': self.strength,
            'atr_at_swing': round(self.atr_at_swing, 5),
            'timestamp': self.timestamp.isoformat(),
        }

class SwingDetector:
    def __init__(self, lookback: int = 5, atr_period: int = 14):
        self.lookback = lookback
        self.atr_period = atr_period
    
    def calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        """ATR standard: causal (t <= T)"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        atr = np.zeros(len(tr))
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (self.atr_period - 1) + tr[i]) / self.atr_period
        
        return atr
    
    def detect(self, df: pd.DataFrame) -> list:
        """
        Détecte les swings confirmés.
        
        Pour chaque bar(t), check si bar(t-lookback) est un pivot:
        - HIGH: max(t-lookback-lb ... t-lookback+lb)
        - LOW: min(t-lookback-lb ... t-lookback+lb)
        
        Only when t >= t-lookback + lookback
        """
        df = df.copy()
        high = df['high'].values
        low = df['low'].values
        
        atr = self.calculate_atr(df)
        
        swings = []
        last_direction = None
        
        for t in range(2 * self.lookback, len(df)):
            # Check si bar(t - lookback) est un pivot
            confirm_idx = t - self.lookback
            
            is_high = True
            is_low = True
            
            # Vérifier sur ±lookback
            for j in range(confirm_idx - self.lookback, confirm_idx + self.lookback + 1):
                if j == confirm_idx or j < 0 or j >= len(df):
                    continue
                if high[j] >= high[confirm_idx]:
                    is_high = False
                if low[j] <= low[confirm_idx]:
                    is_low = False
            
            # Déterminer direction (alternance)
            if is_high and is_low:
                direction = 'high' if last_direction != 'high' else 'low'
            elif is_high:
                direction = 'high' if last_direction != 'high' else None
            elif is_low:
                direction = 'low' if last_direction != 'low' else None
            else:
                direction = None
            
            if direction:
                swing = Swing(
                    bar_index=confirm_idx,
                    price=high[confirm_idx] if direction == 'high' else low[confirm_idx],
                    direction=direction,
                    strength=self.lookback,
                    atr_at_swing=atr[confirm_idx],
                    timestamp=pd.to_datetime(df.iloc[confirm_idx]['time']),
                )
                swings.append(swing)
                last_direction = direction
        
        return swings

# Pipeline
def run_swing_detection(ohlcv_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(ohlcv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    detector = SwingDetector(lookback=5)
    swings = detector.detect(df)
    
    swings_df = pd.DataFrame([s.to_dict() for s in swings])
    swings_df.to_csv(output_path, index=False)
    
    print(f"✓ Detected {len(swings)} swings")
    return swings_df
