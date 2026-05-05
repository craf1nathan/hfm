# src/01_swing_detection.py
"""
Dual timeframe swing detection: Fast (3) + Slow (7)
Avec synchronisation et scoring de potentialité.
ZÉRO lookahead — utilise uniquement (t-lookback):t
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

@dataclass
class Swing:
    bar_index: int
    price: float
    direction: str          # 'high' | 'low'
    timeframe: str          # 'fast' | 'slow'
    lookback: int
    strength: int
    atr_at_swing: float
    timestamp: datetime
    
    def to_dict(self):
        return {
            'bar_index': self.bar_index,
            'price': round(self.price, 5),
            'direction': self.direction,
            'timeframe': self.timeframe,
            'lookback': self.lookback,
            'strength': self.strength,
            'atr_at_swing': round(self.atr_at_swing, 5),
            'timestamp': self.timestamp.isoformat(),
        }

class DualSwingDetector:
    """Détecte les swings sur deux timeframes simultanément."""
    
    def __init__(self, 
                 fast_lookback: int = 3,
                 slow_lookback: int = 7,
                 atr_period: int = 14):
        self.fast_lookback = fast_lookback
        self.slow_lookback = slow_lookback
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
    
    def detect_swings(self, 
                     df: pd.DataFrame,
                     lookback: int,
                     timeframe: str) -> List[Swing]:
        """Détecte les swings pour un timeframe donné."""
        
        high = df['high'].values
        low = df['low'].values
        atr = self.calculate_atr(df)
        
        swings = []
        last_direction = None
        
        # On peut confirmer à partir de 2*lookback
        for t in range(2 * lookback, len(df)):
            confirm_idx = t - lookback
            
            # Check si bar(confirm_idx) est un pivot local
            is_high = True
            is_low = True
            
            for j in range(confirm_idx - lookback, confirm_idx + lookback + 1):
                if j == confirm_idx or j < 0 or j >= len(df):
                    continue
                if high[j] >= high[confirm_idx]:
                    is_high = False
                if low[j] <= low[confirm_idx]:
                    is_low = False
            
            # Alternance stricte
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
                    timeframe=timeframe,
                    lookback=lookback,
                    strength=lookback,
                    atr_at_swing=atr[confirm_idx],
                    timestamp=pd.to_datetime(df.iloc[confirm_idx]['time']),
                )
                swings.append(swing)
                last_direction = direction
        
        return swings
    
    def detect_dual(self, df: pd.DataFrame) -> Tuple[List[Swing], List[Swing], np.ndarray]:
        """
        Détecte les swings sur Fast et Slow simultanément.
        
        Returns:
            (fast_swings, slow_swings, atr)
        """
        atr = self.calculate_atr(df)
        
        fast_swings = self.detect_swings(df, self.fast_lookback, 'fast')
        slow_swings = self.detect_swings(df, self.slow_lookback, 'slow')
        
        return fast_swings, slow_swings, atr


# Legacy support for single lookback
class SwingDetector:
    def __init__(self, lookback: int = 5, atr_period: int = 14):
        self.lookback = lookback
        self.atr_period = atr_period
        self.detector = DualSwingDetector(fast_lookback=lookback, slow_lookback=lookback, atr_period=atr_period)
    
    def calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        return self.detector.calculate_atr(df)
    
    def detect(self, df: pd.DataFrame) -> list:
        fast, slow, atr = self.detector.detect_dual(df)
        return fast  # Return fast swings for legacy


def run_swing_detection(ohlcv_path: str, 
                       output_path_fast: str,
                       output_path_slow: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Lance la détection dual-swing."""
    
    print("Loading OHLCV...")
    df = pd.read_csv(ohlcv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    print("Detecting Fast (3) and Slow (7) swings...")
    detector = DualSwingDetector(fast_lookback=3, slow_lookback=7)
    fast_swings, slow_swings, atr = detector.detect_dual(df)
    
    # Sauvegarde
    fast_df = pd.DataFrame([s.to_dict() for s in fast_swings])
    slow_df = pd.DataFrame([s.to_dict() for s in slow_swings])
    
    fast_df.to_csv(output_path_fast, index=False)
    slow_df.to_csv(output_path_slow, index=False)
    
    print(f"✓ Fast swings: {len(fast_swings)} → {output_path_fast}")
    print(f"✓ Slow swings: {len(slow_swings)} → {output_path_slow}")
    
    return fast_df, slow_df, atr


def run_swing_detection_single(ohlcv_path: str, output_path: str, lookback: int = 5) -> pd.DataFrame:
    """Legacy: détection simple pour compatibilité."""
    df = pd.read_csv(ohlcv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    detector = SwingDetector(lookback=lookback)
    swings = detector.detect(df)
    
    swings_df = pd.DataFrame([s.to_dict() for s in swings])
    swings_df.to_csv(output_path, index=False)
    
    print(f"✓ Detected {len(swings)} swings (lookback={lookback})")
    return swings_df
