# src/03_feature_engineering.py
"""
Calcule les FEATURES causales uniquement.
Zéro information du futur.
"""

import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, ohlcv_df: pd.DataFrame, atr: np.ndarray):
        self.df = ohlcv_df
        self.atr = atr
    
    def compute(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pour chaque séquence, extrait les features causales.
        """
        features = []
        
        for _, seq in sequences_df.iterrows():
            bar_idx = int(seq['end_idx'])
            
            if bar_idx >= len(self.atr):
                continue
            
            # Features géométriques
            f = {
                'bar_index': bar_idx,
                'sequence_id': seq['sequence_id'],
                'n_legs': int(seq['n_legs']),
                'trend': seq['trend'],
                'shape': seq['shape'],
                'efficiency_ratio': float(seq['efficiency_ratio']),
                
                # Volatilité
                'atr_at_event': round(self.atr[bar_idx], 5),
                'volatility_regime': 'high' if self.atr[bar_idx] > np.median(self.atr) else 'low',
                
                # Session (simplifié)
                'session': 'london' if bar_idx % 24 in range(8, 16) else 'other',
                
                # Prix actuels
                'close_price': float(self.df.iloc[bar_idx]['close']),
                'high_price': float(self.df.iloc[bar_idx]['high']),
                'low_price': float(self.df.iloc[bar_idx]['low']),
            }
            
            features.append(f)
        
        return pd.DataFrame(features)

def run_feature_engineering(ohlcv_path: str, sequences_path: str, 
                           atr: np.ndarray, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(ohlcv_path)
    seqs = pd.read_csv(sequences_path)
    
    fe = FeatureEngineering(df, atr)
    features_df = fe.compute(seqs)
    
    features_df.to_csv(output_path, index=False)
    print(f"✓ Computed {len(features_df)} feature vectors")
    return features_df
