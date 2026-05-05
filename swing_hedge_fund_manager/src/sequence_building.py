# src/02_sequence_building.py
"""
Construit les legs et sequences depuis les swings.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class Leg:
    leg_id: str
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    direction: str          # 'up' | 'down'
    amplitude_price: float
    amplitude_atr: float
    duration_bars: int
    is_impulse: bool
    
    def to_dict(self):
        return {
            'leg_id': self.leg_id,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'start_price': round(self.start_price, 5),
            'end_price': round(self.end_price, 5),
            'direction': self.direction,
            'amplitude_price': round(self.amplitude_price, 5),
            'amplitude_atr': round(self.amplitude_atr, 4),
            'duration_bars': self.duration_bars,
            'is_impulse': int(self.is_impulse),
        }

@dataclass
class Sequence:
    sequence_id: str
    trend: str              # 'ascending' | 'descending' | 'ranging'
    shape: str              # 'converging' | 'diverging' | 'channel'
    efficiency_ratio: float
    legs: List[Leg] = field(default_factory=list)
    
    def to_dict(self):
        return {
            'sequence_id': self.sequence_id,
            'n_legs': len(self.legs),
            'trend': self.trend,
            'shape': self.shape,
            'efficiency_ratio': round(self.efficiency_ratio, 4),
            'start_idx': self.legs[0].start_idx if self.legs else 0,
            'end_idx': self.legs[-1].end_idx if self.legs else 0,
        }

class SequenceBuilder:
    def __init__(self, min_legs: int = 2, max_legs: int = 7):
        self.min_legs = min_legs
        self.max_legs = max_legs
    
    def build_legs(self, swings_df: pd.DataFrame, atr: np.ndarray) -> List[Leg]:
        """Construit les legs depuis les swings."""
        legs = []
        swings = swings_df.values
        
        for i in range(len(swings) - 1):
            s1 = swings[i]
            s2 = swings[i + 1]
            
            start_idx = int(s1[0])
            end_idx = int(s2[0])
            start_price = float(s1[1])
            end_price = float(s2[1])
            
            direction = 'up' if end_price > start_price else 'down'
            amplitude = abs(end_price - start_price)
            amplitude_atr = amplitude / atr[end_idx] if atr[end_idx] > 1e-6 else 0.0
            
            leg = Leg(
                leg_id=f"L{i}",
                start_idx=start_idx,
                end_idx=end_idx,
                start_price=start_price,
                end_price=end_price,
                direction=direction,
                amplitude_price=amplitude,
                amplitude_atr=amplitude_atr,
                duration_bars=end_idx - start_idx,
                is_impulse=(i % 2 == 0),  # Simple heuristique
            )
            legs.append(leg)
        
        return legs
    
    def build_sequences(self, legs: List[Leg]) -> List[Sequence]:
        """Crée des sequences en glissant une fenêtre."""
        sequences = []
        
        for n_legs in range(self.min_legs, min(self.max_legs + 1, len(legs) + 1)):
            for i in range(len(legs) - n_legs + 1):
                window_legs = legs[i:i + n_legs]
                
                # Déterminer trend
                net_move = window_legs[-1].end_price - window_legs[0].start_price
                if abs(net_move) < 1e-6:
                    trend = 'ranging'
                elif net_move > 0:
                    trend = 'ascending'
                else:
                    trend = 'descending'
                
                # Déterminer shape (simplifié)
                amps = [l.amplitude_atr for l in window_legs]
                if amps[-1] < amps[0]:
                    shape = 'converging'
                elif amps[-1] > amps[0]:
                    shape = 'diverging'
                else:
                    shape = 'channel'
                
                # Efficiency ratio
                total_amp = sum(abs(l.amplitude_atr) for l in window_legs)
                net_amp = abs(net_move) / (window_legs[0].amplitude_atr if window_legs else 1)
                efficiency = net_amp / total_amp if total_amp > 1e-6 else 0.0
                
                seq = Sequence(
                    sequence_id=f"S{i}_{n_legs}",
                    legs=window_legs,
                    trend=trend,
                    shape=shape,
                    efficiency_ratio=efficiency,
                )
                sequences.append(seq)
        
        return sequences

# Pipeline
def run_sequence_building(swings_path: str, atr: np.ndarray, output_path: str) -> tuple:
    swings_df = pd.read_csv(swings_path)
    
    builder = SequenceBuilder()
    legs = builder.build_legs(swings_df, atr)
    sequences = builder.build_sequences(legs)
    
    seqs_df = pd.DataFrame([s.to_dict() for s in sequences])
    seqs_df.to_csv(output_path, index=False)
    
    print(f"✓ Built {len(sequences)} sequences from {len(legs)} legs")
    return seqs_df, sequences
