# src/02_sequence_building.py
"""
Construit les séquences Fast et Slow séparément,
puis synchronise quand Fast swing = confirmation de Slow.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Leg:
    leg_id: str
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    direction: str
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
    timeframe: str
    legs: List[Leg]
    trend: str
    shape: str
    efficiency_ratio: float
    
    def to_dict(self):
        return {
            'sequence_id': self.sequence_id,
            'timeframe': self.timeframe,
            'n_legs': len(self.legs),
            'trend': self.trend,
            'shape': self.shape,
            'efficiency_ratio': round(self.efficiency_ratio, 4),
            'start_idx': self.legs[0].start_idx if self.legs else 0,
            'end_idx': self.legs[-1].end_idx if self.legs else 0,
        }

@dataclass
class SynchronizedEvent:
    """Event quand Fast swing se confirme avec contexte Slow."""
    bar_index: int
    event_id: str
    fast_sequence_id: str
    fast_n_legs: int
    fast_trend: str
    fast_shape: str
    fast_efficiency: float
    slow_sequence_id: Optional[str]
    slow_n_legs: Optional[int]
    slow_trend: Optional[str]
    slow_shape: Optional[str]
    slow_efficiency: Optional[float]
    slow_age_bars: int
    trend_aligned: bool
    shape_aligned: bool
    potentiality_score: float
    
    def to_dict(self):
        return {
            'bar_index': self.bar_index,
            'event_id': self.event_id,
            'fast_sequence_id': self.fast_sequence_id,
            'fast_n_legs': self.fast_n_legs,
            'fast_trend': self.fast_trend,
            'fast_shape': self.fast_shape,
            'fast_efficiency': round(self.fast_efficiency, 4),
            'slow_sequence_id': self.slow_sequence_id or '',
            'slow_n_legs': self.slow_n_legs or 0,
            'slow_trend': self.slow_trend or '',
            'slow_shape': self.slow_shape or '',
            'slow_efficiency': round(self.slow_efficiency or 0, 4),
            'slow_age_bars': self.slow_age_bars,
            'trend_aligned': int(self.trend_aligned),
            'shape_aligned': int(self.shape_aligned),
            'potentiality_score': round(self.potentiality_score, 4),
        }


class DualSequenceBuilder:
    """Construit les séquences pour chaque timeframe."""
    
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
                is_impulse=(i % 2 == 0),
            )
            legs.append(leg)
        
        return legs
    
    def build_sequences(self, legs: List[Leg], timeframe: str) -> List[Sequence]:
        """Crée des sequences en glissant une fenêtre."""
        sequences = []
        
        for n_legs in range(self.min_legs, min(self.max_legs + 1, len(legs) + 1)):
            for i in range(len(legs) - n_legs + 1):
                window_legs = legs[i:i + n_legs]
                
                net_move = window_legs[-1].end_price - window_legs[0].start_price
                if abs(net_move) < 1e-6:
                    trend = 'ranging'
                elif net_move > 0:
                    trend = 'ascending'
                else:
                    trend = 'descending'
                
                amps = [l.amplitude_atr for l in window_legs]
                if len(amps) >= 2 and amps[-1] < amps[0]:
                    shape = 'converging'
                elif len(amps) >= 2 and amps[-1] > amps[0]:
                    shape = 'diverging'
                else:
                    shape = 'channel'
                
                total_amp = sum(abs(l.amplitude_atr) for l in window_legs)
                net_amp = abs(net_move) / (window_legs[0].amplitude_atr if window_legs else 1)
                efficiency = net_amp / total_amp if total_amp > 1e-6 else 0.0
                
                seq = Sequence(
                    sequence_id=f"S_{timeframe}_{i}_{n_legs}",
                    timeframe=timeframe,
                    legs=window_legs,
                    trend=trend,
                    shape=shape,
                    efficiency_ratio=efficiency,
                )
                sequences.append(seq)
        
        return sequences
    
    def synchronize(self, fast_swings_df: pd.DataFrame, slow_swings_df: pd.DataFrame,
                   fast_sequences: List[Sequence], slow_sequences: List[Sequence]) -> List[SynchronizedEvent]:
        """Synchronise Fast et Slow."""
        events = []
        fast_swings = fast_swings_df['bar_index'].values
        slow_swings = slow_swings_df['bar_index'].values
        
        fast_seq_by_end = {s.legs[-1].end_idx: s for s in fast_sequences}
        slow_seq_by_end = {s.legs[-1].end_idx: s for s in slow_sequences}
        
        for fast_bar in fast_swings:
            fast_seq = fast_seq_by_end.get(fast_bar)
            if not fast_seq:
                continue
            
            slow_bars_before = slow_swings[slow_swings < fast_bar]
            slow_seq = None
            slow_age = 0
            if len(slow_bars_before) > 0:
                last_slow_bar = slow_bars_before[-1]
                slow_seq = slow_seq_by_end.get(last_slow_bar)
                slow_age = fast_bar - last_slow_bar
            
            trend_aligned = (fast_seq.trend == slow_seq.trend) if slow_seq else False
            shape_aligned = (fast_seq.shape == slow_seq.shape) if slow_seq else False
            
            pot_score = self._calculate_potentiality(fast_seq, slow_seq, trend_aligned, shape_aligned, slow_age)
            
            event = SynchronizedEvent(
                bar_index=int(fast_bar),
                event_id=f"EV_{fast_bar}",
                fast_sequence_id=fast_seq.sequence_id,
                fast_n_legs=len(fast_seq.legs),
                fast_trend=fast_seq.trend,
                fast_shape=fast_seq.shape,
                fast_efficiency=fast_seq.efficiency_ratio,
                slow_sequence_id=slow_seq.sequence_id if slow_seq else None,
                slow_n_legs=len(slow_seq.legs) if slow_seq else None,
                slow_trend=slow_seq.trend if slow_seq else None,
                slow_shape=slow_seq.shape if slow_seq else None,
                slow_efficiency=slow_seq.efficiency_ratio if slow_seq else None,
                slow_age_bars=slow_age,
                trend_aligned=trend_aligned,
                shape_aligned=shape_aligned,
                potentiality_score=pot_score,
            )
            events.append(event)
        
        return events
    
    def _calculate_potentiality(self, fast_seq: Sequence, slow_seq: Optional[Sequence],
                               trend_aligned: bool, shape_aligned: bool, slow_age: int) -> float:
        score = 0.0
        if trend_aligned:
            score += 0.4
        if shape_aligned:
            score += 0.2
        if fast_seq.efficiency_ratio > 0.6:
            score += 0.2
        if slow_age < 10:
            score += 0.2
        elif slow_age < 20:
            score += 0.1
        return min(1.0, score)


def run_sequence_building(fast_swings_path: str, slow_swings_path: str,
                         atr: np.ndarray, output_path: str) -> Tuple[pd.DataFrame, List[SynchronizedEvent]]:
    """Lance la construction des séquences dual."""
    
    print("Loading swings...")
    fast_swings = pd.read_csv(fast_swings_path)
    slow_swings = pd.read_csv(slow_swings_path)
    
    print("Building sequences...")
    builder = DualSequenceBuilder()
    
    fast_legs = builder.build_legs(fast_swings, atr)
    slow_legs = builder.build_legs(slow_swings, atr)
    
    fast_seqs = builder.build_sequences(fast_legs, 'fast')
    slow_seqs = builder.build_sequences(slow_legs, 'slow')
    
    print(f"  Fast: {len(fast_legs)} legs → {len(fast_seqs)} sequences")
    print(f"  Slow: {len(slow_legs)} legs → {len(slow_seqs)} sequences")
    
    print("Synchronizing Fast + Slow...")
    events = builder.synchronize(fast_swings, slow_swings, fast_seqs, slow_seqs)
    
    events_df = pd.DataFrame([e.to_dict() for e in events])
    events_df.to_csv(output_path, index=False)
    
    print(f"✓ Synchronized {len(events)} events → {output_path}")
    
    return events_df, events


class SequenceBuilder:
    """Legacy support."""
    def __init__(self, min_legs: int = 2, max_legs: int = 7):
        self.dual_builder = DualSequenceBuilder(min_legs, max_legs)
    
    def build_legs(self, swings_df: pd.DataFrame, atr: np.ndarray) -> List[Leg]:
        return self.dual_builder.build_legs(swings_df, atr)
    
    def build_sequences(self, legs: List[Leg]) -> List[Sequence]:
        sequences = self.dual_builder.build_sequences(legs, 'single')
        for seq in sequences:
            seq.timeframe = 'single'
            seq.sequence_id = seq.sequence_id.replace('_single_', '_')
        return sequences
