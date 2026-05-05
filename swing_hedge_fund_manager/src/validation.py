# src/09_validation.py
"""
Validation croisée et diagnostic des biais.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class ValidationFramework:
    """
    Walk-forward validation pour séries temporelles.
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
    
    def walk_forward_split(self, df: pd.DataFrame) -> list:
        """
        TimeSeriesSplit: train sur ancien, test sur nouveau.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        splits = []
        
        for train_idx, test_idx in tscv.split(df):
            splits.append({
                'train': df.iloc[train_idx],
                'test': df.iloc[test_idx],
                'train_range': (train_idx[0], train_idx[-1]),
                'test_range': (test_idx[0], test_idx[-1]),
            })
        
        return splits
    
    def validate(self, df: pd.DataFrame) -> dict:
        """
        Effectue la validation walk-forward.
        """
        splits = self.walk_forward_split(df)
        fold_results = []
        
        for fold_idx, split in enumerate(splits):
            train_df = split['train']
            test_df = split['test']
            
            # Stats par régime (sur train)
            regime_stats = {}
            for regime in train_df['regime_viterbi'].unique():
                regime_train = train_df[train_df['regime_viterbi'] == regime]
                regime_stats[regime] = {
                    'win_rate': (regime_train['label_win'] == 1).mean(),
                    'avg_pnl': regime_train['label_pnl_atr'].mean(),
                }
            
            # Évaluer sur test
            test_df_eval = test_df.copy()
            test_df_eval['expected_pnl'] = test_df_eval['regime_viterbi'].map(
                lambda r: regime_stats.get(r, {}).get('avg_pnl', 0.0)
            )
            
            fold_corr = np.corrcoef(
                test_df_eval['label_pnl_atr'],
                test_df_eval['expected_pnl']
            )[0, 1]
            
            fold_results.append({
                'fold': fold_idx,
                'train_range': split['train_range'],
                'test_range': split['test_range'],
                'correlation': fold_corr,
                'test_win_rate': (test_df['label_win'] == 1).mean(),
            })
        
        print("\nWalk-Forward Validation:")
        for res in fold_results:
            print(f"  Fold {res['fold']}: Corr={res['correlation']:.3f}, "
                 f"WR={res['test_win_rate']:.1%}")
        
        return fold_results

def run_validation(dataset_path: str) -> dict:
    df = pd.read_csv(dataset_path)
    
    validator = ValidationFramework(n_splits=5)
    results = validator.validate(df)
    
    avg_corr = np.mean([r['correlation'] for r in results])
    avg_wr = np.mean([r['test_win_rate'] for r in results])
    
    print(f"\nOverall Metrics:")
    print(f"  Avg Correlation: {avg_corr:.3f}")
    print(f"  Avg Win Rate:    {avg_wr:.1%}")
    
    return results
