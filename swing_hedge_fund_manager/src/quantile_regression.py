# src/07_quantile_regression.py
"""
Quantile Regression Forests par régime.

Pour chaque régime, prédis:
  P05, P25, P50, P75, P90 de next_swing_amplitude

Ceci permet une stratégie de sizing basée sur les centiles.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

class QuantileRegressionForest:
    """
    Approximation simple de Quantile Regression Forests.
    
    Idée: chaque arbre donne une distribution au leaf,
    on extrait les quantiles de la distribution empirique.
    """
    
    def __init__(self, n_trees: int = 50, max_depth: int = 10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.quantiles = [0.05, 0.25, 0.50, 0.75, 0.90]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit N regression trees (shallow).
        """
        for i in range(self.n_trees):
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=5
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
    
    def predict_quantiles(self, X: np.ndarray) -> dict:
        """
        Prédit les quantiles pour chaque sample.
        
        Returns:
            {quantile: (n_samples,) array}
        """
        # Prédictions de tous les arbres
        predictions = np.array([tree.predict(X) for tree in self.trees]).T
        # (n_samples, n_trees)
        
        quantile_preds = {}
        for q in self.quantiles:
            quantile_preds[q] = np.quantile(predictions, q, axis=1)
        
        return quantile_preds

class RegimeQuantileRegression:
    """
    Wrapper: fit une QRF par régime.
    """
    
    def __init__(self, regimes: list = None):
        self.regimes = regimes or ['BREAKOUT', 'PULLBACK', 'CHOPZONE', 'COMPRESSION']
        self.models = {r: QuantileRegressionForest() for r in self.regimes}
        self.quantiles = [0.05, 0.25, 0.50, 0.75, 0.90]
    
    def fit(self, df: pd.DataFrame, 
            feature_cols: list,
            target_col: str = 'label_pnl_atr'):
        """
        Fit les modèles par régime.
        """
        X_cols = feature_cols
        
        for regime in self.regimes:
            mask = df['regime_viterbi'] == regime
            n_samples = mask.sum()
            
            if n_samples < 10:
                print(f"  ⚠ Regime {regime}: only {n_samples} samples, skipping")
                continue
            
            X = df.loc[mask, X_cols].fillna(0).values
            y = df.loc[mask, target_col].values
            
            print(f"  Training {regime:15s}: {n_samples:4d} samples")
            self.models[regime].fit(X, y)
    
    def predict(self, df: pd.DataFrame, 
               feature_cols: list) -> pd.DataFrame:
        """
        Prédit les quantiles pour chaque sample selon son régime.
        """
        X_cols = feature_cols
        
        predictions = []
        
        for regime in self.regimes:
            mask = df['regime_viterbi'] == regime
            
            if not mask.any():
                continue
            
            X = df.loc[mask, X_cols].fillna(0).values
            quants = self.models[regime].predict_quantiles(X)
            
            for i, idx in enumerate(df[mask].index):
                pred = {
                    'bar_index': df.loc[idx, 'bar_index'],
                    'regime': regime,
                }
                for q in self.quantiles:
                    pred[f'pred_p{int(q*100):02d}'] = quants[q][i]
                
                predictions.append(pred)
        
        return pd.DataFrame(predictions)

def run_quantile_regression(regimes_df: str, output_path: str) -> pd.DataFrame:
    """
    Lance la quantile regression par régime.
    """
    df = pd.read_csv(regimes_df)
    
    # Features pour le modèle (causales, pas les labels!)
    feature_cols = [
        'n_legs', 'efficiency_ratio', 'atr_at_event',
        'trend', 'shape',  # Convert to numeric if needed
    ]
    
    # Convertir colonnes catégories en numériques
    df['trend_numeric'] = pd.Categorical(df['trend']).codes
    df['shape_numeric'] = pd.Categorical(df['shape']).codes
    
    feature_cols_numeric = [
        'n_legs', 'efficiency_ratio', 'atr_at_event',
        'trend_numeric', 'shape_numeric',
    ]
    
    # Train quantile regression
    print("Training Quantile Regression per Regime...")
    model = RegimeQuantileRegression()
    model.fit(df, feature_cols_numeric, target_col='label_pnl_atr')
    
    # Predict
    print("Predicting quantiles...")
    predictions = model.predict(df, feature_cols_numeric)
    
    # Merge avec dataset original
    result = pd.merge(df, predictions, on='bar_index', how='left')
    result.to_csv(output_path, index=False)
    
    print(f"✓ Quantile regression complete")
    print(f"\nSample predictions:")
    print(predictions.head(10))
    
    return result
