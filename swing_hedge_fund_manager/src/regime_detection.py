# src/06_regime_detection.py
"""
Hidden Markov Model pour détecter les régimes de marché.

Régimes:
  0: BREAKOUT (trend fort, volatilité haute, efficacité haute)
  1: PULLBACK (trend moyen, volatilité moyenne)
  2: CHOPZONE (ranging, volatilité basse, efficacité basse)
  3: COMPRESSION (ranging serré, setup probable)

Features utilisées pour le HMM:
  - efficiency_ratio (qualité structurelle)
  - volatility_regime (ATR-based)
  - trend_strength (net_amplitude / total_amplitude)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class RegimeState:
    regime_id: int
    name: str      # P(Z_t | obs)
    
    NAMES = {
        0: 'BREAKOUT',
        1: 'PULLBACK',
        2: 'CHOPZONE',
        3: 'COMPRESSION',
    }
    
    def __post_init__(self):
        self.name = self.NAMES.get(self.regime_id, 'UNKNOWN')

class HiddenMarkovModel:
    """
    Simple HMM avec Viterbi + Forward-Backward.
    
    States: 0=BREAKOUT, 1=PULLBACK, 2=CHOPZONE, 3=COMPRESSION
    """
    
    def __init__(self, n_states: int = 4, n_features: int = 3):
        self.n_states = n_states
        self.n_features = n_features
        
        # Transition matrix: P(state_t | state_t-1)
        # Diagonale haute = persistance
        self.transition = np.array([
            [0.70, 0.20, 0.05, 0.05],  # BREAKOUT → (BO, PB, CH, CM)
            [0.20, 0.60, 0.10, 0.10],  # PULLBACK → 
            [0.10, 0.15, 0.60, 0.15],  # CHOPZONE →
            [0.10, 0.25, 0.20, 0.45],  # COMPRESSION →
        ])
        
        # Emission probabilities: P(obs | state)
        # Gaussiennes paramétrées par (mu, sigma)
        self.emission_means = np.array([
            [0.75, 0.70, 0.80],         # BREAKOUT: high eff, high vol, high trend
            [0.55, 0.50, 0.60],         # PULLBACK: mid eff, mid vol, mid trend
            [0.35, 0.30, 0.35],         # CHOPZONE: low eff, low vol, low trend
            [0.40, 0.25, 0.40],         # COMPRESSION: mid-low eff, very low vol, mid trend
        ])
        
        self.emission_stds = np.array([
            [0.10, 0.15, 0.12],         # BREAKOUT: tight
            [0.12, 0.15, 0.15],         # PULLBACK: medium
            [0.10, 0.10, 0.10],         # CHOPZONE: tight low
            [0.08, 0.08, 0.12],         # COMPRESSION: very tight vol
        ])
        
        # Initial state distribution
        self.initial = np.array([0.25, 0.25, 0.25, 0.25])
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalise X en [0, 1] par rolling standardization.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            X_norm: (n_samples, n_features) ∈ [0, 1]
        """
        X_norm = np.zeros_like(X)
        window = 20  # rolling window
        
        for col in range(X.shape[1]):
            for i in range(len(X)):
                start = max(0, i - window)
                end = i + 1
                
                vals = X[start:end, col]
                v_min = vals.min()
                v_max = vals.max()
                
                if v_max > v_min:
                    X_norm[i, col] = (X[i, col] - v_min) / (v_max - v_min)
                else:
                    X_norm[i, col] = 0.5
        
        return np.clip(X_norm, 0, 1)
    
    def emission_probability(self, obs: np.ndarray, state: int) -> float:
        """
        P(obs | state) — produit gaussien sur features
        
        Args:
            obs: (n_features,)
            state: int ∈ [0, n_states)
        
        Returns:
            float > 0
        """
        mu = self.emission_means[state]
        sigma = self.emission_stds[state]
        
        # Gaussienne univariée
        prob = 1.0
        for i in range(len(obs)):
            exp = -0.5 * ((obs[i] - mu[i]) / sigma[i]) ** 2
            prob *= (1.0 / (sigma[i] * np.sqrt(2 * np.pi))) * np.exp(exp)
        
        return max(1e-10, prob)  # avoid 0
    
    def forward(self, observations: np.ndarray) -> tuple:
        """
        Forward algorithm: compute alpha[t, state] = P(obs[0:t], z[t])
        
        Args:
            observations: (n_samples, n_features)
        
        Returns:
            (alpha, scaling_factors)
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        scale = np.zeros(T)
        
        # t=0
        for s in range(self.n_states):
            alpha[0, s] = self.initial[s] * self.emission_probability(observations[0], s)
        
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        
        # t > 0
        for t in range(1, T):
            for s in range(self.n_states):
                alpha[t, s] = self.emission_probability(observations[t], s) * \
                              np.sum(alpha[t-1] * self.transition[:, s])
            
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t]
        
        return alpha, scale
    
    def backward(self, observations: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Backward algorithm: compute beta[t, state] = P(obs[t:T] | z[t])
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # t=T-1
        beta[-1] = 1.0
        
        # t < T-1 (backward)
        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                beta[t, s] = np.sum(
                    self.transition[s] * 
                    np.array([self.emission_probability(observations[t+1], s2) 
                             for s2 in range(self.n_states)]) *
                    beta[t+1]
                )
            
            beta[t] /= scale[t + 1]
        
        return beta
    
    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """
        Viterbi: most likely path of hidden states
        """
        T = len(observations)
        viterbi_path = np.zeros(T, dtype=int)
        
        # DP table: log probabilities
        dp = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)
        
        # t=0
        for s in range(self.n_states):
            dp[0, s] = np.log(self.initial[s] + 1e-10) + \
                      np.log(self.emission_probability(observations[0], s) + 1e-10)
        
        # t > 0
        for t in range(1, T):
            for s in range(self.n_states):
                # arg max over previous states
                prev_scores = dp[t-1] + np.log(self.transition[:, s] + 1e-10)
                best_prev = np.argmax(prev_scores)
                
                dp[t, s] = prev_scores[best_prev] + \
                          np.log(self.emission_probability(observations[t], s) + 1e-10)
                backpointer[t, s] = best_prev
        
        # Backtrack
        viterbi_path[-1] = np.argmax(dp[-1])
        for t in range(T - 2, -1, -1):
            viterbi_path[t] = backpointer[t + 1, viterbi_path[t + 1]]
        
        return viterbi_path
    
    def predict(self, observations: np.ndarray) -> tuple:
        """
        Prédit les régimes (Viterbi path + Forward probs)
        
        Returns:
            (viterbi_path, smoothed_probs)
        """
        X_norm = self.normalize_features(observations)
        
        viterbi_path = self.viterbi(X_norm)
        alpha, scale = self.forward(X_norm)
        beta = self.backward(X_norm, scale)
        
        # Smoothed probabilities: P(z_t | obs[0:T])
        smoothed = (alpha * beta)
        smoothed /= smoothed.sum(axis=1, keepdims=True)
        
        return viterbi_path, smoothed

def run_regime_detection(dataset_path: str, output_path: str) -> pd.DataFrame:
    """
    Lance la détection de régimes sur le dataset.
    """
    df = pd.read_csv(dataset_path)
    
    # Features du HMM
    # 1. efficiency_ratio (proxy pour trend_strength)
    # 2. volatility_regime (ATR-based)
    # 3. n_legs normalized (structure complexity)
    
    X = np.column_stack([
        df['efficiency_ratio'].values,
        (df['atr_at_event'] / df['atr_at_event'].rolling(20).mean()).fillna(0.5).values,
        (df['n_legs'] / 7.0).values,  # normalize to [0, 1]
    ])
    
    # Fit HMM
    hmm = HiddenMarkovModel(n_states=4, n_features=3)
    viterbi_path, smoothed_probs = hmm.predict(X)
    
    # Ajouter au dataset
    regime_names = ['BREAKOUT', 'PULLBACK', 'CHOPZONE', 'COMPRESSION']
    df['regime_viterbi'] = [regime_names[s] for s in viterbi_path]
    df['regime_smoothed_prob'] = smoothed_probs[np.arange(len(smoothed_probs)), viterbi_path]
    
    for i, name in enumerate(regime_names):
        df[f'regime_prob_{name}'] = smoothed_probs[:, i]
    
    df.to_csv(output_path, index=False)
    
    print(f"✓ Regime detection complete")
    print(f"\nRegime distribution (Viterbi):")
    print(df['regime_viterbi'].value_counts())
    
    return df
