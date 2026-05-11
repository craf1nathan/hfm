# Final Master Plan: Cross‑Asset M1 Prediction Engine Using Genetic Evolution

*This plan integrates the two proposals into a single, actionable pipeline. It combines the rigorous feature engineering and validation framework of Proposal 1 with the pragmatic, weighted‑scoring chromosome and pragmatic data splitting of Proposal 2, and elevates both into a production‑ready system.*

---

## 🎯 Core Mission

```
Given M1 data from a basket of assets [A, B, C, … N]
→ Predict the full trade lifecycle of a target asset X:
    • Entry direction (LONG / SHORT)
    • Entry timing
    • Stop‑Loss, Take‑Profit (in ATR or pips)
    • Time exit & risk sizing
```

Every design choice is tested against the M1 reality: **tiny edges + high noise + extreme sensitivity to costs and execution latency**.

---

## 🏗️ High‑Level Architecture

```
 ┌─────────────────────────────────────────┐
 │          RAW M1 DATA                    │
 │   Assets: X + [A, B, ... N]            │
 └───────────────┬─────────────────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────┐
 │  PHASE 1 – Data Sync, Clean, Session   │
 └───────────────┬─────────────────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────┐
 │  PHASE 2 – Feature Engineering         │
 │  (Price, Vol, Cross‑Asset, Lags)       │
 └───────────────┬─────────────────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────┐
 │  PHASE 3 – GA Chromosome Design        │
 │  (Weighted Scoring + Trade Params)      │
 └───────────────┬─────────────────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────┐
 │  PHASE 4 – Fitness & Robustness Tests  │
 │  (Full Trade Simulation + Multi‑Obj)    │
 └───────────────┬─────────────────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────┐
 │  PHASE 5 – Evolution Engine & Islands  │
 └───────────────┬─────────────────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────┐
 │  PHASE 6 – Walk‑Forward & Stress Tests │
 │  (Monte Carlo, Parameter Stability)     │
 └───────────────┬─────────────────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────┐
 │  PHASE 7 – Paper Trading & Deployment  │
 │  (Kill Switches, Re‑Evolution)          │
 └─────────────────────────────────────────┘
```

---

## Phase 1: Data Pipeline – Synchronisation, Cleaning & Splitting
Need a data_config.json to store all parameters
### 1.1 Universe Selection
- **Target (X):** The instrument to trade. /to setup on data_config.json
- **Predictors (A–N):** Assets with demonstrable, logical relationships to X.
  - *Example:* X = EURUSD → A=GBPUSD, B=USDJPY, C=DAX, D=US10Y, E=VIX, F=GOLD. /to setup on data_config.json
- **Minimum** 3 predictors, **maximum** 8 to keep the search space manageable and avoid excessive multicollinearity.
- The data folder location should be configurable on data_config.json data config naming is like XAUUSD_M1.csv 
- The content of each ohlcv data is always like:
    timestamp,open,high,low,close,volume
    2025-12-07 22:21:00+00:00,63.815,63.815,63.802,63.81,0

### 1.2 Temporal Alignment (Master Clock)
- All assets are aligned to **Asset X’s M1 timestamp index** in UTC. 
- **Rule:** Forward‑fill missing bars (market closed/illiquid) **only**. Never backward‑fill.
- Flag filled bars with `is_filled = 1`, leave gaps >5 bars as session‑breaks.
- Drop bars where X has no data.

### 1.3 Data Cleaning
| Issue | Handling |
|-------|----------|
| Duplicate timestamps | Keep last, log occurrence |
| Weekends / holidays | Remove entirely |

### 1.4 Session Labelling
Each bar receives a session tag (used later as a filter gene):
```
0 = Dead zone      (00:00–02:00 UTC)
1 = Asian          (02:00–08:00)
2 = London         (08:00–12:00)
3 = Overlap        (12:00–16:00)
4 = New York       (16:00–20:00)
5 = Late NY        (20:00–24:00)
```

### 1.5 Sacred Data Splitting
Adopt a **three‑tier split** to completely isolate final validation:
- **Training Window (In‑Sample)** – Months 3 to 6 /to setup on pipeline_config.json
- **Validation Window (OOS‑1)** – Months 1 to 2 /to setup on pipeline_config.json
- **Final Test Window (OOS‑2)** – > 1 year /to setup on pipeline_config.json

*This preserves a completely unseen period for the ultimate go/no‑go decision.*

---

## Phase 2: Normalized Feature Engineering for GA-Driven Trade Cycles
Objective: Engineer bounded, regime-adaptive features for target asset X and predictors A–N that feed a Genetic Algorithm to produce complete trade signals: Direction, Entry, Valid Until, TP, SL, Risk Factor.

Design Principles
Principle	Implementation	Why
Scale Invariance	All features ∈ [-1, +1]	GA crossover/mutation operates uniformly across features
Regime Adaptability	Rolling z-scores + percentile ranks	Features remain stationary across volatility regimes
Outlier Resistance	tanh(z-score) instead of raw z-score	Flash crashes become -0.99, not -47.3
Cross-Asset Parity	Normalize before cross-asset operations	EUR/USD and BTC comparable on same scale
Smooth Transitions	Replace binary flags with scored gradients	GA learns nuance, not brittle thresholds
Temporal Decay	Exponential weights on rolling windows	Recent bars matter more than ancient history
Notation
text

Z(x, w)     = rolling z-score over w bars: (x - μ_w) / σ_w
T(x)        = tanh(x) — soft squashing to (-1, 1)
R(x, w)     = percentile rank in last w bars, mapped to [-1,1] via 2·rank - 1
ATR(n)      = Average True Range over n bars
EMA(x, α)   = Exponential moving average with decay α
2.1 Per-Asset Feature Catalogue
All features computed on M1 bars, all outputs ∈ [-1, +1] unless noted

A. Price-Based Features
Feature ID	Formula	Range	Description
ret_n	T(Z(log(close/close_n), 120))	(-1,1)	Returns at n∈{1,5,15,60}, z-scored over 2hr window
bar_range	R((high-low)/ATR(20), 60)	[-1,1]	Range percentile rank (ATR-normalized)
body_pct	`2·	close-open	/(high-low) - 1`
wick_upper	2·(high-max(O,C))/(high-low) - 1	[-1,1]	Upper shadow dominance
wick_lower	2·(min(O,C)-low)/(high-low) - 1	[-1,1]	Lower shadow dominance
vwap_dev	T(Z((close-VWAP)/ATR(20), 60))	(-1,1)	VWAP deviation (ATR-normalized, tanh-squashed)
price_pos	(close - low_60) / (high_60 - low_60) · 2 - 1	[-1,1]	Position in 60-bar range
B. Momentum Features
Feature ID	Formula	Range	Description
rsi_n	RSI(n)/50 - 1	[-1,1]	Remapped RSI, n∈{5,14,20}
roc_n	T(Z(ROC(n), 120))	(-1,1)	Rate of change, n∈{5,15,60}
mom_accel	T(Z(roc_5 - roc_5_lag1, 120))	(-1,1)	Momentum acceleration
mom_regime	EMA(rsi_14, 0.1) - 0 → map to [-1,1]	[-1,1]	Persistent momentum bias
trend_strength	T((close - EMA(close,20)) / ATR(20))	(-1,1)	Deviation from trend
C. Volatility Features
Feature ID	Formula	Range	Description
vol_regime	R(ATR(20)/EMA(ATR(20),80), 120)	[-1,1]	Is volatility expanding?
bb_squeeze	R(BBwidth(20,2)/ATR(20), 60)	[-1,1]	Bollinger Band compression
hvol_rank	R(std(ret,20), 120)	[-1,1]	Historical volatility percentile
vol_accel	T(Z(Δ(ATR(20)), 120))	(-1,1)	Volatility expansion rate
vol_ratio	T(ATR(10)/ATR(20) - 1)	(-1,1)	Short-term vs long-term vol
D. Structure Features (Smoothed, Not Binary)
Feature ID	Formula	Range	Description
hh_score	Σ[i=1→5] 0.5^i · I(high > high_i) → norm to [-1,1]	[-1,1]	Decayed higher-high agreement
ll_score	Σ[i=1→5] 0.5^i · I(low < low_i) → norm to [-1,1]	[-1,1]	Decayed lower-low agreement
inside_bar	2·exp(-(range/range_20)) - 1	(-1,1)	Strength of compression
engulf_score	T(log(body/body_1))	(-1,1)	Body size ratio (tanh-protected)
pivot_prox	2·exp(-min_dist(close, SR_levels)/ATR) - 1	(-1,1)	Proximity to S/R
swing_pos	R(close - EMA(close,60), 240)	[-1,1]	Position in swing cycle
E. Microstructure & Volume Features
Feature ID	Formula	Range	Description
vol_shock	T(Z(volume/EMA(volume,20), 60))	(-1,1)	Volume surge detector
vol_trend	T(Z(slope(log(volume),10), 120))	(-1,1)	Volume direction
vol_sync	`T(Z(corr(	ret	, volume, 20), 60))`
cvd_delta	T(Z(Σ sign(Δprice)·volume, 20) / √20)	(-1,1)	Cumulative delta proxy
spread_rank	R((ask-bid)/ATR(5), 60)	[-1,1]	Spread percentile (if available)
2.2 Cross-Asset Features (The Predictive Edge)
For each predictor asset A_i predicting target X

F. Correlation & Lead-Lag
Feature ID	Formula	Range	Description
corr_w	corr(ret_X, ret_Ai, w), w∈{5,15,60,240}	[-1,1]	Rolling correlation (native bounded)
corr_accel	T((corr_15 - corr_15_lag5) / σ(corr,60))	(-1,1)	Correlation regime shift
lead_lag_max	`argmax_τ∈[-5,+5]	xcorr(τ)	· sign(xcorr)` → /5
sync_score	EMA(I(sign(ret_X)==sign(ret_Ai)), 0.2) · 2 - 1	[-1,1]	Co-movement persistence
phase_diff	T((phase_X - phase_Ai) / π)	(-1,1)	Hilbert phase divergence (advanced)
G. Relative Strength (The Killer)
Feature ID	Formula	Range	Description
rel_ret_n	T(Z(ret_Ai_n - ret_X_n, 120))	(-1,1)	Relative performance, n∈{1,5,15,60}
ratio_z	T(Z(price_Ai/price_X, 120))	(-1,1)	Price ratio z-score
ratio_mom	T(Z(Δ(ratio_z), 60))	(-1,1)	Spread velocity
ratio_accel	T(Z(Δ²(ratio_z), 60))	(-1,1)	Spread acceleration
rel_strength	(rsi_Ai - rsi_X) → clamp [-1,1]	[-1,1]	Momentum divergence
H. Divergence & Basket
Feature ID	Formula	Range	Description
mom_div	clamp(rsi_Ai - rsi_X, -1, 1)	[-1,1]	Direct RSI divergence
price_mom_div	`sign(rel_ret_15) · (1 -	rel_strength	)`
vol_div	T(Z(vol_Ai/vol_X - 1, 60))	(-1,1)	Volume imbalance
basket_ret	T(Z(mean(ret_5, assets=A–N), 120))	(-1,1)	Basket momentum vs X
basket_mom	T(mean(rsi_14, assets=A–N)/50 - 1)	(-1,1)	Basket RSI
dispersion	R(std(ret_5, assets=A–N), 120)	[-1,1]	Inter-asset divergence
leadership	R(ret_X - median(ret_5, A–N), 120)	[-1,1]	Is X leading or lagging?
2.3 Lag Structure
Each feature is computed at 6 temporal offsets:

text

lags = {t, t-1, t-2, t-5, t-15, t-60}
Critical Implementation Detail:
Normalization parameters (μ, σ, percentile ranks) are computed once at time t, then applied to lagged values to prevent look-ahead bias.

Python

 # Correct
mu, sigma = rolling_stats(feature, window=120, end=t)
feature_lag5 = (raw_feature[t-5] - mu) / sigma

 # WRONG (look-ahead)
feature_lag5 = Z(raw_feature, 120)[t-5]  # Uses future data in past!
2.4 GA Chromosome Structure
Each gene encodes:

Python

Gene = {
    'feature_id': int,          # Index from catalogue (0-87)
    'asset_id': int,            # 0=X, 1-N=predictors
    'lag': int,                 # {0,1,2,5,15,60}
    'transform': str,           # {'raw', 'diff1', 'diff2', 'abs'}
    'weight': float             # [-2.0, +2.0]
}
Example Gene:
{'feature_id': 42 (rel_ret_5), 'asset_id': 3 (EUR/USD), 'lag': 5, 'transform': 'diff1', 'weight': 1.3}

Translation: "5 bars ago, EUR/USD's relative return was accelerating (diff1) strongly vs X, bullish signal"

2.5 Output: Trade Cycle Decoder
The GA produces a weighted sum of features → 4 intermediate signals → 6 outputs:

Python

# Intermediate (learned by GA)
signal_strength = Σ (w_i · feature_i)           # [-∞, +∞]
confidence = tanh(|signal_strength|)            # [0, 1]

# Final Outputs
direction       = sign(signal_strength)         # {-1, +1}
entry           = current_price                 # market
valid_until     = f(confidence, vol_regime)     # [2, 30] bars
tp              = entry + dir · f(conf, disp) · ATR   # adaptive
sl              = entry - dir · g(conf, div) · ATR    # adaptive  
risk_factor     = conf · (1 - dispersion) · (1 - vol_regime/2)  # [0.1, 1.0]
See Phase 3 for decoder formulas.

Feature Summary Stats
Category	Count	Key Predictors
Price-Based	7	ret_5, vwap_dev, price_pos
Momentum	5	rsi_14, trend_strength
Volatility	5	vol_regime, bb_squeeze
Structure	6	pivot_prox, swing_pos
Microstructure	5	vol_shock, cvd_delta
Cross-Correlation	5	lead_lag_max, sync_score
Relative Strength	5	rel_ret_5, ratio_mom
Divergence/Basket	7	dispersion, leadership, price_mom_div
Total Base	45	× 6 lags = 270 features/asset
With N predictors		270 · (N+1) features in search space
Validation Checklist
Python

def validate_features(df):
    for col in df.columns:
        assert df[col].min() >= -1.01, f"{col} below -1"
        assert df[col].max() <= 1.01, f"{col} above +1"
        assert df[col].isna().sum() == 0, f"{col} has NaN"
        assert df[col].std() > 0.01, f"{col} is flat"
    print("✅ All features validated: bounded, non-null, variant")

# Understanding the Code: 3-Head Architecture + Genetic Evolution

## Overview

The system uses **3 independent specialized models** (heads) that collaborate to produce a complete trading signal:

1. **HEAD 1 - Direction**: Predicts Long/Short/Neutral
2. **HEAD 2 - Sequence**: Predicts 6 spatio-temporal dimensions (entry, target, stop) × (price, time)
3. **HEAD 3 - Confidence**: Predicts reliability and sizes the position

Each head is a mini-ensemble of features (3–12 genes) with evolutionary weights ∈ [-1, +1].

---

## 3.1 Chromosome Structure

```python
Chromosome = {
    # The 3 heads
    'direction_head':  DirectionModel,
    'sequence_head':   SequenceModel,
    'confidence_head': ConfidenceModel,
    
    # Evolutionary metadata
    'meta': {
        'min_rr': float,
        'filters': {...}
    }
}
Total genome size: ~161 dimensions

Direction: 26 params (5 genes × 5 + 1 threshold)

Sequence: 108 params (12 genes × 9)

Confidence: 26 params (5 genes × 5 + 1 risk_base)

Meta: 1 param

3.2 HEAD 1 : Direction
Goal
Predict the movement direction: Long (+1), Short (-1), or Neutral (0).

Structure
python
DirectionModel = {
    'genes': [Gene₁, Gene₂, ..., Gene₅],  # max 5 genes
    'threshold': float ∈ [0.2, 0.5]        # minimum conviction level
}

Gene = {
    'asset_id':   int ∈ {0..N},           # 0 = target X, 1..N = predictors
    'feature_id': int ∈ {0..44},          # index from Phase 2 catalog
    'lag':        int ∈ {0,1,2,5,15,60},  # time lag (bars)
    'weight':     float ∈ [-1, +1],       # signed contribution
    'active':     bool                     # gene active/inactive
}
Calculation
python
# Raw score
score_raw = Σ (w_i · feature_i)  for all active genes

# Final normalization
score_dir = tanh(score_raw)  ∈ (-1, +1)

# Decision
if |score_dir| < threshold:
    direction = NEUTRAL
else:
    direction = sign(score_dir)  # +1 (LONG) or -1 (SHORT)
Typical Features
Feature	Asset	Why
rel_ret_5	EUR/USD	Strong relative momentum
lead_lag_max	GBP/USD	Detect lead-lag relationships
rsi_14	X (target)	Overbought/oversold
basket_ret	Basket A-N	Directional consensus
corr_15	DXY	Inverse USD correlation
Example
yaml
Active genes:
  - EUR/USD rel_ret_5  [lag=5]  : weight = +0.85
  - GBP/USD lead_lag   [lag=2]  : weight = +0.70
  - X rsi_14           [lag=0]  : weight = -0.40
  - Gold basket_ret    [lag=0]  : weight = +0.55
  - DXY corr_15        [lag=1]  : weight = -0.30

Threshold: 0.35

Current features (at t):
  rel_ret_5[t-5]  = +0.7
  lead_lag[t-2]   = +0.6
  rsi_14[t]       = +0.5
  basket_ret[t]   = +0.4
  corr_15[t-1]    = -0.2

Calculation:
  score_raw = 0.85×0.7 + 0.70×0.6 - 0.40×0.5 + 0.55×0.4 - 0.30×(-0.2)
            = 0.595 + 0.420 - 0.200 + 0.220 + 0.060
            = 1.095
  
  score_dir = tanh(1.095) = 0.799

Decision:
  |0.799| > 0.35 ✓
  sign(0.799) = +1
  → LONG
3.3 HEAD 2 : Sequence (Vector Model)
Goal
Simultaneously predict 6 dimensions:

entry_price : Limit order entry price

entry_time : How many bars until entry is hit

target_price : Take-profit price

target_time : Bars to reach target (after entry)

stop_price : Stop-loss price

stop_time : Not used (permanent stop)

Structure
python
SequenceModel = {
    'genes': [MultiGene₁, ..., MultiGene₁₂],  # max 12 genes
}

MultiGene = {
    'asset_id':   int ∈ {0..N},
    'feature_id': int ∈ {0..44},
    'lag':        int ∈ {0,1,2,5,15,60},
    'active':     bool,
    
    # Weight matrix: each gene contributes to all 6 dimensions
    'weights': {
        'entry_price':  float ∈ [-1, +1],
        'entry_time':   float ∈ [-1, +1],
        'target_price': float ∈ [-1, +1],
        'target_time':  float ∈ [-1, +1],
        'stop_price':   float ∈ [-1, +1],
        'stop_time':    float ∈ [-1, +1]   # optional
    }
}
Each gene has 6 weights instead of one, enabling per-dimension specialization.

Calculation
python
# 1. For each dimension, compute a composite score
for dim in ['entry_price', 'entry_time', 'target_price', 
            'target_time', 'stop_price']:
    
    score_raw[dim] = Σ (gene.weights[dim] · feature_i) for active genes
    score[dim] = tanh(score_raw[dim])  ∈ (-1, +1)

# 2. Map scores to real values

# Entry
entry_offset = 0.3 + 1.5 · sigmoid(score['entry_price'])  # [0.3, 1.8] ATR
entry_time   = 2 + 28 · sigmoid(score['entry_time'])      # [2, 30] bars

# Target
target_mult  = 2.0 + 5.0 · sigmoid(score['target_price']) # [2.0, 7.0] ATR
target_time  = 10 + 110 · sigmoid(score['target_time'])   # [10, 120] bars

# Stop
stop_mult    = 0.8 + 2.0 · sigmoid(score['stop_price'])   # [0.8, 2.8] ATR

# 3. Calculate absolute prices
ATR = ATR(20)[t]

if direction == LONG:
    entry_price  = current_price - entry_offset · ATR
    target_price = entry_price + target_mult · ATR
    stop_price   = entry_price - stop_mult · ATR
else:  # SHORT
    entry_price  = current_price + entry_offset · ATR
    target_price = entry_price - target_mult · ATR
    stop_price   = entry_price + stop_mult · ATR

# 4. Total expiry
expiry = max(entry_time + target_time + 10, 30)  # safety margin
Typical Features by Dimension
Dimension	Relevant Features	Logic
entry_price	bar_range, vwap_dev, inside_bar	Wide range → wait for retracement
entry_time	vol_regime, momentum	High vol → fast move
target_price	vol_regime, bb_squeeze, trend_strength	Squeeze → big breakout
target_time	mom_accel, trend_strength	Acceleration → fast
stop_price	dispersion, price_mom_div, vol_accel	Disagreement → wide stop
Example
yaml
Active genes: 8

Gene 1: X vol_regime [lag=0]
  Weights:
    entry_price:  -0.30  # high vol → closer entry (reactive)
    entry_time:   -0.50  # high vol → fast entry
    target_price: +0.80  # high vol → far target (big move)
    target_time:  -0.60  # high vol → fast target
    stop_price:   +0.40  # high vol → wide stop (breathing room)

Gene 2: X bb_squeeze [lag=1]
  Weights:
    entry_price:  -0.60  # compression → close entry (imminent breakout)
    entry_time:   -0.40  # squeeze → fast
    target_price: +0.90  # strong squeeze → explosive breakout
    target_time:  +0.30  # breakout may be gradual
    stop_price:   -0.20  # squeeze → high conviction → tight stop

... (4 other genes)

Features (at t):
  vol_regime     = +0.5
  bb_squeeze     = +0.9
  rel_ret_5      = +0.6
  trend_strength = +0.7

Calculations:
  score['entry_price'] = tanh(-0.30×0.5 -0.60×0.9 + ...) = -0.42
  score['entry_time']  = tanh(-0.50×0.5 -0.40×0.9 + ...) = -0.55
  score['target_price']= tanh(+0.80×0.5 +0.90×0.9 + ...) = +0.88
  score['target_time'] = tanh(-0.60×0.5 +0.30×0.9 + ...) = -0.20
  score['stop_price']  = tanh(+0.40×0.5 -0.20×0.9 + ...) = +0.15

Mapping:
  entry_offset = 0.3 + 1.5·sigmoid(-0.42) = 0.90 ATR
  entry_time   = 2 + 28·sigmoid(-0.55) = 12 bars
  target_mult  = 2.0 + 5.0·sigmoid(+0.88) = 5.55 ATR
  target_time  = 10 + 110·sigmoid(-0.20) = 60 bars
  stop_mult    = 0.8 + 2.0·sigmoid(+0.15) = 1.88 ATR

Absolute prices (LONG):
  Current_price = 1.08500, ATR = 0.00012 (12 pips)
  entry_price  = 1.08500 - 0.90×0.00012 = 1.08392
  target_price = 1.08392 + 5.55×0.00012 = 1.09058
  stop_price   = 1.08392 - 1.88×0.00012 = 1.08166
  expiry = max(12 + 60 + 10, 30) = 82 bars
Interpretation:

Squeezing market → close entry, large TP

Average volatility → moderate timing

Strong momentum → far TP, fast timing

Moderate agreement → moderate SL

3.4 HEAD 3 : Confidence
Goal
Assess signal reliability and size the position.

Structure
python
ConfidenceModel = {
    'genes': [Gene₁, ..., Gene₅],         # max 5 genes
    'risk_base': float ∈ [0.3%, 2.0%]     # evolutionary parameter
}

# Gene structure same as Direction HEAD
Calculation
python
score_raw = Σ (w_i · feature_i) for active genes
score_conf = tanh(score_raw)  ∈ (-1, +1)

confidence = (score_conf + 1) / 2  ∈ [0, 1]

prob_success = 0.35 + 0.45 · confidence  # [35%, 80%]
risk_pct = risk_base · confidence         # [0%, 2.0%]
Typical Features
Feature	Asset	Why
sync_score	EUR/USD	Synchronization → confidence
dispersion	Basket	Low dispersion → agreement → confidence
corr_15	EUR/USD	Clear correlation → predictability
leadership	X	X leads market → fresh signal
direction_score	Feedback	Strong direction signal → reinforces confidence
Note: direction_score is the raw score from HEAD 1, providing feedback.

Example
yaml
Active genes:
  - EUR/USD sync_score    : +0.90
  - Basket dispersion     : -0.85
  - EUR/USD corr_15       : +0.75
  - X leadership          : +0.60
  - direction_score       : +0.50

risk_base: 1.8%

Features:
  sync_score      = +0.7
  dispersion      = +0.2
  corr_15         = +0.6
  leadership      = +0.5
  direction_score = +0.799

Calculation:
  score_raw = 0.90×0.7 -0.85×0.2 +0.75×0.6 +0.60×0.5 +0.50×0.799 = 1.610
  score_conf = tanh(1.610) = 0.922
  confidence = (0.922+1)/2 = 0.961

Outputs:
  prob_success = 35% + 45%×0.961 = 78.2%
  risk_pct = 1.8% × 0.961 = 1.73%
Interpretation: Predictors synchronized, in agreement, with strong directional signal → very high confidence (96.1%) → large position (1.73% of capital).

3.5 Final Output : Complete Trade Signal
python
class TradeSignal:
    # HEAD 1: Direction
    direction: int              # +1 (LONG) or -1 (SHORT)
    
    # HEAD 2: Sequence
    sequence: {
        'entry': {'price': float, 'time_expected': int},
        'target': {'price': float, 'time_expected': int},
        'stop': {'price': float, 'time_expected': None},
        'expiry': int
    }
    
    # HEAD 3: Confidence
    confidence: {
        'score': float,
        'prob_success': float,
        'risk_pct': float
    }
    
    # Metadata
    current_price: float
    atr: float
    rr: float
    timestamp: datetime
Example JSON
json
{
  "timestamp": "2024-01-15T10:15:00",
  "current_price": 1.08500,
  "atr": 0.00012,
  "direction": "LONG",
  "sequence": {
    "entry": {"price": 1.08392, "time_expected": 12},
    "target": {"price": 1.09058, "time_expected": 60},
    "stop": {"price": 1.08166, "time_expected": null},
    "expiry": 90
  },
  "confidence": {"score": 0.961, "prob_success": 0.782, "risk_pct": 0.0173},
  "metrics": {"entry_offset_atr": 0.90, "target_distance_atr": 5.55, "stop_distance_atr": 1.88, "rr": 3.54},
  "orders": {
    "entry": {"type": "LIMIT_BUY", "price": 1.08392, "valid_until": "2024-01-15T10:27:00"},
    "target": {"type": "LIMIT_SELL", "price": 1.09058},
    "stop": {"type": "STOP_LOSS", "price": 1.08166}
  }
}
3.6 Modeling with Genetic Evolution
3.6.1 Chromosome Representation
python
class Chromosome:
    def __init__(self):
        # HEAD 1: Direction (max 5 genes)
        self.dir_genes = [Gene(...), ...]
        self.dir_threshold = 0.35
        
        # HEAD 2: Sequence (max 12 genes with 6 weights each)
        self.seq_genes = [MultiGene(...), ...]
        
        # HEAD 3: Confidence (max 5 genes)
        self.conf_genes = [Gene(...), ...]
        self.risk_base = 0.018
        
        # Meta
        self.min_rr = 1.5
        self.filters = {...}
3.6.2 Vector Encoding
python
def encode(self) -> np.ndarray:
    """Encode chromosome into a vector for GA"""
    vector = []
    # Direction: 5 genes × 5 params + 1 threshold = 26 dims
    # Sequence: 12 genes × 9 params = 108 dims
    # Confidence: 5 genes × 5 params + 1 risk_base = 26 dims
    # Meta: 1 dim
    return np.array(vector)  # Total: 161 dimensions
3.6.3 Genetic Operators
Initialization
Randomly generate population with active/inactive genes, random weights within bounds.

Crossover Types
By head: Each head inherited from either parent

Uniform per gene: Mix genes from both parents

Arithmetic: Child = α·parent1 + (1-α)·parent2 for continuous parameters

Mutation
Gaussian mutation on weights, discrete jumps for asset/feature/lag, toggle active flag.

3.6.4 Fitness Function
python
def fitness(chromo, historical_data):
    # Simulate trades over historical data
    # Compute metrics: PnL, winrate, avg win/loss, Sharpe, max drawdown
    # Composite fitness = expectancy * sqrt(n_trades) + 10*Sharpe + ...
    return fitness_score
3.6.5 Main Evolutionary Loop
python
def evolve(train_data, pop_size=100, generations=500):
    population = initialize_population()
    for gen in range(generations):
        fitnesses = evaluate(population)
        elite = select_best(population, fitnesses, elite_pct=0.10)
        new_population = elite.copy()
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2) if random() < crossover_rate else copy(parent1)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    return best_chromosome
3.7 Constraints & Validation
Structural Constraints
Direction: 2–5 active genes

Sequence: 4–12 active genes

Confidence: 2–5 active genes

All weights ∈ [-1, +1]

No duplicate (asset, feature, lag) within a head

Signal Validation
For LONG: stop < entry < target

For SHORT: target < entry < stop

Minimum R:R ≥ 1.2 (configurable)

Time consistency: 0 < entry_time < expiry, 0 < target_time < expiry - entry_time

Confidence ∈ [0,1], risk_pct ∈ [0.3%, 2.0%]

3.8 Example Evolution Progress
Generation 0 (Random)
Best fitness: 120

Average: -15

Random direction, poor R:R, low winrate

Generation 50
Best fitness: 340

Discovery: EUR/USD rel_ret_5 at lag=5 predicts direction well

bb_squeeze → large TP + tight stop

Synchronization → high confidence

Generation 200
Best fitness: 820

Trades: 187, Winrate: 58%, Avg R:R: 2.8, Sharpe: 1.9

Robust combination found: accurate direction, adaptive TP/SL, intelligent position sizing

3.9 Visualization
python
def plot_evolution_history(history):
    # Plot best and average fitness over generations
    # Also plot improvement per generation
    pass
3.10 Summary of Evolvable Dimensions
Component	Evolvable Parameters	Impact
Direction HEAD	5 × (asset, feat, lag, weight) + threshold	Directional accuracy
Sequence HEAD	12 × (asset, feat, lag, 6 weights)	TP/SL/timing quality
Confidence HEAD	5 × (asset, feat, lag, weight) + risk_base	Position sizing
Total	~161 dimensions	Complete strategy
The GA explores a space of 10^100+ possible combinations to find the optimal trading strategy.

3.11 Advantages of This Architecture
Advantage	Explanation
Separation of concerns	Each head solves a specific subproblem
Interpretability	You can see why a decision is made
Specialization	Features relevant per dimension (vol → TP, dispersion → SL)
Robustness	Cross-validation between heads (strong direction + low confidence → skip)
Scalability	Easy to add a 4th head (e.g., "predicted future volatility")
Targeted evolution	Mutation can improve TP without breaking direction
Fast convergence	161 dimensions vs millions of if/then rules
End of Phase 3 Document

## Phase 4: The Fitness Function – The Heart of the System

The fitness function simulates **every possible trade** produced by a chromosome on the Training data (Years 1–2) and returns a scalar fitness. It is composed of three layers:

### Layer 1 – Hard Eligibility (Trade‑Off Kill)
| Condition | Action |
|-----------|--------|
| Total trades < 200 in Training | Fitness = 0 |
| Max drawdown > 30% | Fitness = 0 |
| Win rate < 30% | Fitness = 0 |
| Average hold time < 3 bars | Fitness = 0 |

*If a chromosome fails any of these, it is not considered further.*

### Layer 2 – Base Multi‑Objective Score
For eligible chromosomes, calculate the following using the **exact full trade simulation** (spread + 1 pip slippage per trade):
```
Base_Fitness = (Sharpe × 1.5) 
             + (ProfitFactor × 1.0) 
             + (WinRate × 0.5) 
             - (MaxDD × 2.0) 
             - (Complexity × 0.5) 
             + TradeCountBonus
```
- **Complexity** = number of active predictors (w > 0.1).
- **TradeCountBonus** = log(trade_count / 200) to reward statistical robustness.

### Layer 3 – Penalties & Robustness Multiplier
**Penalties (subtracted from Base_Fitness):**
- **Edge Decay Penalty:** Split Training data into two chronological halves. If Profit Factor drops by >30% from first half to second half → fitness × 0.5.
- **Consecutive Loss Penalty:** If longest losing streak > 8 → fitness × 0.8.
- **Over‑asset penalty:** Extra penalty for each active predictor beyond 3, gradually increased.

**Robustness Multiplier (applied after penalties):**  
The chromosome must pass three internal stress tests. If it passes all, the fitness is multiplied by 1.2 (a bonus for robustness). If it fails any, the fitness is multiplied by 0.2 (a severe handicap).

*Stress Tests (run inside fitness evaluation):*
1. **Parameter Stability** – Nudge all continuous genes by ±10%; the resulting fitness must not drop >20%.
2. **Time Slice Consistency** – The Training data is split into 3 equal sub‑periods; the strategy must be profitable in ≥2 of them.
3. **Spread Stress** – Double the spread; the strategy must still have a positive expectancy.

The final fitness used by the GA:
```
Final_Fitness = (Base_Fitness – Penalties) × Robustness_Multiplier
```

---

## Phase 5: Evolution Engine & Island Model

| GA Parameter | Value | Rationale |
|--------------|-------|-----------|
| Population size | 500 | Large multi‑asset search space needs diversity |
| Generations | 50–300 (early stopping) | Plateau detection stops early |
| Selection | Tournament (k=5) | Balances selection pressure and diversity |
| Crossover | Uniform (50% per gene) | Effective for heterogeneous genes |
| Mutation rate | Adaptive: 0.10 → 0.02 | High exploration early, fine‑tuning later |
| Elitism | Top 5% preserved | Prevents regression |
| Island model | 3 independent populations of 500 | Migration every 20 generations (top 5% emigrate) |
| Early stopping | No improvement in best fitness for 50 generations | Prevents over‑iteration |

**Validation fitness** (on Year 3 data) is computed for the elite chromosomes after each generation. If the Validation fitness stops improving while Training fitness continues to rise, the run is stopped – this signals overfitting.

---

## Phase 6: Walk‑Forward Analysis & Final Validation

### 6.1 Walk‑Forward in‑Depth
Run a rolling walk‑forward on the combined Year 1–3 data to confirm time‑persistent edge:
```
IS:  6 months
OOS: 1 month
Step: 1 month forward
Minimum: 12 IS/OOS pairs
```
Calculate:
- **Walk‑Forward Efficiency (WFE)** = OOS annualised return / IS annualised return. Target WFE > 0.5.
- **Walk‑Forward Factor (WFF)** = Σ(OOS profits) / Σ(IS profits). Target WFF > 0.3.
- **Percent profitable OOS windows** ≥ 60%.

### 6.2 Monte Carlo Robustness
Apply to the concatenated OOS trade list from the walk‑forward:
- Shuffle trade order 1,000× → distribution of Sharpe, MaxDD.
- Bootstrap resample 1,000×.
- Report **5th percentile Sharpe**, **95th percentile MaxDD**, **Probability of ruin (DD > 50%)**.

**Acceptance criteria:**
- 5th pctl Sharpe > 0.3
- 95th pctl MaxDD < 25%
- P(ruin) < 5%

### 6.3 The “Untouchable” Final Test
Run the single best champion **once** on the Year 4 data (OOS‑2).  
- If the equity curve and metrics stay within the Monte Carlo confidence range, the strategy passes.
- If it fails, **do not tweak**. Return to Phase 2 with new features or asset selection.

---

## Phase 7: Paper Trading & Live Deployment

### 7.1 Paper Trading (Forward Test)
- Run on a live demo account for **minimum 4 weeks**.
- Log every attempted trade, including the assumed backtest entry price vs. actual fill.
- If realised slippage > 1.5× the backtest assumption, the strategy is rejected.
- Compare daily PnL trajectory with the OOS simulation; divergence beyond 2 standard deviations triggers a halt.

### 7.2 Hardcoded Kill Switches (Outside the GA)
- Daily loss > 3% of equity → halt.
- Weekly loss > 7% → halt.
- 6 consecutive losing trades → halt.
- Live rolling 30‑day Sharpe < 0.2 → halt.
- Win rate over last 100 trades drops below 50% of backtest win rate → halt.
- Any predictor data feed disconnected > 5 minutes → halt.

### 7.3 Scheduled Re‑Evolution
- **Every 3 months**, automatically re‑run the GA on the latest rolling 2‑year data window.
- The current champion is inserted as one of the “elite” seeds in the new population to preserve learned structure.
- Old chromosome archives are kept for comparison; switch only if the new chromosome shows **statistically significant** improvement on the latest 3‑month OOS slice.

---

## ⚠️ M1‑Specific Pitfalls & Mitigations (Hard‑Coded)

| Pitfall | Mitigation |
|--------|------------|
| **Multicollinearity** (GA stacks EURUSD, GBPUSD, etc.) | Feature block normalises and differences; fitness penalises correlated pairs (enforce max correlation ≤ 0.9). |
| **Execution latency** | Pre‑compute feature vectors seconds before the bar closes (e.g., at xx:59.5). Execute on next bar open. |
| **News/event wipeouts** | Volatility gate gene blocks trading when 10‑min vol > gate × average. Optionally connect economic calendar API. |
| **Look‑ahead bias** | All features are computed on closed bars; simulation engine steps bar‑by‑bar with next‑open execution. |
| **Overfitting on lucky sequences** | Minimum 200 trades, time‑slice consistency, parameter stability, and the untouchable OOS‑2 test guard this. |

---

## 🛠️ Technology Stack

```
Language        : Python (with Cython/numba‑optimised hot paths)
GA Framework    : DEAP + custom simulation engine
Data Storage    : PostgreSQL + TimescaleDB
Feature Calc    : Pandas, NumPy, TA‑Lib, bottleneck
Backtesting     : Custom vectorised/event‑driven engine (can leverage vectorbt)
Visualisation   : Plotly / Grafana dashboards
Broker API      : MetaTrader 5 Python API (or FIX for institutional)
Monitoring      : Prometheus + Grafana with alerting
Versioning      : Git for code; dedicated DB table for chromosomes
```

---

## ✅ Final Execution Timeline

```
Weeks 1–2  : Data collection, synchronisation, session labelling
Week 3     : Feature engineering (standalone library)
Weeks 4–5  : Chromosome definition + trade simulation engine
Weeks 6–7  : GA engine build, island model, multi‑objective fitness
Week 8     : Walk‑forward, Monte Carlo, final Year 4 test (happens exactly once)
Week 9–12  : Paper trading (minimum 4 weeks)
Month 4+   : Gradual live deployment, scheduled re‑evolution
```

---

**Core Principle:**  
*On M1, a cross‑asset GA will always find patterns. Your sole job is to verify, through brutal statistical and stress testing, that the edge is genuine and not simply a reflection of noise, cost, or latency. Build the system to distrust itself—and you’ll have a real trading edge.*
