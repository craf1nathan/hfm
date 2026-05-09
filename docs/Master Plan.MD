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

### 1.1 Universe Selection
- **Target (X):** The instrument to trade.
- **Predictors (A–N):** Assets with demonstrable, logical relationships to X.
  - *Example:* X = EURUSD → A=GBPUSD, B=USDJPY, C=DAX, D=US10Y, E=VIX, F=GOLD.
- **Minimum** 3 predictors, **maximum** 8 to keep the search space manageable and avoid excessive multicollinearity.

### 1.2 Temporal Alignment (Master Clock)
- All assets are aligned to **Asset X’s M1 timestamp index** in UTC.
- **Rule:** Forward‑fill missing bars (market closed/illiquid) **only**. Never backward‑fill.
- Flag filled bars with `is_filled = 1`, leave gaps >5 bars as session‑breaks.
- Drop bars where X has no data.

### 1.3 Data Cleaning
| Issue | Handling |
|-------|----------|
| Price spikes (>3σ from 20‑bar median) | Winsorize or remove the bar |
| Zero‑volume bars | Flag, use only if spread is also missing |
| Duplicate timestamps | Keep last, log occurrence |
| Weekends / holidays | Remove entirely |

Add two **survival features** directly to the raw data:
- `spread_X`: Bid‑Ask spread of Asset X at every bar.
- `tick_activity_i`: Normalised tick volume of each predictor asset.

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
- **Training Window (In‑Sample)** – Years 1–2  
- **Validation Window (OOS‑1)** – Year 3 (used for early stopping & model selection)  
- **Final Test Window (OOS‑2)** – Year 4 (untouchable until Phase 6)

*This preserves a completely unseen period for the ultimate go/no‑go decision.*

---

## Phase 2: Feature Engineering

Engineer features for X **and** each predictor A–N. The GA selects which features, which lags, and which assets matter.

### 2.1 Per‑Asset Feature Catalogue

*All features are computed on an M1 bar‑by‑bar basis.*

**Price‑Based**
- `ret_1`, `ret_5`, `ret_15`, `ret_60` – log returns
- `bar_range` – (high‑low)/close
- `body_ratio` – |close‑open|/(high‑low)
- `upper_wick`, `lower_wick`
- VWAP deviation – (close‑VWAP)/ATR

**Momentum**
- RSI(period = 5, 14, 20)
- ROC(5), ROC(15)
- Bar‑over‑bar acceleration

**Volatility**
- ATR(10), ATR(20)
- Bollinger Band width (20,2)
- Historical volatility (rolling std of returns, 20‑bar)

**Structure**
- Higher‑High / Lower‑Low flags (HH, LL, LH, HL)
- Inside bar flag
- Engulfing flag
- Pivot distance (nearest S/R level)

**Microstructure & Volume**
- Volume ratio: vol / avg_vol(20)
- Volume trend: slope of vol MA(10)
- CVD proxy (if tick data available)

### 2.2 Cross‑Asset Features (The Predictive Edge)

**Correlation & Lead‑Lag**
- Rolling correlation (X, asset_i) for windows 5, 15, 60, 240
- Correlation change: corr_now – corr_lag(5)
- Cross‑correlation at lags [-5, -3, -1, 0, +1, +3, +5]

**Relative Strength**
- `(ret_i_5m – ret_X_5m)`, same for 1m, 15m
- Z‑score of price ratio (asset_i / asset_X) rolling 60 bars
- Spread velocity (derivative of price ratio)

**Divergence & Basket**
- Momentum divergence: RSI(asset_i) – RSI(X)
- Volume divergence: vol(asset_i) >> vol(X)
- Equal‑weight basket of A–N vs X: basket return, basket momentum, basket volatility

### 2.3 Lag Structure
Each feature is generated at lags: `t`, `t‑1`, `t‑3`, `t‑5`, `t‑15`.  
The GA will select which lag to use, capturing genuine lead‑lag relationships while discarding noise.

---

## Phase 3: Chromosome Architecture

We evolve a **weighted scoring model** that triggers a complete trade, plus all trade management and filter genes. This avoids the combinatorial explosion of logic‑gate‑based entry rules while still allowing exquisite control over risk and timing.

### 3.1 Gene Blocks

| Block | Parameters | Type & Range | Description |
|-------|------------|--------------|-------------|
| **1. Predictor Weights** | `w[1..N]` | Float [0.0, 1.0] | GA sets importance of each predictor A–N |
| **2. Feature Selection** | `feat_type[1..N]` | Categorical 1..K | Maps each predictor to a chosen base feature (from a curated list of 15–20) |
| **3. Feature Lag** | `lag[1..N]` | Integer [1, 15] | The lag applied to that feature |
| **4. Entry Thresholds** | `long_thresh` | Float [0.5, 3.0] | Composite score > thresh → BUY |
| | `short_thresh` | Float [‑3.0, ‑0.5] | Composite score < thresh → SELL |
| **5. Trade Management** | `sl_atr_mult` | Float [0.5, 5.0] | SL = entry ± (mult × ATR) |
| | `tp_atr_mult` | Float [1.0, 10.0] | TP = entry ± (mult × ATR) |
| | `min_rr` | Enforced ≥ 1.2 | GA discards chromosomes that violate |
| | `max_hold_bars` | Integer [10, 500] | Time exit |
| | `trailing_on` | Binary | Whether trail stop is active |
| | `trail_atr_mult` | Float [0.5, 3.0] | Trailing stop distance |
| | `breakeven_atr` | Float [0.5, 2.0] | ATR profit to move SL to entry |
| **6. Filters** | `session_mask` | Integer bitmask | 1 = trade this session, 0 = ignore |
| | `min_atr` | Float [0.0001, 0.0100] | Lowest volatility allowed |
| | `max_atr` | Float [0.0001, 0.0100] | Highest volatility allowed |
| | `max_spread_pips` | Float [1.0, 10.0] | Skip if spread > X pips |
| | `min_vol_ratio` | Float [0.5, 2.0] | Market activity threshold |
| | `volatility_gate` | Float [2.0, 5.0] | Block trading if 10‑min vol > gate × avg vol |
| **7. Position Sizing** | `risk_per_trade` | Float [0.5%, 3.0%] | Fixed fraction of equity |
| | `vol_scalar` | Binary | If ON, reduce size when ATR high |

**Entry Logic:**  
For every new bar, the composite signal is:
```
Score = Σ( w[i] × feat_i(t – lag[i]) )   for i = A..N
```
If `Score > long_thresh` → enter **LONG**; if `Score < short_thresh` → enter **SHORT**.  
Otherwise, no trade.

### 3.2 Parsimony Constraints
- **Maximum active predictors** (weights > 0.1) limited to 5. A heavy penalty is applied if exceeded.
- **Total active feature count** is indirectly constrained because only one feature per asset is selected.

---

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
