Genetic Evolution for M1 Trading Systems
This master guide merges and refines the two sources into a single, authoritative reference. It covers everything from data preparation to live deployment, organized into key phases.

1. Data & Environment
✅ DO
Use tick or true M1 data — never coarser; always include spread, slippage, and commission in fitness evals.

Normalize inputs — price-based features should be returns, ratios, or z‑scores, never raw prices.

Use a massive dataset — minimum 2–3 years of clean M1 data (~1440 bars/day) for training; reserve additional years for validation.

Segment by market session — label Asian/London/NY to let the GA exploit session‑specific behavior or build session‑aware strategies.

Test across multiple instruments — avoid single‑symbol overfit; validate on correlated and uncorrelated pairs.

Inject real-world costs — model spread variability (especially news spikes), realistic slippage (1–3 pips on M1 is significant), and commissions.

❌ DON’T
Don’t rely on broker demo data — M1 demo feeds often have gaps, false ticks, and requotes.

Don’t ignore spread variability — a static spread assumption will hide strategies that die in live conditions.

Don’t backtest on clean/perfect data — you must degrade fills and include latency effects.

Don’t use the same data for training and final validation — ever. This includes snooping during development.

Don’t ignore data gaps — handle missing bars, holidays, and feed drops explicitly.

2. Chromosome & Strategy Architecture
✅ DO
Start simple — fewer genes = smaller search space = faster convergence; 8–12 genes is a sensible ceiling.

Constrain all parameters — set logical, bounded ranges (RSI period: 2–50, SMA: 2–200) to avoid absurd fits.

Separate signal logic from money management — evolve entry/exit and position sizing in separate passes or genome sections.

Include risk & filter genes — session filter, volatility regime filter, max daily loss, and position sizing (fixed fraction, ATR‑based).

Use simple building blocks — individual indicators, comparisons, and thresholds as genes; avoid monolithic logic.

Enforce logical consistency — design crossover and mutation to produce chromosomes that always generate a valid strategy (no trend+mean‑reversion contradictions).

❌ DON’T
Don’t hardcode magic numbers — e.g., entry times fixed to specific minutes overfit history; let the GA choose from time windows if needed.

Don’t mix incompatible logic in one chromosome — contradictory styles ruin interpretability and robustness.

Don’t use unbounded gene ranges — the GA will explore extreme noise‑fitting values.

Don’t exceed 8–12 free parameters — more leads to combinatorial explosion and guaranteed overfit on M1.

3. Fitness Function
✅ DO
Use multi‑objective fitness — combine risk‑adjusted metrics: Sharpe ratio × profit factor × (1 / max drawdown) plus trade count penalty.

Penalize low trade count — require at least 100 trades in‑sample (30–100 OOS) to achieve statistical relevance.

Penalize overtrading — spread/slippage destroys M1 strategies that fire too often; add a cost‑drag penalty.

Reward parameter stability — a “robustness penalty” that checks fitness degradation when parameters vary ±5–10% around the optimum.

Use bootstrapped fitness estimates — smooth out the M1 noise by evaluating on multiple random subsamples of the in‑sample period.

❌ DON’T
Don’t optimize on raw profit alone — you’ll get lottery‑ticket strategies (rare huge wins).

Don’t ignore drawdown — a strategy with a 
5
,
000
D
D
f
o
r
5,000DDfor1,000 profit is untradeable.

Don’t reward complexity — add a parsimony pressure that penalizes chromosome length or indicator count.

Don’t use short OOS windows — 1–2 weeks on M1 is meaningless; use multiple walk‑forward windows of at least 3–6 months.

Don’t leak the future — recalculate all indicators bar‑by‑bar in simulation; check for look‑ahead bias obsessively.

4. Genetic Algorithm Design
✅ DO
Preserve elites — keep the top 5–10% of each generation unchanged.

Use tournament or rank‑based selection — avoids the premature dominance of a few super‑fit individuals (roulette wheel risk).

Adapt mutation rates — start high for exploration, decay across generations for exploitation.

Run island models — 20+ independent populations that occasionally exchange top genes; prevents local optima traps.

Run multiple independent seeds — 10–50 evolutions; look for convergence to similar structures, not just similar fitness.

Limit generations — 50–200 is usually enough; beyond that, overfitting accelerates.

❌ DON’T
Don’t use tiny populations — minimum 50–200 individuals per island.

Don’t stop at first convergence — restart with fresh random seeds and compare results.

Don’t allow diversity to crash — monitor gene variance; if it drops too fast, increase mutation or island exchange.

Don’t over‑run generations — stop when fitness plateaus and OOS performance starts degrading.

5. Validation & Stress Testing
✅ DO
Walk‑forward optimization — rolling IS/OOS windows; minimum 5 OOS windows to build a reliability distribution.

Monte Carlo permutation tests — shuffle trade order, randomize entry timing; the edge should survive these distortions.

Parameter degradation test — shift the best genome’s parameters by ±5–10% and retest; fragile strategies collapse.

Cross‑symbol and cross‑session validation — what works on EURUSD M1 should show some edge (maybe weaker) on GBPUSD or during NY only.

Forward test on demo — at least 4–8 weeks of paper trading before real capital, because live data behavior differs from any backtest.

❌ DON’T
Don’t trust a single backtest — ever.

Don’t skip statistical significance — OOS trade count must be >30, ideally >100, to trust the fitness score.

Don’t confuse “no overfitting detected” with “no overfitting” — always assume the GA found something; your job is to prove it’s real.

Don’t use adjusted data incorrectly — for equities, split adjustment must be handled; for FX/crypto it’s less critical but still check.

6. Live Deployment & Monitoring
✅ DO
Run a extended forward test — 1–3 months minimum on a real broker demo/micro account.

Hard‑code kill switches — max daily loss, max consecutive losses, and automatic shutdown outside evolved regimes.

Size conservatively — 1–2% risk per trade; the GA found an edge, not certainty.

Build a retraining pipeline — M1 strategies degrade in weeks to months; plan for rolling re‑optimization (weekly/monthly).

Monitor for regime change — use volatility and session‑based filters to pause trading when conditions deviate.

❌ DON’T
Don’t deploy straight from backtest to live — the gap is huge on M1.

Don’t over‑leverage — even tiny execution delays or spread widening can flip a thin edge negative.

Don’t ignore execution latency — M1 signals decay fast; strategies should have an entry tolerance of at least one full bar (1 minute).

Don’t assume stability — shelf life is typically 3–6 months; schedule periodic re‑evolution.

🔑 Critical M1‑Specific Constants
text
Real M1 edge = signal quality + flawless execution + cost control

If (spread + slippage) > (avg_win × win_rate)
    → Strategy is dead regardless of GA fitness score
Factor	M1 Impact	Non‑negotiable Action
Spread	Very High	Hard‑code spread in every fitness eval; test with real spread and with spread=0 to see raw edge
Noise	Very High	Minimum reward:risk of 1.5:1; use longer indicator lookbacks or aggregate signal to M5/M15
Overfit risk	Extreme	Minimum 5 OOS windows + parameter degradation + Monte Carlo
Shelf life	Short (weeks‑months)	Build automated retraining pipeline
Trade frequency	High	Monitor cost drag; a strategy trading 50x/day needs a razor‑thin cost model
Latency	Critical	Design strategies that work with entry at next bar open, not tick‑perfect timing
🏗️ Recommended End‑to‑End Pipeline
text
1. DATA COLLECTION (multi‑year M1 + tick)
          ↓
2. FEATURE ENGINEERING (returns, normalized indicators, session flags)
          ↓
3. GENOME DESIGN (bounded params, capped complexity, separated signal/risk)
          ↓
4. MULTI‑OBJECTIVE FITNESS (Sharpe × Prop Factor × (1/Drawdown) × TradeCountPenalty)
          ↓
5. ISLAND MODEL EVOLUTION (20+ islands, 50–200 generations each)
          ↓
6. CONVERGENCE ANALYSIS (do islands find similar structure?)
          ↓
7. WALK‑FORWARD VALIDATION (5+ rolling OOS windows)
          ↓
8. STRESS TESTS (parameter degradation, Monte Carlo, spread variation)
          ↓
9. FORWARD TEST (demo/micro, 4–8 weeks)
          ↓
10. LIVE DEPLOYMENT (small size, kill switches, retraining schedule)
Golden Rule: On M1, the GA will always find something. Your mission is to ensure it’s genuine, repeatable edge — not an elaborate noise‑fitting exercise. True M1 edges are small, noisy, and fragile. That’s acceptable only if you control costs, validate relentlessly, and respect the strategy’s short shelf life.
