# OddsJam EV Filter Research Plan (White Paper)

## Purpose

This document defines the **objectives, hypotheses, analytical framework, and decision rules** for the `oddsjam-ev-filters` repository.

The goal is to establish a **repeatable, closed-loop research system** for:
- Discovering new betting edges
- Testing them via exploratory TEST filters
- Quantifying true EV vs variance
- Iteratively improving a small set of PROD filters over time

This repo is not an archive of analysis — it is a **decision engine**.

If an insight does not result in:
- a new TEST filter,
- a PROD filter modification, or
- a filter being killed,

then the analysis is incomplete.

---

## Scope & Constraints

### In Scope
- **Pre-match Positive EV betting only**
- OddsJam-supported sportsbooks, exchanges, and prediction markets
- Historical analysis via OddsJam Bet Tracker CSV exports
- Scalable, rules-based betting strategies

### Explicitly Out of Scope
- Live / in-play betting
- Arbitrage betting
- Gut-feel decision making
- Pure projection-based modeling (unless explicitly added later)

Filters must remain **simple, monitorable, and operationally disciplined**.

---

## Filter Lifecycle Philosophy

### Filter Types

#### TEST Filters
- Exploratory and hypothesis-driven
- Paper traded only
- Designed to probe:
  - New sports or markets
  - Time-to-event windows
  - EV bands
  - Liquidity constraints
  - Book- or platform-specific behavior
- High tolerance for noise
- Zero tolerance for capital risk

#### PROD Filters
- Capital-deployed
- **Small in number (≤ 3)**
- Must demonstrate:
  - Statistically supported positive EV
  - Acceptable variance and drawdowns
  - Operational simplicity

**All PROD filters originate from TEST filters.**

---

## Closed-Loop Research Framework

This repository operates as a **continuous feedback loop**:

### 1. Explore
Identify new betting angles and hypotheses using TEST filters.

### 2. Evaluate
Quantitatively measure:
- EV vs realized ROI
- Variance and drawdowns
- Timing effects
- Liquidity effects
- Sport, market, and book-level performance

### 3. Decide
Make explicit, data-backed decisions:
- **Promote** → TEST → PROD
- **Patch** → refine constraints and re-test
- **Kill** → remove entirely

### 4. Iterate
- Update PROD filters
- Design new TEST filters
- Re-run notebooks on new data
- Repeat continuously

---

## Core Research Pillars

### 1. Filter-Level Profitability & Stability

**Key Questions**
- Which filters are truly positive EV?
- Which are riding variance?
- Which are structurally negative?

**Analyses**
- Profit, EV, ROI, bet count
- Stake-weighted metrics
- Odds distributions

**Decision Outputs**
- Promote / patch / kill filters
- Candidate TEST filters

---

### 2. Variance vs True Edge

**Key Questions**
- Where am I running hot vs cold?
- What is the volatility per filter?
- What sample size is required for confidence?

**Analyses**
- Rolling ROI
- Drawdowns
- Volatility-adjusted returns

**Decision Outputs**
- Continue testing
- Scale cautiously
- Kill early

---

### 3. Time-to-Event & Market Timing

**Key Questions**
- Optimal betting windows by sport?
- Does timing materially affect EV?
- Hour-of-day or day-of-week effects?

**Analyses**
- Time-to-event buckets
- Hour-of-day (EST)
- Day-of-week performance

**Decision Outputs**
- Time-based PROD rules
- Excluded betting windows

---

### 4. Liquidity & EV Bucket Optimization

**Key Questions**
- Which liquidity buckets are real vs misleading?
- Which EV bands convert to profit?

**Analyses**
- EV × liquidity × sport interactions
- Profitability by bucket

**Decision Outputs**
- Minimum liquidity rules
- EV band constraints

---

### 5. Sports & Market Diagnostics

**Key Questions**
- Why am I losing in NBA / NHL / NFL (2026)?
- Strategy flaw or variance?
- Do these trends exist in 2025 data?

**Analyses**
- Sport-level ROI
- Market-level breakdowns
- Cross-year comparisons

**Decision Outputs**
- Sport-specific PROD exclusions
- Sport-specific TEST filters

---

### 6. Book & Platform Sharpness

**Key Questions**
- Which books are sharp in which markets?
- Are losses platform-specific?

**Analyses**
- Book-level ROI
- Market–book interactions

**Decision Outputs**
- Book-specific rules
- Platform-aware filters

---

### 7. Player Props (Exploratory)

**Key Questions**
- Are player props exploitable top-down?
- Or do they require projections/sims?

**Analyses**
- Prop-specific EV performance
- Liquidity and depth diagnostics

**Decision Outputs**
- Go / no-go on props
- Dedicated prop-focused TEST filters

---

### 8. Explanatory Modeling

**Key Questions**
- What actually drives filter success?
- Which features matter most?

**Analyses**
- Tree-based models
- SHAP values
- Interaction effects

**Decision Outputs**
- Simplified heuristics
- Feature-driven filter redesigns

---

## Notebook Roadmap & Decision Checklists

### Notebook 01: Raw Preprocessing & EDA

**Checklist**
- [ ] Deduplicate bets across timestamps
- [ ] Standardize timestamps (EST)
- [ ] Validate stake, odds, and profit
- [ ] Tag liquidity buckets

**Decision Outputs**
- Data readiness sign-off
- Known data caveats

---

### Notebook 02: Filter-Level Performance

**Checklist**
- [ ] Profit, EV, ROI by filter
- [ ] Variance metrics
- [ ] Odds distributions

**Decision Outputs**
- Filters to promote / patch / kill
- New TEST filter ideas

---

### Notebook 03: Timing & Liquidity Analysis

**Checklist**
- [ ] Time-to-event buckets
- [ ] Hour/day effects
- [ ] Liquidity vs profit

**Decision Outputs**
- Timing rules
- Liquidity constraints

---

### Notebook 04: Sports, Books & Markets

**Checklist**
- [ ] Sport-level ROI
- [ ] Book-level sharpness
- [ ] Market diagnostics

**Decision Outputs**
- Sport/book exclusions
- Exploratory angles

---

### Notebook 05: Exploratory Modeling

**Checklist**
- [ ] Feature importance
- [ ] SHAP analysis
- [ ] Interaction effects

**Decision Outputs**
- Filter redesign recommendations
- New TEST filter hypotheses

---

## Data & Methodology Caveats

- Deduplicate bets placed at multiple timestamps
- Weight recent data more heavily than older data
- Liquidity fields may be missing or inconsistent in 2025
- Reported liquidity ≠ true market depth
- Order book visibility varies by exchange
- Paper trading stopped ~Jan 20
- Bankroll increased from $10K → $20K at that time

All conclusions must be interpreted with these constraints in mind.

---

## North Star

This repository exists to enforce **discipline**.

The objective is not more bets.
The objective is not more analysis.

The objective is **fewer, better bets — backed by evidence**.
