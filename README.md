# oddsjam-ev-filters

Evaluate, stress-test, and continuously improve Positive EV betting filters using exported OddsJam Bet Tracker data.

This repository serves as a **repeatable research framework** for:
- Identifying durable betting edges
- Eliminating negative-EV leakage
- Designing **exploratory TEST filters**
- Promoting validated TEST filters into a small, high-quality set of PROD filters

Primary focus is **pre-match Positive EV betting only** (no live betting).

---

## Core Goals

- Reduce overtrading and filter sprawl
- Converge toward **≤ 3 production filters**
- Continuously identify new betting angles via structured exploration
- Separate **variance** from **true edge**
- Make filter changes evidence-driven, not reactive

This repo is intentionally designed so that **new OddsJam Bet Tracker CSVs can be dropped in and the entire analysis re-run with minimal code changes**.

---

## Setup

```bash
poetry install
poetry run pre-commit install
```

Put your OddsJam Bet Tracker export CSV in:

```bash
data/raw/
```

Start Jupyter:

```bash
poetry run jupyter lab
```

---

## Repo layout

- `src/oddsjam_ev/` – reusable analysis + plotting code  
- `notebooks/` – EDA, diagnostics, and experiments
- `data/raw/` – raw OddsJam exports (ignored by git)  
- `data/processed/` – derived datasets (ignored by git)
- `docs/` – research plans and decision frameworks
- `figures/` – generated plots (optional, ignored)

---

## Research Framework

This repository is structured as a **closed-loop system**:

### 1. Explore
Identify new betting angles and hypotheses via exploratory **TEST filters**.

### 2. Evaluate
Measure EV, variance, timing, liquidity, and book-level performance.

### 3. Decide
Promote, patch, or kill filters based on data — not gut feel.

### 4. Iterate
Update **PROD filters** and design new **TEST filters**.

For the detailed research plan and decision framework, see:

**[docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)**
