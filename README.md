# oddsjam-ev-filters

Evaluate long-term profitability of OddsJam Positive EV filters using exported Bet Tracker data.

## Setup

```bash
poetry install
poetry run pre-commit install
```

Put your OddsJam Bet Tracker export CSV in:

```
data/raw/
```

Start Jupyter:

```
poetry run jupyter lab
```

## Repo layout

- `src/oddsjam_ev/` – reusable analysis + modeling code  
- `notebooks/` – EDA and experiments  
- `data/raw/` – raw OddsJam exports (ignored by git)  
- `data/processed/` – derived datasets (ignored by git)
