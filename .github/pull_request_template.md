## Summary

Describe the objective of this pull request and its main changes.

- 

## Change classification

- **Type:** feature / bugfix / refactor / research / documentation / chore
- **Size:** small / medium / large
- **Changes live betting filters or production execution?** yes / no
- **Related issue:** optional

## Research validity and reproducibility

Complete when the PR changes data processing, evaluation, metrics, or research conclusions. Otherwise write `Not applicable`.

- **Dataset snapshot / as-of date:**
- **Data grain:** execution / opportunity / both / other
- **Evaluation period and holdout strategy:**
- **Potential leakage, overlapping filters, and dependence:**
- **Missingness, liquidity, sample size, and uncertainty:**
- **Known limitations and interpretation constraints:**

## Testing

Describe the tests performed and their outcomes. Include relevant notebook outputs or screenshots when useful.

- [ ] `poetry check --lock`
- [ ] `poetry run ruff check .`
- [ ] `poetry run ruff format --check .`
- [ ] `poetry run pytest -q`
- [ ] `git diff --check`
- [ ] Relevant notebooks executed locally against the intended snapshot, if applicable

## Follow-up work

Document unresolved issues, methodological limitations, and deferred improvements. Use `None` if there are no known follow-ups.

## Merge checklist

- [ ] Target branch is `main`
- [ ] Scope and changes reviewed
- [ ] No unintended data, credentials, or generated artifacts included
- [ ] No unresolved merge conflicts
- [ ] GitHub Actions `ruff` and `tests` passed
- [ ] Ready to merge
