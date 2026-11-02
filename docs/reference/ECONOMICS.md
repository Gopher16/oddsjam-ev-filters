# Economics, eligibility and denominators

## USD, reported profit, and fee uncertainty

USD is the user's confirmed source-level currency assumption, recorded as
`user_confirmed_2026-10-22`. Source amounts are parsed into separate reported USD
columns without rounding or modifying the raw strings. No conversion is applied.

OddsJam's `bet_profit` is retained as reported profit. There is no explicit source
fee/fill breakdown or fee-verification field. `fee_inclusion` remains `unverified`
and `fee_verified_net_roi_eligible` is false for every record. No estimated fees
are deducted, avoiding a potential double charge.

`reported_roi_eligible` permits supported ordinary pre-match single candidates on
configured fixed-odds or exchange products with finite positive stake, finite
profit, nonzero finite odds, supported terminal status, and explicit false values
for free-bet, odds-boost, and live-bet flags. Missing/unknown flags do not establish
ordinary wagers. Middle/arbitrage/unknown types remain unsupported. Execution mode
is retained as an independent evidence dimension; unknown execution is never
silently described as funded. Economic summaries separate assumed real, paper,
and unknown cohorts.

For fixed-odds singles, modeled profit is a consistency check using the existing
American-odds multiplier:

- Won: stake × profit multiplier.
- Lost: −stake.
- Half won: half the modeled winning profit.
- Half lost: −half the stake.

The default mismatch tolerance is USD 0.01. A mismatch excludes the row from the
reported ROI cohort but never overwrites reported profit. No model is applied to
exchange returns: they remain source-reported diagnostics with unknown fee status.
No actual payout is inferred from `potential_payout`; it remains potential.

Pending/open/unknown settlement, refunds/cancellations, unlinked Pick'em plays,
explicit tickets, unknown products, promotions, invalid amounts, and unsupported
bet types remain visible with explicit exclusion reasons. Refund/cancellation
amounts are preserved but excluded from the new denominator by policy. Legacy
notebook metric definitions are unchanged.

Reported ROI equals the sum of eligible reported profit divided by the sum of
stake for those exact same eligible candidates. Excluded amounts never enter
either side of that ratio. Empty eligible populations have null totals and ROI,
not a false zero result. Do not call these estimates verified bankroll returns.

## Metrics and denominators

Within each funding/product population:

| Measure | Definition |
| --- | --- |
| Canonical reported ROI | Sum of canonical-eligible profit / sum of canonical-eligible stake; baseline for reconciliation. |
| Stake-weighted ROI | Sum of profit / sum of stake on the paired diagnostic cohort. |
| Equal-stake mean return | Mean of each eligible candidate's profit / stake. |
| Fixed-unit notional profit | Sum of unit returns times the declared hypothetical USD unit. |
| Stake concentration | Largest and top-five stake shares, HHI, and inverse HHI. |
| Profit concentration | Largest absolute profit / sum of absolute profit; zero denominator remains unavailable. |
| Event concentration | Largest known namespace/event stake / event-covered stake. |
| Bankroll-relative stake | Source stake / evidenced setting bankroll, only for verified application with known, supported bankroll and eligible stake. |

All ROI/return values are **fractions**, not percentage points. Inverse stake HHI is a
weight-equivalent count, not a number of independent observations. Dollar profit
reflects deployed capital. Sign disagreements are flagged rather than selecting
whichever estimand looks better. Paper results remain hypothetical.

Equal-stake scaling assumes profit changes proportionally with stake. Fees, minimum
stakes, price impact, limits, fills and payout rules can violate that assumption.
A fixed unit is an arithmetic comparison, not an executable stake recommendation,
a bankroll simulation or an exact Kelly replay.

## Paired cohorts and unavailable results

Canonical eligibility and diagnostic eligibility are separate flags. Diagnostics
retain canonical exclusions, then reject nonpositive/nonfinite stakes, nonfinite
profits and nonfinite unit returns on the same paired cohort for both estimands.
If this adds exclusions, show the canonical baseline and diagnostic denominator
separately. Matched EV/return charts also publish individual coverage and common
cohort counts/fingerprints; different denominators cannot share a calibration claim.

Empty/undefined economics stay null with a reason. A genuine zero profit, push or
zero EV is a value, not missing evidence. A count of zero eligible wagers is a
known count; it does not make that population's ROI zero. In prospective paper
reports, pending returns stay unknown and voids are excluded from settled support.

For a simple diagnostic example, stakes 9 and 1 with profits 9 and −1 produce
80% weighted ROI and 0% equal-stake mean return. Dollar profit is 8; fixed-unit
notional profit at unit 1 is 0. Neither measure should replace the other because
it looks more favorable. Consistently scaling all stakes/profits changes dollars,
not either return measure. These examples are arithmetic, not recommended sizing.

## Sizing and uncertainty

Regime summaries use verified applied-policy linkage; interval matches alone do
not establish application. Unknown regimes remain separate. Bankroll-relative
stake requires independently supported bankroll and temporal application evidence.
See [sizing-history schema and linkage](../SIZING_HISTORY.md).

Whole-event intervals for diagnostics describe the common event-covered subset,
with its own counts, money and covered point estimates, not automatically the full
point-estimate cohort. See [statistical evidence](STATISTICAL_EVIDENCE.md).
Unknown fees, placement/settlement order, concurrent exposure, cash flows and fills
block exact Kelly replay, bankroll growth or drawdown claims. Fixed-unit arithmetic
does not supply those missing inputs or remove selection bias.

Implementation: [payout eligibility](../../src/oddsjam_ev/analysis/payouts.py),
[sizing diagnostics](../../src/oddsjam_ev/analysis/sizing_diagnostics.py),
[sizing linkage](../../src/oddsjam_ev/analysis/sizing_linkage.py).
Verification: [canonical economics](../../tests/test_bet_facts.py),
[sizing estimands/exclusions](../../tests/test_sizing_diagnostics.py),
[sizing provenance](../../tests/test_sizing_history.py),
[visual matched cohorts](../../tests/test_visual_diagnostics.py).

[Contract reference](README.md) · [Documentation home](../README.md)
