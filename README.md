# Lending Club Loan Default — Profit-Based Funding Decisions

Predict whether a Lending Club loan will default, using **only information an
investor could see at listing time**, and fund a loan only when its
**expected profit is positive**.

- [`PLAN.md`](PLAN.md) — full project plan, phase by phase
- [`DECISIONS.md`](DECISIONS.md) — every decision, with the reasoning

> **Status:** Phase 1 complete (data pipeline + EDA). Phase 2 (feature pipeline) is next.
> Full setup and run instructions land in Phase 8.

**Headline numbers so far:** 621,022 matured 36-month loans, 13.95% default rate,
average realized profit +$1,029 per loan (+8.25% per dollar). Charged-off loans
still repay 63.5% of principal, so the loss given default is 0.365.

## Quick start

```bash
uv sync --extra dev          # or: pip install -r requirements.txt
uv run pytest                            # 18 tests
uv run python -m lending_club.data.ingest   # raw CSV -> parquet (~2 min)
uv run python -m lending_club.data.prepare  # clean + label + filter
jupyter lab notebooks/01_eda.ipynb          # Phase 1 EDA
```

## Data

`accepted_2007_to_2018Q4.csv` (about 2.26M loans) from the Kaggle mirror
`wordsforthewise/lending-club`. It is tracked by DVC, not git — see
[D-028](DECISIONS.md#d-028-raw-data-via-manual-download-stored-as-uncompressed-csv)
for the file checksum and why.
