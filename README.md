# Lending Club Loan Default — Profit-Based Funding Decisions

Predict whether a Lending Club loan will default, using **only information an
investor could see at listing time**, and fund a loan only when its
**expected profit is positive**.

- [`PLAN.md`](PLAN.md) — full project plan, phase by phase
- [`DECISIONS.md`](DECISIONS.md) — every decision, with the reasoning

> **Status:** Phase 0 complete (project setup). Phase 1 (data loading + EDA) is next.
> Full setup and run instructions land in Phase 8.

## Quick start

```bash
uv sync --extra dev          # or: pip install -r requirements.txt
uv run pytest                # smoke tests
```

## Data

`accepted_2007_to_2018Q4.csv` (about 2.26M loans) from the Kaggle mirror
`wordsforthewise/lending-club`. It is tracked by DVC, not git — see
[D-028](DECISIONS.md#d-028-raw-data-via-manual-download-stored-as-uncompressed-csv)
for the file checksum and why.
