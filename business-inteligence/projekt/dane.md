# Poland Market Snapshot (Finwire MCP)

Updated: 2026-11-08

## 1) Monetary policy (NBP policy rates)

As of 2026-03-05:

| Rate | Value | Change vs previous |
|---|---:|---:|
| Reference | 3.75% | -0.25 pp |
| Lombard | 4.25% | -0.25 pp |
| Deposit | 3.25% | -0.25 pp |
| Rediscount | 3.80% | -0.25 pp |
| Discount | 3.85% | -0.25 pp |

## 2) Inflation (CPI, GUS)

Latest reading (2026-10): **3.5% YoY**, **-0.4% MoM**.

Recent 12-month range (from pulled data):
- YoY high: **4.1%** (2026-09)
- YoY low: **2.1%** (2026-01 and 2026-02)

## 3) FX vs PLN (NBP, latest in snapshot)

Date: 2026-11-06

| Pair | Mid |
|---|---:|
| EUR/PLN | 4.3925 |
| USD/PLN | 3.9321 |
| GBP/PLN | 5.2630 |
| CHF/PLN | 4.8037 |

## 4) Finwire nowcast (model-based, informational)

As of 2026-11-08:

- CPI nowcast for 2026-11: **3.5% YoY** (80% interval: **2.8%–4.2%**)
- Core inflation nowcast for 2026-10: **2.8% YoY** (80% interval: **2.6%–3.0%**)

## 5) Public debt and budget (Ministry of Finance / Eurostat via finwire)

Key values:

- State Treasury debt (2026-08): **2,249,051.0 mln PLN**
- YoY debt change: **+396,088.2 mln PLN** (**+21.38%**)
- Debt per resident (finwire_public_debt_per_capita): **60,244 PLN**
- General government debt (Eurostat definition, 2026-Q2): **64.5% GDP**
- State budget balance (2026-09, cumulative from January): **-172,211.3 mln PLN**

## 6) Dataset granularity and pull capabilities

### A) `interest-rates` (tool: `get_dataset`, id=`interest-rates`)

- **Time granularity / resolution:** event-based policy decisions (effective date of NBP rates), with historical window controlled in days.
- **Geographic granularity:** Poland (national, NBP policy rates).
- **What it contains:**
  - 5 core NBP rates: reference, lombard, deposit, rediscount, discount.
  - For each entry: id, indicator code, value, change, change %, previous value/date, current date.
- **What you can pull (filters/params):**
  - `indicator` (single rate id, e.g. `stopa-referencyjna`) for focused pull.
  - `days` (history window, 7–3650) for time-depth control.

### B) `cpi` (tool: `get_dataset`, id=`cpi`)

- **Time granularity / resolution:** monthly and yearly CPI series.
- **Geographic granularity:** Poland (national CPI).
- **What it contains:**
  - CPI YoY and MoM readings by period.
  - Short series metadata (source, count, period labels).
- **What you can pull (filters/params):**
  - `type`: `monthly` (default) or `yearly`.
  - `limit`: number of periods (months for monthly mode).

### C) `fx` (tool: `get_dataset`, id=`fx`) and FX block inside `snapshot`

- **Time granularity / resolution:** daily FX observations.
- **Geographic granularity:** Poland quote currency (PLN) vs selected foreign currencies.
- **What it contains:**
  - NBP table A/C-based FX values: mid, bid, ask.
  - Per observation: date, daily change and change %, table metadata.
- **What you can pull (filters/params):**
  - `code` (ISO currency code, e.g. EUR, USD).
  - `days` (history window length).

### D) `snapshot` (tool: `get_dataset`, id=`snapshot`)

- **Time granularity / resolution:** mixed-frequency aggregate (latest available readings per component).
- **Geographic granularity:** Poland (national macro/market snapshot).
- **What it contains:**
  - Latest **interest rates** block.
  - Latest **inflation** block.
  - Latest **currency board** (multi-currency FX vs PLN).
  - Market stress/barometer sections (aggregated state indicators).
- **What you can pull (filters/params):**
  - One-call market state payload (no slicing params in this call pattern).
  - Best for dashboards / top-level status, less for deep historical extraction.

### E) `nowcast` (tool: `get_dataset`, id=`nowcast`)

- **Time granularity / resolution:** daily model refresh; target periods are monthly macro releases.
- **Geographic granularity:** Poland (national inflation / policy expectation context).
- **What it contains:**
  - CPI and core inflation point forecasts.
  - 80% confidence interval (`lo80`, `hi80`).
  - Release/freeze dates, model version, key drivers.
  - Backtest and model quality metrics (e.g., MAE, directional accuracy, coverage).
  - For policy context: probability-type structures where provided by indicator.
- **What you can pull (filters/params):**
  - Full nowcast package (current forecast + history + diagnostics) in one call.
  - Useful for scenario and forecast-quality analysis.

### F) `public-debt` (tool: `get_dataset`, id=`public-debt`)

- **Time granularity / resolution:** mixed:
  - Monthly: State Treasury debt and budget execution snapshots.
  - Quarterly: national/general government debt and GDP-ratio metrics.
  - Long history (starting from 2001 for core debt timeline).
- **Geographic granularity:** Poland national public finance.
- **What it contains:**
  - State Treasury debt level, MoM/YoY/YTD changes.
  - Debt structure (domestic/foreign, currency composition, holder structure).
  - Per-capita debt index (`finwire_public_debt_per_capita`).
  - Public debt ratios (% GDP) and Eurostat-compatible views.
  - Budget execution lines (revenue, expenditure, deficit, debt service, plan progress).
- **What you can pull (filters/params):**
  - Unified fiscal/debt snapshot payload combining debt stock, structure, and budget execution.
  - Suitable for both current-state monitoring and long-horizon trend extraction.

## 7) Practical pull design notes

- Use **`snapshot`** first for a fast top-level state.
- Use **`interest-rates` / `cpi` / `fx`** for higher-resolution historical slices.
- Use **`nowcast`** for forward-looking signals and uncertainty bands.
- Use **`public-debt`** for fiscal structure and debt sustainability views.
- For BI pipelines, keep the `asOf`/`updatedAt` timestamps to align mixed-frequency datasets.

## Sources

- finwire.pl dataset `interest-rates` (NBP): https://public-api.finwire.pl/datasets/interest-rates
- finwire.pl dataset `cpi` (GUS): https://public-api.finwire.pl/datasets/cpi
- finwire.pl dataset `snapshot` (aggregated market snapshot): https://public-api.finwire.pl/datasets/snapshot
- finwire.pl dataset `nowcast` (Finwire Nowcast): https://public-api.finwire.pl/datasets/nowcast
- finwire.pl dataset `public-debt` (MF/GUS/Eurostat): https://public-api.finwire.pl/datasets/public-debt

## 8) Full pull to PostgreSQL (all datasets)

Script: `pull_finwire_to_postgres.py`

What it does:
- Uses direct Finwire REST API (`https://public-api.finwire.pl/v1/...`)
- Pulls configured live endpoints (no MCP wrapper)
- Stores:
  - dataset catalog metadata (`finwire_dataset_catalog`) with endpoint -> table mapping
  - one table per dataset (name pattern: `finwire_ds_<dataset_id_sanitized>`)
    - each row is an actual record from API `data` (e.g. `entries[]`, `points[]`, or nested list blocks)
    - no summary text is stored as the primary payload
    - raw record in `record` JSONB + parsed technical columns (`record_date`, `record_period`, `record_year`, etc.)
    - each record key is flattened into regular TEXT columns (auto-created as `f_<key_path>`)
  - pull errors (`finwire_pull_errors`)

Setup:

```bash
pip install -r requirements.txt
```

Connection:
- Edit `POSTGRES_CONFIG` in `pull_finwire_to_postgres.py`:
  - `host`
  - `port`
  - `dbname`
  - `user`
  - `password`
  - optional `sslmode`

Run:

```bash
python pull_finwire_to_postgres.py
```

Optional flags:
- `--limit N` pull first N configured datasets (for testing)
- `--delay-seconds 0.5` throttle calls (helps avoid Cloudflare blocks)
- `--timeout-seconds 120` per-request timeout
- `--max-retries 5` retry transient HTTP/network failures
- `--retry-base-delay-seconds 1.5` exponential backoff base delay
- `--api-base-url ...` override API host