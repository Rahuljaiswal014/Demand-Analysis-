<<<<<<< HEAD
# HomeMonde Demand Analysis

Analyses historical and continuously updated Amazon.in sales to answer: **which** products sell, **where**,
**when** demand rises or falls, how **season and weather** move it, and which **lifecycle stage** each product is in.

## Run it

```bash
pip install -r requirements.txt
python -m demand.ingest          # reads new report files, rebuilds the store, fetches missing weather
streamlit run app.py             # dashboard at http://localhost:8501
```

**Claude features, two routes** (`AI_BACKEND` in `.env`: `auto` (default), `claude_code`, or `api`):

- `claude_code` - free with a Claude Pro/Max subscription. The dashboard calls the Claude Code CLI installed on this computer
  (`claude -p`), which runs on your subscription login. No API bill. Works only on a machine where `claude` is installed and
  logged in, shares your subscription usage limits, and takes 20-90 s per answer. The API key is deliberately kept out of that
  process so the subscription is what gets used. For the analyst chat, Claude Code may run exactly one command,
  `python -m demand.sql "<SELECT ...>"`, against a read-only database file (`data/store/analyst.duckdb`).
- `api` - the Anthropic API with `ANTHROPIC_API_KEY` in `.env` (copy `.env.example`). Pay-as-you-go with prepaid credit from
  console.anthropic.com; a subscription does not fund it. Right for a server or a team. Roughly USD 0.10-0.50 per click on
  `claude-opus-5`; `CLAUDE_MODEL` in `demand/config.py` can be set to `claude-sonnet-5` for ~40% of that.
- `auto` uses `claude_code` when the CLI is found, otherwise `api`. The **Data** page shows which route is active.

**Fabric consumption** (`demand/fabric.py`, page "Fabric consumption"). Reads HomeMonde's manufacturing sheet
`Fabric Consumption*.xlsx` (project root or `data/raw`; one row per SKU with fabric code, type, family, product type, colour,
size, metres per piece and pieces per SKU) into `data/store/fabric.parquet`, refreshed by `python -m demand.ingest` whenever the
file changes (a workbook open in Excel is fine). Joined to products by ASIN first (79% of units), then by SKU. Products the sheet
does not list get metres from the sheet's own size rule where it is unambiguous - curtain metres per piece depend on length only
(5 ft 1.74 m ... 10 ft 3.28 m), cushion / pillow / bolster metres on cover size - and are flagged **inferred, fabric code unknown**;
outsourced (bought-in) products are excluded, see below. 91.6% of units carry a metres
figure. Every table shows the source, so sheet figures are never silently mixed with inferred ones. The page gives metres by
fabric code / type / family / product type / month / place, a 12-week requirement per fabric code from the product-level forecast
(baseline and festive scenario), a 12-month long-range view from the ML prediction, a lookup per code, and the list of products
missing from the sheet (download it, add them to the sheet, re-ingest). The analyst can query `product_fabric`, `fabric_sheet` and
`fabric_requirement`; the AI suggestions page has a "Fabric purchase plan".

**Sourcing and outsourced stock** (`demand/sourcing.py`, page "Outsourced stock"). Reads `Outsource*.xlsx` (project root or
`data/raw`; columns Item sku, Ready stock, Virtual quantities) into `data/store/outsource.parquet`; every changed sheet is also
appended to `data/store/outsource_history.parquet`, dated by the file's modified time, so stock movement builds up over time.
- **Mapping**: sheet SKU -> Amazon listing by exact SKU, then ignoring punctuation and case (`HMCF-60x90-BLK_MRN` = `HMCF-60X90-BLK&MRN`).
  Of 420 SKUs: 281 exact, 27 by punctuation, 112 never sold in the data.
- **Sourcing of every product**: In-house (on the fabric sheet), Outsourced (on the outsource sheet), otherwise inherited from
  its SKU range when all classified products of that range agree (flagged "by range"), else Unclassified. It is a sidebar filter,
  so every page can be viewed for in-house or outsourced products only. Outsourced = 7.4% of units (mattress protectors, rugs,
  throws, comforters, sofa covers almost entirely; cushion covers 44%).
- **Fabric correction**: outsourced products consume no in-house fabric and are no longer given metres by the size rule.
- **Stock against demand**: per SKU the product-level forecast from the snapshot date on gives weeks of cover, run-out date, a
  status (Out of stock / Critical / Reorder now / Healthy / Overstock / Idle stock / Inactive) and a reorder quantity. Where a
  product stopped selling before the data ends while still selling well (stock-out suspected), its last selling rate replaces
  the collapsed forecast (column `demand_basis`).
- **Assumptions to confirm** (`demand/config.py`): `OUTSOURCE_AVAILABLE` = `max` (virtual is read as the quantity available to
  sell and already includes ready stock; use `sum` if it is extra), `OUTSOURCE_LEAD_WEEKS` = 4, `OUTSOURCE_REVIEW_WEEKS` = 8.
  The sheet does not include stock already at Amazon fulfilment centres.
The analyst can query `product_sourcing` and `outsource_stock`; the AI suggestions page has an "Outsourced stock plan".

**AI suggestions page.** Four plans, each written by Claude from a curated evidence set (`demand/plays.py`) in a fixed format
(action table with product, place, date, evidence and confidence, then reasoning, what would change the plan, and what is not
recommended): this week's priorities, the festive stock plan, a plan for any region / state / district, and a verdict on any SKU.
Plans are saved under `data/store/ai_notes/` with their timestamp, can be regenerated and downloaded, and show the evidence
Claude was given. "Order by" dates use `PRODUCTION_WEEKS` and `INBOUND_WEEKS` in `demand/config.py`: **set these to your real
lead times**, the defaults (8 and 2) are placeholders.

**AI analyst page.** Starter questions, live display of each query as it runs, answer-first replies that name products by
HomeMonde SKU, the rows behind the last query, three suggested follow-up questions as buttons, optional scoping to the sidebar
filters, chat download. On the subscription route an answer takes roughly 20 seconds to 2 minutes, a plan 1 to 5 minutes.

Independent of both, every page carries a **Key points** card and the forecast page an **Action list** from `demand/insights.py`:
rule-based, free, instant, no AI. Everything else in the tool works with no key and no subscription.

## Light / dark mode

The ⋮ menu (top right) has **System / Light / Dark** at the top. Both themes are defined in `.streamlit/config.toml`
(`[theme.light]`, `[theme.dark]`), the page styling follows the choice instantly, and the charts pick up the matching palette on
the next rerun (any click, or press **R**). The choice is remembered per browser.

## Monthly update

Download **All Orders** from Seller Central for the new month (the native `.txt` is fine, so is `.xlsx`/`.csv`), then
either upload it on the dashboard's **Data** page or drop it into `data/raw/` and run `python -m demand.ingest`.
Files already ingested are skipped. Re-downloading an old month is safe and useful: rows are keyed on
order id + order item id + SKU and the most recently updated version wins, so late cancellations are picked up.

## How the data is cleaned (`demand/ingest.py`)

| Issue in the raw report | Handling |
|---|---|
| Timestamps are UTC, the business runs on IST | converted to Asia/Kolkata before any day/week/month cut |
| Cancelled lines (qty 0, no price) | kept in the store, flagged, excluded from demand |
| `Non-Amazon` rows (MCF: bulk transfers to Jaipur, website orders) | flagged `is_amazon = False`, excluded from demand |
| One listing sold under several SKU aliases (`…%-OOS`, `-NEW`, `-S2`) | **ASIN is the product key**; SKUs listed on the product |
| No category column | title rules + HomeMonde's own SKU stems (`PREFIX_CATEGORY`), audited against each other; curtain length / type / pack size parsed from title with the SKU code as second source (Amazon cuts titles at ~150 characters) |
| 215 spellings of state names | canonical list → aliases → fuzzy match → majority state of the PIN prefix |
| Zero-priced shipped lines (replacements) | counted as units, ignored for price |

**Sept-25 added (2026-09-21).** New sheet `Sept-25`: 35,516 rows, 42,716 units in sheet, 42,023 demand units after removing 693
non-Amazon units; other sheets unchanged row for row. History is now a full year. A workbook that is open in Excel is read through a
temporary copy (`ingest._copy_locked`).

**Audit (2026-09-21).** The workbook was re-read cell by cell with a second library and reconciled to the store: rows, units and
revenue match in every month with zero difference (the Data page shows the reconciliation). State names disagree with the PIN
code on 0.4% of rows, all genuine border cases (Chandigarh/Punjab, Puducherry/Tamil Nadu). `data/reference/listing_length_conflicts.csv`
lists 12 listings whose title and SKU code state different curtain lengths; those need fixing at source.

Not in the report, so not in the tool yet: **customer returns** (figures are gross) and **inventory**.

## The analytics (`demand/analytics.py`)

- **Demand calendar** (`predict.demand_calendar`): for All India, a region, a state or a district, the top categories or SKUs of
  every month: actual months from the orders, future months from the ML model. The model predicts state x category; a district
  gets its historical share of its state (category by category), and SKUs get their last-12-week share of their category
  (only SKUs sold in the last 28 days).
- **Districts** (`geo.add_district`): every order's district comes from its PIN code via the India Post directory
  (`data/reference/pin_district.csv`, 19,100 PINs). Buyer-typed city names (9,500 spellings) are not used. Checked on this data:
  98.8% of rows match the directory exactly and the directory's state agrees with the order's state 99.93% of the time; the 1.2%
  of newer PINs take the district of neighbouring PINs. The report holds city, state and PIN only, no street addresses.
- **Most & least ordered** (`rank_cells`, `item_peaks`): for SKU / design family / category, the top and bottom sellers in every
  cell of *month, week, date, season, temperature band or rain band* x *All India, region or state*. Every order line is tagged
  with the weather of that day in the buyer's state, so "what sells when it is above 36 C in the North" is a direct lookup.
  Rank by **units**, or by **over-index (lift)**: on raw units the national best-seller wins almost everywhere, while lift shows
  what is distinctively strong or weak in a place or climate. An item only competes in periods between its first and last sale.
  "SKU (aliases merged)" rolls alias SKUs (`...%-OOS`, `-NEW`, `%-8EYT`) into the listing's main SKU; raw SKU is available too,
  but a raw SKU can look dead when its sales simply moved to an alias.
- **Forecast** (`demand/forecast.py`): 4-16 weeks ahead at total / category / region / curtain type / SKU level. Baseline is
  damped-trend exponential smoothing with an 80% band; thin series get their recent share of the total. A separate **festive
  scenario** reuses the 2025 uplift at the same distance from Diwali (dates come from `events.csv`). Backtest from four past
  dates: about 14% weekly error for the total, 18% by category, and about 70% weekly (47% on 8-week totals) across all products,
  because most products sell a few units a week. Plan SKUs on multi-week totals and trust the top sellers' forecasts most.
- **Future years (ML)** (`demand/predict.py`): monthly prediction to Dec 2028 by region x category. History is one year (Sep-25 to
  Aug-26): every calendar month seen once, none repeated, so "same month last year" is one number with trend, season and festival
  mixed together. The model therefore learns *why* demand moves instead of *when*: a
  gradient-boosted model of weekly state x category units on temperature, rain, humidity and weeks-to-Diwali (~20k rows, 36
  states living through different weather in the same weeks). Future months are predicted from climate normals and the
  festival calendar in `events.csv`. **Growth is an input, not an output**: year-on-year growth cannot be measured in under
  12 months. Validation (after Sep-25 was added): holding out 4-week blocks across the year, the model removes 23-37% of a flat
  average's error; holding out the last 12 weeks it errs 14.7% by week against 17.0% for the flat average (11.0% vs 14.9% by
  region x month). Before September was added it had no edge on that second test, because it had never seen monsoon-season demand.
  So: a real but modest edge, theoretical beyond ~6 months, and due another upgrade once Sep-Nov 2026 actuals are in (first repeated
  months, first measurable year-on-year growth). The festive window is 5 weeks before to 4 weeks after the Diwali week: in 2025 the
  sale opened on 22 Sep, four weeks before Diwali week. Climate normals come from
  `python -m demand.weather` (10-year, resumable; the free API has an hourly quota). Until they are downloaded the model uses
  last year's actual weather as the normal year.
- **AI suggestions**: Claude reads the forecast, lifecycle, regional over-index and climate sensitivity and writes a 12-week
  action list (stock up, regional placement, climate timing, slow movers, risks).
- **Trends** use complete Monday–Sunday weeks only. Direction = log-linear slope over the last 8 weeks with a t-test.
- **Demand events** = days ≥40% off a weekday-adjusted 4-week rolling median, merged into spikes/slumps.
- **Season**: units/day by month and by Indian season as an index (100 = the group's own average). With under
  12 months of history this is one observation of the cycle, not a confirmed pattern. It firms up after Oct 2026.
- **Weather sensitivity**: two-way fixed-effects regression of log weekly units on temperature, rain and humidity
  across the larger states. State effects remove "big states buy more"; week effects remove national events
  (festive sales, Prime Day). The estimate is identified from states being hotter/wetter than *each other* in the
  same week, so it does not need multiple years. Weather comes from the Open-Meteo archive, one point per state
  (`demand/geo.py`). Open-Meteo's free tier is non-commercial; buy their plan or swap the source for production.
- **Lifecycle**: Launch / Growth / Mature / Decline / Dormant / Sporadic. Growth and decline are judged on the
  product's *share of its category*, so a seasonal dip in the whole category does not mark everything as declining.
  Thresholds are in `demand/config.py`. Products already selling when the data begins (1 Sep 2025) have an unknown true launch date.
- **Stock-out suspicion**: steady sellers with a ≥7-day zero-sales gap that later resumed. This is a stand-in until
  FBA inventory history is loaded. Without it a "decline" can be a supply problem, not demand.

## The AI layer (`demand/ai.py`)

Claude (`claude-opus-5`) gets a read-only, in-memory DuckDB copy of the store (file and network access disabled)
and one tool, `run_sql`. It answers questions by querying, and the chat shows every query it ran. The page-level
"Explain" buttons send the tables already on screen for a narrative read-out.

## Layout

```
app.py                  Streamlit dashboard
demand/ingest.py        report files -> data/store/orders.parquet + products.parquet
demand/geo.py           state normalisation, regions, weather coordinates
demand/weather.py       Open-Meteo daily weather per state -> weather.parquet
demand/analytics.py     products, rankings, geography, trends, climate, lifecycle
demand/forecast.py      weekly forecast, festive scenario, backtest
demand/predict.py       ML long-range prediction by region / category / month, validation, learned drivers, written outlook
demand/ai.py            Claude analyst (API or Claude Code backend)
demand/sql.py           the one read-only query command the Claude Code backend may run
demand/insights.py      rule-based key points and action list (no AI)
demand/plays.py         evidence tables and goals for the five AI suggestion plans
demand/fabric.py        fabric sheet loader, product join, metres consumed, fabric requirement
demand/sourcing.py      outsource sheet loader, SKU mapping, in-house / outsourced classification, stock cover and reorder
demand/config.py        paths and thresholds
data/raw/               drop new reports here
data/reference/events.csv   festival / sale calendar shown on charts (edit freely)
```

## Next steps

1. **FBA inventory history** (Inventory Ledger, daily, by ASIN) → replace the stock-out heuristic with real in-stock
   days and correct demand for lost sales.
2. Returns report → net demand and return rates.
3. Amazon sale-event dates in `events.csv`, so spikes are labelled rather than just detected.
4. After Oct-Nov 2026 is ingested, the festive uplift has two observations and yearly seasonality can enter the forecast.
=======
# Demand-Analysis-
An AI-powered demand intelligence platform that analyzes sales trends, regional performance, sell-through rates, inventory levels, seasonality, and weather patterns to forecast product demand, identify lifecycle stages, and deliver actionable insights for smarter inventory planning and data-driven business decisions.
>>>>>>> c67141bfce124db3a5674c912390651a9a15e86c
