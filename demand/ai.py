"""Claude-powered demand analyst.

Two entry points:
  Analyst.ask()  - conversational Q&A; Claude writes SQL against a read-only DuckDB copy of the store
  brief()        - one-shot narrative over tables the dashboard already computed

Two backends, see config.AI_BACKEND: the Claude Code CLI on the user's subscription login (no API bill), or the
Anthropic API (ANTHROPIC_API_KEY). Nothing is hard-coded, and the key is never passed to the CLI.
"""
import json
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime

import anthropic
import duckdb
import pandas as pd
from anthropic import beta_tool

from . import analytics as A
from . import config as cfg
from . import forecast as F
from . import predict as ML
from . import fabric as FB
from . import sourcing as SRC

MAX_ROWS = 200
BETAS = ["server-side-fallback-2026-07-01"]

SYSTEM = """You are the demand analyst for HomeMonde Lifestyle, an Indian home-textiles brand \
(curtains are ~80% of units; also bedsheets, cushion covers, comforters, mattress protectors, table linen, rugs) \
selling mainly on Amazon.in through FBA. You answer questions from the business team about what is selling, \
where, when demand is rising or falling, how season and weather affect it, and where products sit in their lifecycle.

You have a `run_sql` tool over a DuckDB database. Query it rather than guessing: every number you state should \
come from a query result in this conversation. Tables:

sales — one row per order line of real marketplace demand (cancelled and non-Amazon MCF rows already removed)
  date DATE (IST), ts TIMESTAMP (IST), order_id, asin, sku, category, family, region, state,
  district (from the PIN code via the India Post directory, labelled like 'JAIPUR, RJ' or 'PUNE, MH'; 'UNKNOWN' if no PIN;
  use this for any sub-state question), city (buyer-typed, ~9,500 spellings: avoid for grouping), pin,
  quantity INT (units), revenue DOUBLE (INR item price), unit_price DOUBLE (NULL for free replacements),
  has_promo BOOL, is_business BOOL, fulfillment ('Amazon' = FBA, 'Merchant' = Easy Ship),
  main_sku (the listing's principal SKU: use this, not sku, to rank "SKUs", because one listing sells under several
  alias SKUs such as ...%-OOS / -NEW / %-8EYT and raw sku splits its history),
  calendar context: month 'YYYY-MM', week DATE (Monday), season ('Summer' Mar-May, 'Monsoon' Jun-Sep,
  'Post-monsoon' Oct-Nov, 'Winter' Dec-Feb),
  climate context of that day in the buyer's state: temp_max, rain_mm, temp_band ('<24°C','24-28°C','28-32°C',
  '32-36°C','>36°C'), rain_band ('Dry day','Light rain (1-10 mm)','Rainy day (>10 mm)')

all_orders — every raw line including cancelled (is_cancelled) and non-Amazon MCF (is_amazon = false); same columns
  as sales plus order_status, sales_channel, is_demand. Use only for cancellation or MCF questions.

products — one row per ASIN (the product key; SKUs are aliases of one listing)
  asin, title, main_sku, skus, n_skus, category, family (design family from the SKU stem, e.g. HM152-063),
  curtain_type ('Blackout'/'Sheer'/'Other curtain', curtains only), curtain_length_ft, pack_size, first_sale, last_sale

lifecycle — one row per ASIN, recomputed on load
  asin, main_sku, title, category, family, stage ('Launch','Growth','Mature','Decline','Dormant','Sporadic'),
  first_sale, last_sale, age_days, launch_date_known BOOL (false = already on sale when the data begins, true age unknown),
  days_since_sale, units_total, units_26w, units_last_4w, peak_4w_avg, peak_week, recent_vs_peak_share,
  growth_pct_wk (last 8 weeks, absolute), growth_vs_category_pct_wk (share of its category, the basis for the stage),
  trend_t_stat, longest_gap_days, gap_ended, stockout_suspected BOOL

weather — daily weather per state at its main demand city: date, state, temp_max, temp_min, temp_mean, rain_mm, humidity

climate_sensitivity — per category, two-way fixed-effects estimates: per_1C_hotter_pct, per_10mm_rain_pct,
  per_10pt_humidity_pct with t-stats, and weather_signal

forecast — next 12 weeks: level ('total','category','region','product'), grp (category / region / asin), title
  (products only), week DATE, baseline, low, high (80% band), festive_multiplier, festive_scenario, method.
  baseline is damped-trend smoothing of recent weeks. festive_scenario applies the uplift seen at the same distance
  from Diwali in 2025 (one observation; Diwali 2026 is 8 Nov). Backtested week-level error is roughly 14% for the
  total, 18% by category, and far higher for individual products: only the top sellers forecast well week by week,
  so quote product forecasts as multi-week totals and say how uncertain they are.

prediction — long-range ML projection to Dec 2028: region, category, month 'YYYY-MM', units, revenue. A gradient-boosted
  model of weekly state x category demand on temperature, rain, humidity and weeks-to-Diwali, fed climate normals and
  the festival calendar, with zero underlying growth assumed (growth is unobservable until a month repeats). Cross-validated
  it removes 25-35% of the error of a flat average, and on the unseen last 12 weeks it errs ~15% by week against 17% for
  the flat average: a real but modest edge. Use `forecast`
  for the next 12 weeks and `prediction` for months and years ahead, and call anything beyond ~6 months theoretical.

product_fabric — one row per ASIN with its fabric: fabric_code (e.g. 'FAB-0030'; '(unknown: new ranges, by size rule)' when
  only the metres are known; '(none)' when the product has no fabric row), fabric_type (Polyester/Sheer/Satin/Duck/Velvet),
  fabric_family (Solid/Dimout/Printed/Blackout/Stripe/Digital Blackout), fabric_product, product_type, colour, size,
  metres_per_piece, pieces, metres_per_unit (metres of fabric one sold unit consumes), fabric_source ('sheet: ASIN' /
  'sheet: SKU' / 'inferred: size rule' / 'outsourced: no fabric' (bought in finished, consumes no in-house fabric) /
  'not in sheet'). Fabric consumed = SUM(sales.quantity * metres_per_unit), joined on asin.
  Only 'sheet:' rows carry a real fabric code; always say when a figure includes inferred metres.

fabric_sheet — HomeMonde's manufacturing sheet as loaded (one row per SKU: sku, asin, fabric_code, fabric_type, fabric_family,
  fabric_product, product_type, colour, size, metres_per_piece, pieces, metres_per_unit).

fabric_requirement — next 12 weeks in metres per fabric_code: baseline_m, festive_m (festive scenario), last_12w_actual_m,
  products. Built from the product-level weekly forecast x metres_per_unit.

product_sourcing — one row per ASIN: asin, main_sku, category, prefix (SKU range), sourcing ('In-house' = HomeMonde manufactures
  it, 'Outsourced' = bought in as finished goods from outside makers, 'Both sheets (check)', 'Unclassified'),
  sourcing_basis ('fabric sheet' / 'outsource sheet' are stated by the business; 'by range (SKU prefix ...)' is inferred because
  every classified product of that SKU range has the same sourcing). Join to sales on asin for in-house vs outsourced demand.

outsource_stock — one row per SKU of HomeMonde's outsource stock sheet (a stock snapshot, date in column snapshot): sku (as
  in the sheet), asin, main_sku, title, category, match (how the SKU was tied to a listing: 'exact SKU' / 'same SKU, different
  punctuation' / 'no sales in the data'), ready (finished stock in hand), virtual (quantity the business keeps available to
  sell; can exceed ready), available (the larger of the two, the figure cover is measured on), units_year, units_8w,
  units_4w, last_sale, weekly_rate (forecast units per week), fc_12w_baseline, fc_12w_festive (forecast for the 12 weeks
  after the snapshot), demand_basis ('forecast', or 'last selling rate (sales stopped: stock-out suspected)' when the product stopped selling before
  the data ends and its forecast had collapsed), cover_weeks (weeks until available stock runs out along the festive forecast;
  NULL = no demand),
  cover_weeks_ready_only, status ('Out of stock','Critical' <2 weeks,'Reorder now' under lead time + 2 weeks,'Healthy',
  'Overstock' >26 weeks,'Idle stock' = stock but no sales in 8 weeks,'Inactive'), stockout_date, reorder_baseline,
  reorder_festive (units to order so that lead time + review period is covered), stage (lifecycle).
  The lead time (4 weeks) and review period (8 weeks) are placeholders the business has not confirmed.
  Caveats to state: the snapshot is later than the last sales date, so cover rests on the forecast; the sheet does not show
  stock already sitting in Amazon fulfilment centres; the meaning of 'virtual' is as understood above, not confirmed.

Things to keep in mind when interpreting:
- History is exactly one year, 2025-09-01 to 2026-08-31: every calendar month seen once, none repeated. Seasonal \
patterns are therefore one observation, not a proven cycle, and year-on-year growth cannot be measured yet; say so when \
it matters. The 2025 festive sale opened on 22 Sep (daily units doubled overnight), peaked 11-14 Oct, Diwali was 20 Oct, \
and November was the trough. Ordinary September 2025 days ran ~1,100 units, the same as August 2026.
- Inventory is known only for outsourced products (outsource_stock, one snapshot). For everything else a fall in sales \
may be a stock-out rather than lost demand; check lifecycle.stockout_suspected and say when that is a live possibility.
- The report carries no customer returns, so all figures are gross demand.
- Compare like with like: use units per day or complete weeks when periods differ in length.
- "Most / least ordered": the national best-seller wins almost every region and month on raw units, so also look at
  over-index (an item's share within a region / season / temp_band divided by its overall share) to find what is
  distinctively strong or weak there. For "least ordered", restrict to items with meaningful overall volume and, for
  time periods, to items on sale in that period (between first_sale and last_sale); otherwise the answer is a long
  list of zero-sellers.
- When asked for suggestions, tie each one to a figure, name the SKU / region / weeks it applies to, and rank by
  business impact. Flag where a stock-out or the single festive observation makes the advice uncertain.

Answer in plain business language, lead with the conclusion, give the figures that support it, and keep it concise. \
Use INR with Indian digit grouping in lakh/crore where natural. If a question cannot be answered from these tables, \
say what data would be needed."""


class ClaudeCodeError(RuntimeError):
    """The Claude Code CLI could not answer (not installed, not logged in, timed out, usage limit)."""


def backend() -> str:
    if cfg.AI_BACKEND in ("api", "claude_code"):
        return cfg.AI_BACKEND
    return "claude_code" if shutil.which("claude") else "api"


def backend_label() -> str:
    return ("Claude Code on this computer's Claude subscription login (no API charge)" if backend() == "claude_code"
            else f"Anthropic API, {cfg.CLAUDE_MODEL} (pay-as-you-go)")


def ensure_database():
    """Build the on-disk analyst database if it is missing or older than any store it is built from. Returns its path."""
    path = cfg.ANALYST_DB
    stores = [f for f in (cfg.ORDERS_PARQUET, cfg.FABRIC_PARQUET, cfg.OUTSOURCE_PARQUET) if f.exists()]
    if path.exists() and path.stat().st_mtime >= max(f.stat().st_mtime for f in stores):
        return path
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    build_database(str(tmp)).close()
    os.replace(tmp, path)
    return path


def _claude_code(prompt: str, allow_sql: bool = False, timeout: int = 600, on_query=None, extra_system: str = "") -> tuple[str, list[str]]:
    """One headless Claude Code run. Returns (answer, SQL queries it ran). `on_query(sql)` fires as each query starts,
    so the dashboard can show progress during the 20-90 seconds an answer takes."""
    exe = shutil.which("claude")
    if not exe:
        raise ClaudeCodeError("Claude Code is not installed on this computer (the `claude` command was not found).")
    system = SYSTEM + extra_system
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose", "--no-session-persistence", "--strict-mcp-config",
           "--model", cfg.CLAUDE_CODE_MODEL]
    if allow_sql:
        ensure_database()
        system += ("\n\nHere `run_sql` is a shell command. Run each query as:  python -m demand.sql \"<SQL>\"  "
                   "(double quotes around the SQL, single quotes inside it, one statement per call). It is the only command "
                   "you may run; do not read or write files.")
        cmd += ["--tools", "Bash", "--allowedTools", "Bash(python -m demand.sql:*)"]
    else:
        cmd += ["--tools", ""]
    cmd += ["--system-prompt", system]
    # the subscription login must be used, so the API key is kept out of the child process
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", cwd=str(cfg.ROOT), env=env)
    killer = threading.Timer(timeout, proc.kill)
    killer.start()
    proc.stdin.write(prompt)
    proc.stdin.close()
    answer, queries, error = "", [], ""
    for line in proc.stdout:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    m = re.search(r"demand\.sql\s+(.*)$", block.get("input", {}).get("command", ""), re.S)
                    queries.append((m.group(1) if m else block["input"].get("command", "")).strip().strip("\"'"))
                    if on_query:
                        on_query(queries[-1])
        elif ev.get("type") == "result":
            answer = ev.get("result") or ""
            if ev.get("is_error"):
                error = answer or ev.get("subtype", "error")
    stderr = proc.stderr.read()
    proc.wait()
    timed_out = not killer.is_alive()
    killer.cancel()
    if timed_out and not answer:
        raise ClaudeCodeError(f"Claude Code did not answer within {timeout // 60} minutes.")
    if error or not answer:
        raise ClaudeCodeError((error or stderr.strip() or "Claude Code returned no answer.")[:600])
    return answer.strip(), queries


def build_database(path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    orders, products, weather = A.load_orders(), A.load_products(), A.load_weather()
    sales = A.demand(orders)
    lifecycle = A.lifecycle(sales, products)
    forecast = forecast_table(sales, products)
    prediction = ML.predict(sales, weather, ML.load_normals(weather)[0]) if not weather.empty else pd.DataFrame({"region": []})
    climate = A.weather_sensitivity(sales, weather).reset_index() if not weather.empty else pd.DataFrame({"category": []})
    fabric_sheet = FB.load_fabric()
    outsource_sheet, product_sourcing = SRC.build(products, orders)
    product_fabric = FB.product_fabric(products, fabric_sheet, set(product_sourcing.query("sourcing == 'Outsourced'")["asin"]))
    if len(outsource_sheet):
        outsource_stock = SRC.stock_cover(sales, outsource_sheet, products, lifecycle).astype({"status": str})
    else:
        outsource_stock = pd.DataFrame({"sku": []})
    fabric_requirement = FB.requirement(sales, product_fabric).reset_index().rename(columns={"key": "fabric_code"}) if not fabric_sheet.empty else pd.DataFrame({"fabric_code": []})

    con = duckdb.connect(path)
    drop = ["is_demand", "is_cancelled", "is_amazon", "order_status", "sales_channel", "source_file", "last_updated",
            "product_name", "item_price", "promo_discount", "order_item_id"]
    sales = A.add_context(sales, weather).merge(products[["asin", "main_sku"]], on="asin", how="left")
    for name, frame in {"sales": sales.drop(columns=drop), "all_orders": orders, "products": products, "lifecycle": lifecycle,
                        "weather": weather, "climate_sensitivity": climate, "forecast": forecast, "prediction": prediction,
                        "product_fabric": product_fabric, "fabric_sheet": fabric_sheet, "fabric_requirement": fabric_requirement,
                        "product_sourcing": product_sourcing, "outsource_stock": outsource_stock}.items():
        con.register("_frame", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _frame")
        con.unregister("_frame")
    # the model writes the SQL, so the connection cannot touch the filesystem or network
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def forecast_table(sales: pd.DataFrame, products: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Forecasts at every level in one long table. Products are kept only where they have enough history of their own."""
    parts = []
    for level, by in (("total", None), ("category", "category"), ("region", "region"), ("product", "asin")):
        fc = F.forecast(sales, by, horizon)
        if level == "product":
            fc = fc[fc["method"] == "own history"]
        parts.append(fc.assign(level=level))
    out = pd.concat(parts, ignore_index=True).rename(columns={"group": "grp"})
    return out.merge(products[["asin", "title"]], left_on="grp", right_on="asin", how="left").drop(columns="asin")


def run_query(con: duckdb.DuckDBPyConnection, query: str) -> str:
    if not re.match(r"^\s*(select|with|describe|show|summarize)\b", query, re.IGNORECASE):
        return "Error: only read-only SELECT / WITH / DESCRIBE / SUMMARIZE statements are allowed."
    try:
        df = con.execute(query).fetch_df()
    except duckdb.Error as e:
        return f"SQL error: {e}"
    note = f"\n(showing first {MAX_ROWS} of {len(df)} rows; aggregate or add LIMIT)" if len(df) > MAX_ROWS else ""
    return _csv(df.head(MAX_ROWS), index=False) + note


def _csv(df: pd.DataFrame, index: bool = True) -> str:
    """Compact CSV for the model: whole-number floats as integers, everything else to 2 decimals."""
    df = df.copy()
    for c in df.select_dtypes("float").columns:
        col = df[c]
        df[c] = col.astype("Int64") if col.dropna().mod(1).eq(0).all() else col.round(2)
    return df.to_csv(index=index)


CHAT_STYLE = """

How to answer in this chat:
- Lead with the answer in one or two sentences, then the figures that support it (a small markdown table when there are more
  than three numbers), then what to do about it if the question invites advice. No preamble, no restating the question.
- When you advise, be specific: name the SKU or category, the place, the weeks, and the quantity or direction.
- Identify products the way the business does: by `main_sku` (HomeMonde's own code, present in sales, products and lifecycle)
  followed by a few words of description, e.g. "HM152-063-BGE-7 FEET (beige blackout 7 ft, set of 2)". Give the ASIN only if asked.
- If the user's view is given in [brackets] before the question, scope the answer to it unless the question says otherwise.
- Finish with one final line in exactly this form, suggesting natural next questions the user could ask (short, specific,
  answerable from these tables):  FOLLOW-UPS: first question | second question | third question"""

STARTERS = {
    "What should I do this week?": "Looking at the last 4-8 weeks: what are the three most important things to act on this week, "
                                   "and what is the evidence for each?",
    "Festive season plan": "Diwali 2026 is on 8 Nov. Using what happened around Diwali 2025, which categories and SKUs should we "
                           "build stock for, by how much over a normal week, and by when must it reach Amazon?",
    "Which products are at risk?": "Which important products look at risk: declining against their category, or showing a possible "
                                   "stock-out? List the top ten with the figure that flags each one.",
    "Where is demand growing?": "Which states and districts grew most over the last 8 weeks versus the 8 weeks before, among places "
                                "with meaningful volume, and what are they buying?",
    "What sells in hot weather?": "Which categories and SKUs sell disproportionately on days above 36 C, and in which states? "
                                  "What does that imply for next summer?",
    "New launches: how are they doing?": "How are products launched in the last 120 days performing? Which deserve more stock or ads, "
                                         "and which are not taking off?",
}


def split_followups(answer: str) -> tuple[str, list[str]]:
    """Separate the trailing 'FOLLOW-UPS: a | b | c' line from the answer text."""
    m = re.search(r"\n?\s*\**FOLLOW-UPS:?\**:?\s*(.+?)\s*$", answer, re.S | re.I)
    if not m:
        return answer.strip(), []
    qs = [q.strip(" *-\n") for q in m.group(1).split("|")]
    return answer[:m.start()].strip(), [q for q in qs if 8 <= len(q) <= 160][:3]


def query_frame(sql: str, limit: int = 500) -> pd.DataFrame | None:
    """The table behind an answer: re-run one of Claude's queries read-only so the user can see and download the rows."""
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        return None
    con = duckdb.connect(str(ensure_database()), read_only=True)
    try:
        con.execute("SET enable_external_access = false")
        return con.execute(sql).fetch_df().head(limit)
    except duckdb.Error:
        return None
    finally:
        con.close()


# ---------------------------------------------------------------- saved suggestions
def notes_dir():
    d = cfg.STORE_DIR / "ai_notes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_note(key: str, text: str) -> None:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:80]
    (notes_dir() / f"{safe}.md").write_text(f"<!-- {datetime.now():%Y-%m-%d %H:%M} -->\n{text}", encoding="utf-8")


def load_note(key: str) -> tuple[str, str] | None:
    """(text, 'generated 21 Sep 2026 16:40') for a saved suggestion, or None."""
    p = notes_dir() / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', key)[:80]}.md"
    if not p.exists():
        return None
    raw = p.read_text(encoding="utf-8")
    m = re.match(r"<!-- (.+?) -->\n", raw)
    stamp = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M").strftime("%d %b %Y %H:%M") if m else ""
    return raw[m.end():] if m else raw, stamp


PLAN_FORMAT = """Write it as a plan someone can act on tomorrow morning.

Start with a two-sentence summary. Then a markdown table with these columns, most important first, 6 to 10 rows:
| # | Action | Product / category | Where | When | Evidence (figure) | Confidence |
Confidence is High / Medium / Low and must reflect the data: one year of history, no inventory data, festive pattern seen once.
After the table, add short sections, each introduced by a bold line (not a markdown heading): "Why these, in this order" (the reasoning, 4-6 bullets), "What would change this plan"
(what to check or what new data would flip a decision), and "Not recommended" (tempting moves the data does not support).
Every figure must come from the tables given. Use SKU codes exactly as written. Indian number formatting (lakh, crore) for money."""


def plan(tables: dict[str, pd.DataFrame], goal: str, key: str) -> str:
    """A structured action plan from Claude over prepared tables; saved so it survives page reloads."""
    text = brief(tables, goal + "\n\n" + PLAN_FORMAT)
    save_note(key, text)
    return text


class Analyst:
    """Holds the database and the running conversation."""

    def __init__(self):
        self.backend = backend()
        self.messages: list = []
        self.turns: list[tuple[str, str]] = []           # (question, answer) history for the claude_code backend
        if self.backend == "claude_code":
            ensure_database()
            return
        self.client = anthropic.Anthropic()
        self.con = build_database()
        con = self.con

        @beta_tool
        def run_sql(query: str) -> str:
            """Run one read-only DuckDB SQL query against the demand database and get the result as CSV.

            Call this whenever a figure is needed. Aggregate in SQL; results are capped at 200 rows.

            Args:
                query: A single SELECT or WITH statement in DuckDB SQL.
            """
            return run_query(con, query)

        self.tools = [run_sql]

    def ask(self, question: str, on_query=None) -> str:
        """Answer one question, keeping conversation history. `on_query(sql)` is called for each query Claude runs."""
        if self.backend == "claude_code":
            past = "".join(f"Earlier question: {q}\nYour answer: {a}\n\n" for q, a in self.turns[-4:])
            answer, _ = _claude_code(past + "Question: " + question, allow_sql=True, on_query=on_query, extra_system=CHAT_STYLE)
            self.turns.append((question, split_followups(answer)[0]))
            return answer
        self.messages.append({"role": "user", "content": question + "\n\n(" + CHAT_STYLE.strip() + ")"})
        runner = self.client.beta.messages.tool_runner(
            model=cfg.CLAUDE_MODEL, max_tokens=16000, system=SYSTEM, tools=self.tools, messages=list(self.messages),
            thinking={"type": "adaptive"}, cache_control={"type": "ephemeral"},
            betas=BETAS, fallbacks="default", max_iterations=15)
        last = None
        for message in runner:
            last = message
            self.messages.append({"role": "assistant", "content": message.content})
            if on_query:
                for block in message.content:
                    if block.type == "tool_use":
                        on_query(block.input.get("query", ""))
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                self.messages.append(tool_response)
        return _final_text(last)


def brief(tables: dict[str, pd.DataFrame], focus: str) -> str:
    """Narrative insight over tables the dashboard already computed (no tool use)."""
    body = "\n\n".join(f"### {name}\n{_csv(df)}" for name, df in tables.items())
    if backend() == "claude_code":
        ask = "" if "markdown table" in focus else " Give the 4-6 findings that matter most to the business and, for each, what to do about it."
        return _claude_code(f"{focus}\n\nWork only from the tables below (no queries needed).{ask}\n\n{body}", timeout=480)[0]
    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=cfg.CLAUDE_MODEL, max_tokens=16000, system=SYSTEM, thinking={"type": "adaptive"},
        betas=BETAS, fallbacks="default",
        messages=[{"role": "user", "content":
                   f"{focus}\n\nWork only from the tables below (no queries needed). Give the 4-6 findings that matter "
                   f"most to the business and, for each, what to do about it.\n\n{body}"}],
    ) as stream:
        return _final_text(stream.get_final_message())


SUGGEST_FOCUS = """Act as the demand planner. From the tables, write an action list for the next 12 weeks:

1. **Stock up / protect availability**: SKUs and categories where demand is rising or a seasonal or festive lift is coming.
2. **Regional placement**: what to position in which region, from the over-index and regional forecast.
3. **Climate timing**: what the weather sensitivities imply for the coming season, region by region.
4. **Slow movers**: large products that are declining or selling nothing in places; say whether to investigate stock, reprice, or wind down.
5. **Risks to the plan**: where a suspected stock-out, the single festive observation, or forecast error makes a call uncertain.

Every line names the SKU / region / weeks it applies to and the figure behind it. Most important first."""


def suggestions(tables: dict[str, pd.DataFrame]) -> str:
    return brief(tables, SUGGEST_FOCUS)


def _final_text(message) -> str:
    if message is None:
        return "No response."
    if message.stop_reason == "refusal":
        return "The model declined to answer this request."
    text = "\n".join(b.text for b in message.content if b.type == "text").strip()
    if message.stop_reason == "max_tokens":
        text += "\n\n_(answer cut off at the length limit)_"
    return text or "No answer was produced; try rephrasing the question."


def friendly_error(e: Exception) -> str:
    """Human message for the dashboard. Order matters: most specific first."""
    if isinstance(e, ClaudeCodeError):
        hint = (" Open a terminal, run `claude`, and log in with your Claude account." if "log" in str(e).lower() or "auth" in str(e).lower() else "")
        return f"Claude Code could not answer: {e}{hint}"
    if isinstance(e, anthropic.AuthenticationError) or (isinstance(e, TypeError) and "authentication" in str(e)):
        return "Anthropic API key missing or invalid. Set the ANTHROPIC_API_KEY environment variable and restart the app."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "This API key is not permitted to use the configured model."
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited by the Claude API. Wait a minute and try again."
    if isinstance(e, anthropic.APIStatusError) and "credit balance" in str(e.message).lower():
        return ("The API key works, but the Anthropic account has no credit. Add credit at console.anthropic.com "
                "(Plans & Billing), then click again. A Claude Max / Pro subscription does not fund the API.")
    if isinstance(e, anthropic.APIStatusError):
        return f"Claude API error {e.status_code}: {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "Could not reach the Claude API. Check the internet connection."
    if isinstance(e, anthropic.AnthropicError):
        return f"Claude client error: {e}"
    raise e
