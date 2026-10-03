"""Paths and tunable constants for the demand analysis system."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """KEY=VALUE lines from .env into the environment. A variable already set in the real environment wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")                # holds ANTHROPIC_API_KEY; never committed (see .gitignore)


def _cloud_secret(name: str) -> str | None:
    """A setting from Streamlit's own secrets store, or None. Never raises: there is no secrets store when
    the app runs locally, and reading one that does not exist is not an error."""
    try:
        import streamlit as st
        return st.secrets.get(name) or None
    except Exception:
        return None


def setting(name: str, default: str = "") -> str:
    """A setting from Streamlit's secrets (Community Cloud), else the environment (which .env feeds), else `default`.

    Community Cloud keeps secrets in the app's dashboard and hands them to the app only through st.secrets,
    never as environment variables, so os.environ on its own finds nothing once the app is deployed.
    """
    value = _cloud_secret(name) or os.environ.get(name) or ""
    value = str(value).strip().strip('"').strip("'")
    return value or default


ANTHROPIC_API_KEY = setting("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:                     # the Anthropic SDK reads the key from the environment
    os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)

RAW_DIR = ROOT / "data" / "raw"            # drop new Amazon "All Orders" reports here
STORE_DIR = ROOT / "data" / "store"        # cleaned parquet store (generated)
REF_DIR = ROOT / "data" / "reference"      # editable reference tables (events calendar)

ORDERS_PARQUET = STORE_DIR / "orders.parquet"
PRODUCTS_PARQUET = STORE_DIR / "products.parquet"
WEATHER_PARQUET = STORE_DIR / "weather.parquet"
NORMALS_PARQUET = STORE_DIR / "climate_normals.parquet"
MANIFEST_JSON = STORE_DIR / "ingest_manifest.json"
EVENTS_CSV = REF_DIR / "events.csv"
FABRIC_PARQUET = STORE_DIR / "fabric.parquet"        # cleaned "Fabric Consumption" sheet
FABRIC_MANIFEST = STORE_DIR / "fabric_manifest.json"
OUTSOURCE_PARQUET = STORE_DIR / "outsource.parquet"            # latest "Outsource" stock snapshot
OUTSOURCE_HISTORY = STORE_DIR / "outsource_history.parquet"    # every snapshot ever ingested
OUTSOURCE_MANIFEST = STORE_DIR / "outsource_manifest.json"

TIMEZONE = "Asia/Kolkata"                  # report timestamps are UTC; business runs on IST

# Claude analyst. Two ways to reach Claude, chosen by AI_BACKEND in .env (or the environment):
#   claude_code  - the Claude Code CLI on this machine, running on the user's Claude subscription login: no API bill,
#                  but it only works on a computer where `claude` is installed and logged in
#   api          - the Anthropic API with ANTHROPIC_API_KEY: pay-as-you-go, works on any server, right for a team
#   auto         - claude_code when the CLI is installed, otherwise api
AI_BACKEND = setting("AI_BACKEND", "auto").strip().lower()
CLAUDE_MODEL = "claude-opus-5"                                   # api backend
CLAUDE_CODE_MODEL = setting("CLAUDE_CODE_MODEL", "sonnet")        # claude_code backend: sonnet is quick and light on usage limits
ANALYST_DB = STORE_DIR / "analyst.duckdb"

# Supply lead times used when advice says "order by" / "be at Amazon by". PLACEHOLDERS: set these to HomeMonde's real figures.
PRODUCTION_WEEKS = 8        # from placing a production order to finished goods
INBOUND_WEEKS = 2           # from dispatch to stock being live at Amazon (FBA receive)

# Outsourced (bought-in) products. PLACEHOLDERS except OUTSOURCE_AVAILABLE's default: set them to HomeMonde's real figures.
OUTSOURCE_AVAILABLE = setting("OUTSOURCE_AVAILABLE", "max")   # how ready and virtual combine: max | sum | ready | virtual
OUTSOURCE_LEAD_WEEKS = 4    # from ordering from the outside maker to stock being sellable
OUTSOURCE_REVIEW_WEEKS = 8  # a reorder should cover demand for the lead time plus this many weeks

# Lifecycle classification
LAUNCH_DAYS = 60            # a product younger than this is in Launch
DORMANT_DAYS = 28           # no sale for this long -> Dormant
CENSOR_DAYS = 14            # first sale within this many days of data start => true launch date unknown
SPORADIC_UNITS_26W = 20     # fewer units than this in the last 26 weeks -> trend not measurable
TREND_WEEKS = 8             # window for the recent trend slope
GROWTH_PCT_WK = 3.0         # share-adjusted growth >= this %/week -> Growth
DECLINE_PCT_WK = -3.0       # share-adjusted growth <= this %/week -> Decline
STOCKOUT_GAP_DAYS = 7       # zero-sale streak on a >=1/day product that later resumes

# Climate model
CLIMATE_MIN_STATE_UNITS = 3000   # states below this total volume are left out of the weather regression
