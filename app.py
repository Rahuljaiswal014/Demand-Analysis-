"""HomeMonde demand analysis dashboard.   Run:  streamlit run app.py"""
import gc
import hmac
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demand import analytics as A
from demand import config as cfg
from demand import forecast as F
from demand import insights as INS
from demand import fabric as FB
from demand import predict as ML
from demand import sourcing as SRC

st.set_page_config(page_title="HomeMonde Demand Analysis", page_icon="📈", layout="wide")
gc.collect()      # plotly figures and big frames from the previous run sit in reference cycles; free them before this run allocates
try:              # pandas 3 keeps string columns in Arrow memory, which is handed back to the OS only on request
    import pyarrow as _pa
    _pa.default_memory_pool().release_unused()
except Exception:
    pass

# Optional viewer gate for a hosted copy. Set APP_PASSWORD in Streamlit's secrets (Community Cloud) or in .env and every
# browser session is asked for it once. Left unset, as on this machine, the dashboard opens straight away.
_GATE = cfg.setting("APP_PASSWORD")
if _GATE and not st.session_state.get("_unlocked"):
    st.title("HomeMonde Demand Analysis")
    with st.form("gate"):
        _pw = st.text_input("Password", type="password", help="Ask the HomeMonde founder's office for access.")
        if st.form_submit_button("Open dashboard"):
            if hmac.compare_digest(_pw.encode(), _GATE.encode()):
                st.session_state["_unlocked"] = True
                st.rerun()
            st.error("That password is not right.")
    st.stop()

# ------------------------------------------------------------------ chart system
# Categorical hues are assigned in this fixed order and follow the entity, never its rank.
LIGHT = dict(series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
             surface="#f6f6f3", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
             mid="#f0efec", seq=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DARK = dict(series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
            surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
            mid="#383835", seq=["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"])
# Follow whatever theme the browser is actually showing (the viewer picks Light / Dark under the ⋮ menu > Settings), so the
# charts and cards always match the page. Falls back to the config file's base when the runtime theme is unavailable.
def _theme_is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return st.get_option("theme.base") == "dark"


P = DARK if _theme_is_dark() else LIGHT
OTHER = P["muted"]
DIVERGING = [[0, P["series"][0]], [0.5, P["mid"]], [1, P["series"][7]]]     # blue <-> red, neutral midpoint


st.markdown("""
<style>
  /* Everything below uses Streamlit's own theme variables, so it flips the instant the viewer picks Light / Dark
     in the ⋮ menu. Chart colours (Plotly) come from P and refresh on the next rerun. */
  /* Streamlit exposes no theme CSS variables, but the text colour (currentColor) flips with the theme instantly, so every
     surface is a faint tint of it over the page: slightly darker cards on the light theme, slightly lighter on the dark one. */
  .stApp {
    --card-bg: color-mix(in srgb, currentColor 5%, transparent);
    --card-border: color-mix(in srgb, currentColor 12%, transparent);
    --muted: color-mix(in srgb, currentColor 62%, transparent);
  }
  .block-container { max-width: 1500px; padding-top: 2.2rem; padding-bottom: 4rem; }
  h1 { font-size: 1.9rem !important; font-weight: 650 !important; letter-spacing: -0.01em; padding-bottom: 0.4rem !important; }
  h3 { font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 0.6rem; }
  [data-testid="stVerticalBlockBorderWrapper"] { background: var(--card-bg); border-radius: 14px !important; border-color: var(--card-border) !important; }
  [data-testid="stPlotlyChart"] { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; padding: 10px 12px 6px 12px; }
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
  [data-testid="stMetric"] { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; padding: 14px 16px; }
  [data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.8rem; }
  [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 650; color: var(--text-color); }
  [data-testid="stVerticalBlockBorderWrapper"] h1, [data-testid="stVerticalBlockBorderWrapper"] h2,
  [data-testid="stVerticalBlockBorderWrapper"] h3, [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
  [data-testid="stChatMessage"] h3 { font-size: 1.02rem !important; font-weight: 650 !important; padding: 0.6rem 0 0.2rem 0 !important; letter-spacing: 0; }
  [data-testid="stChatMessage"] table, [data-testid="stVerticalBlockBorderWrapper"] table { font-size: 0.86rem; }
  .card-title { font-weight: 600; font-size: 1.0rem; color: var(--text-color); margin-bottom: 2px; }
  .card-note { font-size: 0.82rem; color: var(--muted); margin-bottom: 4px; }
  [data-testid="stCaptionContainer"] { color: var(--muted); }
</style>""", unsafe_allow_html=True)


def style(fig: go.Figure, height=340, legend=True, ytitle=None, xtitle=None) -> go.Figure:
    """One look for every chart: transparent so the card shows through, margins that grow to fit their labels,
    legend in its own row above the plot, recessive grid and axes."""
    fig.update_layout(
        height=height, margin=dict(l=12, r=36, t=44 if legend else 16, b=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=P["ink2"]),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font=dict(color=P["ink2"], size=12),
                                       bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
        hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["axis"], font=dict(color=P["ink"], size=12)), hovermode="x unified",
        separators=".,")
    fig.update_xaxes(showgrid=False, showline=True, linecolor=P["axis"], ticks="outside", ticklen=4, tickcolor=P["axis"],
                     tickfont=dict(color=P["muted"], size=11), title=dict(text=xtitle, font=dict(color=P["muted"], size=12), standoff=10),
                     automargin=True)
    fig.update_yaxes(gridcolor=P["grid"], zeroline=False, tickfont=dict(color=P["muted"], size=11), rangemode="tozero", automargin=True,
                     title=dict(text=ytitle, font=dict(color=P["muted"], size=12), standoff=12), tickformat="~s", ticksuffix=" ")
    return fig


def show(fig):
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


class card:
    """A framed panel: title and optional note on top, content inside. `with card("Units per day"): show(fig)`"""

    def __init__(self, title: str | None = None, note: str | None = None):
        self.title, self.note = title, note

    def __enter__(self):
        self.box = st.container(border=True)
        self.box.__enter__()
        if self.title:
            st.markdown(f"<div class='card-title'>{self.title}</div>" + (f"<div class='card-note'>{self.note}</div>" if self.note else ""),
                        unsafe_allow_html=True)
        return self

    def __exit__(self, *exc):
        return self.box.__exit__(*exc)


def pretty(name) -> str:
    """snake_case column name -> readable header."""
    t = str(name).replace("_pct", " %").replace("pct_", "% ").replace("_", " ").strip()
    for a, b in (("asp", "avg price"), ("4w", "4 wk"), ("26w", "26 wk"), ("sku", "SKU"), ("asin", "ASIN")):
        t = re.sub(rf"\b{a}\b", b, t)
    return t[:1].upper() + t[1:]


def table(data, **kw):
    """st.dataframe with readable headers for every column the caller did not label itself."""
    cfg = dict(kw.pop("column_config", {}) or {})
    cols = getattr(data, "columns", None)
    if cols is not None and not isinstance(cols, pd.MultiIndex):
        for col in cols:
            cfg.setdefault(col, st.column_config.Column(pretty(col)))
    kw.setdefault("width", "stretch")
    return st.dataframe(data, column_config=cfg, **kw)


def lines(df: pd.DataFrame, colors: dict, height=340, ytitle="Units", width=2) -> go.Figure:
    fig = go.Figure()
    for name in df.columns:
        fig.add_scatter(x=df.index, y=df[name], name=str(name), mode="lines", line=dict(color=colors[name], width=width),
                        hovertemplate="%{y:,.0f}")
    return style(fig, height, legend=df.shape[1] > 1, ytitle=ytitle)


def hbar(s: pd.Series, color=None, height=None, fmt=",.0f", xtitle=None) -> go.Figure:
    """Ranked horizontal bars with the value written at the bar end, so the value axis itself is dropped."""
    s = s.iloc[::-1]
    lo, hi = min(float(s.min()), 0.0), max(float(s.max()), 0.0)
    pad = (hi - lo) * 0.24 or 1                                  # room for the value label even in a narrow column
    fig = go.Figure(go.Bar(x=s.values, y=[str(i) for i in s.index], orientation="h", marker=dict(color=color or P["series"][0], cornerradius=4),
                           text=[format(v, fmt) for v in s.values], textposition="outside", textfont=dict(color=P["ink2"], size=11),
                           cliponaxis=False, hovertemplate="%{y}: %{x:,.1f}<extra></extra>"))
    fig = style(fig, height or max(200, 28 * len(s) + 44), legend=False, xtitle=xtitle)
    fig.update_layout(hovermode="closest", bargap=0.32, margin=dict(b=40 if xtitle else 12))
    fig.update_xaxes(range=[lo - (pad if lo < 0 else 0), hi + pad], showticklabels=False, ticks="", showline=False, showgrid=False,
                     zeroline=lo < 0, zerolinecolor=P["axis"])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=P["ink2"], size=12), ticksuffix="  ", tickformat=None, rangemode="normal")
    return fig


def heat(df: pd.DataFrame, diverging_mid=None, height=None, fmt=".0f") -> go.Figure:
    """Labelled matrix. Diverging scale (blue - neutral - red) around `diverging_mid`, else a single-hue ramp."""
    top = float(np.nanmax(df.to_numpy(dtype=float)))
    if diverging_mid:
        zmax = min(max(top, diverging_mid * 1.2), 2.2 * diverging_mid)
        kw = dict(colorscale=DIVERGING, zmid=diverging_mid, zmin=max(0, 2 * diverging_mid - zmax), zmax=zmax)
    else:
        kw = dict(colorscale=[[i / 4, c] for i, c in enumerate(P["seq"])])
    def head(c):
        c = str(c)
        return pd.Period(c).strftime("%b %y") if re.fullmatch(r"\d{4}-\d{2}", c) else c

    fig = go.Figure(go.Heatmap(z=df.values, x=[head(c) for c in df.columns], y=[str(i) for i in df.index], xgap=3, ygap=3, showscale=False,
                               text=df.values, texttemplate=f"%{{text:{fmt}}}", textfont=dict(size=11),
                               hovertemplate="%{y} · %{x}: <b>%{z:.0f}</b><extra></extra>", **kw))
    fig = style(fig, height or max(240, 30 * len(df) + 70), legend=False)
    fig.update_layout(hovermode="closest", margin=dict(t=8))
    fig.update_xaxes(side="top", showline=False, ticks="", type="category", tickangle=0 if len(df.columns) <= 14 else -45,
                     tickfont=dict(color=P["ink2"], size=11))
    fig.update_yaxes(autorange="reversed", showgrid=False, rangemode="normal", type="category", tickformat=None, ticksuffix="  ", ticks="",
                     tickfont=dict(color=P["ink2"], size=12))
    return fig


def mark_forecast_start(fig: go.Figure, x, label="predicted →"):
    """Hairline where actuals end and the model takes over."""
    fig.add_vline(x=x, line=dict(color=P["axis"], width=1, dash="dot"))
    fig.add_annotation(x=x, y=1, yref="paper", text=label, showarrow=False, xanchor="left", yanchor="bottom", xshift=6,
                       font=dict(size=11, color=P["muted"]))
    return fig


def inr(v: float) -> str:
    if v >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if v >= 1e5:
        return f"₹{v / 1e5:,.1f} L"
    return f"₹{v:,.0f}"


# ------------------------------------------------------------------ data
def _mtime() -> float:
    """Cache key for everything derived from the stores: changes when the orders or either reference sheet is re-ingested."""
    return max((f.stat().st_mtime for f in (cfg.ORDERS_PARQUET, cfg.FABRIC_PARQUET, cfg.OUTSOURCE_PARQUET) if f.exists()), default=0.0)


@st.cache_resource(show_spinner="Loading order store …")
def load(mtime: float):
    """The four base tables, held once for the whole server. cache_resource hands every run the same objects, where
    cache_data would keep a pickled copy and unpickle another ~600 MB for each rerun; the tables are read-only by
    convention (every page filters or copies, nothing assigns into them), which the page test checks by hashing them."""
    orders, products, weather = A.load_orders(), A.load_products(), A.load_weather()
    attrs = products[["asin", "main_sku", "curtain_type", "curtain_length_ft", "pack_size"]]
    d = A.add_context(A.demand(orders), weather).merge(attrs, on="asin", how="left")
    d["curtain_length"] = d["curtain_length_ft"].map(lambda v: f"{int(v)} ft" if pd.notna(v) else None)
    d["sourcing"] = d["asin"].map(SRC.build(products, orders)[1].set_index("asin")["sourcing"]).fillna(SRC.UNCLASSIFIED)
    # Only the Data page's reconciliation and the outsource SKU matching read the raw order lines; keeping just their columns
    # resident saves ~150 MB on the server. The analyst database is built from the full store separately.
    orders = orders[["date", "sku", "asin", "quantity", "revenue", "is_demand", "is_cancelled", "is_amazon", "source_file"]]
    return orders, d, products, weather


@st.cache_data(show_spinner="Mapping the outsource sheet …")
def sourcing_tables(mtime: float) -> tuple:
    """(outsource sheet matched to listings, sourcing per product)."""
    orders, _, products, _ = load(mtime)
    return SRC.build(products, orders)


@st.cache_data(show_spinner="Measuring stock against the forecast …")
def stock_cover(mtime: float) -> pd.DataFrame:
    _, d, products, _ = load(mtime)
    return SRC.stock_cover(d, sourcing_tables(mtime)[0], products, lifecycle(mtime))


@st.cache_data(show_spinner="Classifying lifecycle stages …")
def lifecycle(mtime: float) -> pd.DataFrame:
    _, d, products, _ = load(mtime)
    return A.lifecycle(d, products)


@st.cache_resource(show_spinner="Training the prediction model …")
def ml_model(mtime: float):
    _, d, _, weather = load(mtime)
    return ML.fit(d, weather)


@st.cache_data(show_spinner="Predicting …")
def ml_predict(mtime: float, until: str, scenario: str, growth: float) -> pd.DataFrame:
    _, d, _, weather = load(mtime)
    return ML.predict(d, weather, ML.load_normals(weather)[0], until, scenario, growth, model=ml_model(mtime))


@st.cache_data(show_spinner="Predicting by state …")
def ml_predict_state(mtime: float) -> pd.DataFrame:
    _, d, _, weather = load(mtime)
    return ML.predict(d, weather, ML.load_normals(weather)[0], "2028-12-31", model=ml_model(mtime), level="state")


@st.cache_data(show_spinner="Reading what the model learned …")
def ml_drivers(mtime: float) -> dict:
    _, d, _, weather = load(mtime)
    return ML.drivers(d, weather, ML.load_normals(weather)[0], model=ml_model(mtime))


@st.cache_data(show_spinner="Cross-validating (about 10 s) …")
def ml_validation(mtime: float) -> tuple:
    _, d, _, weather = load(mtime)
    return ML.validate_blocked(d, weather), ML.validate(d, weather)


@st.cache_data(show_spinner="Joining the fabric sheet …")
def fabric_map(mtime: float, fab_mtime: float) -> pd.DataFrame:
    ps = sourcing_tables(mtime)[1]
    return FB.product_fabric(load(mtime)[2], FB.load_fabric(), set(ps.loc[ps["sourcing"] == SRC.OUTSOURCED, "asin"]))


@st.cache_data(show_spinner="Listing SKUs without fabric data …")
def missing_skus(mtime: float) -> pd.DataFrame:
    _, d, products, _ = load(mtime)
    return FB.missing_skus(fabric_map(mtime, _fab_mtime()), d, products)


def _fab_mtime() -> float:
    return cfg.FABRIC_PARQUET.stat().st_mtime if cfg.FABRIC_PARQUET.exists() else 0.0


@st.cache_data(show_spinner="Forecasting fabric requirement …")
def fabric_requirement(mtime: float, fab_mtime: float, filt: tuple, horizon: int) -> pd.DataFrame:
    return FB.requirement(D, fabric_map(mtime, fab_mtime), horizon)


@st.cache_data(show_spinner="Ranking …")
def ranked(mtime: float, filt: tuple, item: str, context: str, scope, n: int, min_units: int, metric: str) -> pd.DataFrame:
    return A.rank_cells(D, item, context, scope, n, min_units, metric)       # `filt` only keys the cache to the sidebar filters


@st.cache_data(show_spinner="Forecasting …")
def forecasts(mtime: float, filt: tuple, by, horizon: int) -> pd.DataFrame:
    return F.forecast(D, by, horizon)


@st.cache_data(show_spinner="Backtesting …")
def backtests(mtime: float, filt: tuple, by) -> pd.DataFrame:
    return F.rolling_backtest(D, by)


@st.cache_data(show_spinner=False)
def events() -> pd.DataFrame:
    return pd.read_csv(cfg.EVENTS_CSV, parse_dates=["date", "end_date"]) if cfg.EVENTS_CSV.exists() else pd.DataFrame()


if not cfg.ORDERS_PARQUET.exists():
    st.title("HomeMonde Demand Analysis")
    st.info(f"No data yet. Put Amazon **All Orders** reports in `{cfg.RAW_DIR}` and run `python -m demand.ingest`, or upload below.")
    up = st.file_uploader("All Orders report (.xlsx / .txt / .csv)", type=["xlsx", "txt", "tsv", "csv"])
    if up and st.button("Ingest"):
        from demand.ingest import ingest
        cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
        (cfg.RAW_DIR / up.name).write_bytes(up.getbuffer())
        ingest(log=st.write)
        st.rerun()
    st.stop()

ORDERS, D_ALL, PRODUCTS, WEATHER = load(_mtime())
CATS = D_ALL.groupby("category")["quantity"].sum().sort_values(ascending=False).index.tolist()
N_MONTHS = D_ALL["month"].nunique()                              # months of history, used in captions instead of a fixed number
HIST_SPAN = f"{D_ALL['date'].min():%b %Y} to {D_ALL['date'].max():%b %Y}"
CAT_COLOR = {c: (P["series"][i] if i < 7 else OTHER) for i, c in enumerate(CATS)} | {"Other categories": OTHER}
STAGE_COLOR = dict(zip(A.STAGES, P["series"]))

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### HomeMonde · Demand")
    PAGES = ["Overview", "AI suggestions", "AI analyst", "Demand calendar", "Most & least ordered", "Products", "Geography", "Trends", "Season & climate", "Lifecycle",
             "Forecast & suggestions", "Future years (ML)", "Fabric consumption", "Outsourced stock", "Data"]
    _want = st.query_params.get("page", "")                      # ?page=Geography opens that view directly (shareable links)
    page = st.radio("View", PAGES, index=PAGES.index(_want) if _want in PAGES else 0, label_visibility="collapsed")
    st.divider()
    lo, hi = D_ALL["date"].min().date(), D_ALL["date"].max().date()
    period = st.date_input("Period", (lo, hi), min_value=lo, max_value=hi)
    f_cats = st.multiselect("Category", CATS, placeholder="All categories")
    f_regions = st.multiselect("Region", sorted(D_ALL["region"].unique()), placeholder="All regions")
    f_states = st.multiselect("State", sorted(D_ALL["state"].unique()), placeholder="All states")
    _pool = D_ALL[D_ALL["state"].isin(f_states)] if f_states else D_ALL
    f_districts = st.multiselect("District", sorted(_pool["district"].unique()), placeholder="All districts")
    _src = [x for x in (SRC.IN_HOUSE, SRC.OUTSOURCED, SRC.BOTH, SRC.UNCLASSIFIED) if x in set(D_ALL["sourcing"].unique())]
    f_sourcing = st.multiselect("Sourcing", _src, placeholder="In-house and outsourced",
                                help="In-house = on the fabric sheet; Outsourced = on the outsource sheet; the rest by SKU range.") if len(_src) > 1 else []
    st.caption(f"Data: {lo:%d %b %Y} – {hi:%d %b %Y} · Amazon.in, cancellations and MCF transfers excluded")

start, end = (period if isinstance(period, tuple) and len(period) == 2 else (lo, hi))
mask = D_ALL["date"].between(pd.Timestamp(start), pd.Timestamp(end))
if f_cats:
    mask &= D_ALL["category"].isin(f_cats)
if f_regions:
    mask &= D_ALL["region"].isin(f_regions)
if f_states:
    mask &= D_ALL["state"].isin(f_states)
if f_districts:
    mask &= D_ALL["district"].isin(f_districts)
if f_sourcing:
    mask &= D_ALL["sourcing"].isin(f_sourcing)
D = D_ALL if bool(mask.all()) else D_ALL[mask]          # no filter, no copy: the unfiltered view is the common case
FILT = (str(start), str(end), tuple(f_cats), tuple(f_regions), tuple(f_states), tuple(f_districts), tuple(f_sourcing))
TITLES = PRODUCTS.set_index("asin")["title"]
SKU_TITLE = D_ALL.drop_duplicates("sku").set_index("sku")["asin"].map(TITLES)
ITEM_LEVELS = {"SKU (aliases merged)": "main_sku", "SKU (raw, as in report)": "sku", "Design family": "family", "Category": "category"}


def with_title(df: pd.DataFrame, item: str, col: str | None = None) -> pd.DataFrame:
    """Add a readable product title next to a sku / main_sku column."""
    col = col or item
    if item not in ("main_sku", "sku") or col not in df.columns:
        return df
    out = df.copy()
    out.insert(out.columns.get_loc(col) + 1, "title" if col == item else f"{col}_title", out[col].map(SKU_TITLE).str[:70])
    return out


if D.empty:
    st.warning("No sales match these filters.")
    st.stop()


@st.cache_data(show_spinner=False)
def _ai_label() -> str:
    from demand import ai
    return ai.backend_label()


def top_n_plus_other(d: pd.DataFrame, by: str, n: int = 7) -> pd.DataFrame:
    """Weekly units for the n largest groups, the rest folded into one 'Other' series."""
    w = A.weekly_matrix(d, by).T
    order = w.sum().sort_values(ascending=False).index
    out = w[order[:n]].copy()
    if len(order) > n:
        out[f"Other {by.replace('_', ' ')}s" if by != "category" else "Other categories"] = w[order[n:]].sum(axis=1)
    return out


def key_points(markdown: str, title: str = "Key points"):
    """Rule-based read-out of the page (demand/insights.py): free, instant, always in step with the numbers."""
    if markdown:
        with st.container(border=True):
            st.markdown(f"<div class='card-title'>{title}</div><div class='card-note'>computed from the figures on this page, no AI</div>",
                        unsafe_allow_html=True)
            st.markdown(markdown)


def ai_brief(tables: dict, focus: str, key: str):
    """Button that asks Claude to narrate the tables on this page."""
    if st.button("✨ Go deeper with Claude", key=f"brief_{key}", help="Runs through: " + _ai_label()):
        from demand import ai
        try:
            with st.spinner("Claude is reading the numbers …"):
                st.session_state[f"brief_out_{key}"] = ai.brief(tables, focus)
        except Exception as e:                       # friendly_error re-raises anything that is not an API problem
            st.session_state[f"brief_out_{key}"] = "⚠️ " + ai.friendly_error(e)
    if out := st.session_state.get(f"brief_out_{key}"):
        with st.container(border=True):
            st.markdown(out)


# ================================================================== pages
if page == "Overview":
    st.title("Demand overview")
    s = A.summary(D.assign(all="all"), "all").iloc[0]
    days = D["date"].nunique()
    k = st.columns(6)
    k[0].metric("Units", f"{s.units:,.0f}")
    k[1].metric("Revenue", inr(s.revenue))
    k[2].metric("Orders", f"{s.orders:,.0f}")
    k[3].metric("Avg selling price", f"₹{s.asp:,.0f}")
    k[4].metric("Units / day", f"{s.units / days:,.0f}")
    k[5].metric("Last 4 wks vs prior 4", f"{s.get('growth_4w_pct', float('nan')):+.1f}%")

    daily = A.daily_series(D)
    fig = go.Figure()
    fig.add_scatter(x=daily.index, y=daily["units"], name="Daily units", mode="lines", line=dict(color=P["muted"], width=1), opacity=0.55,
                    hovertemplate="%{y:,.0f}")
    fig.add_scatter(x=daily.index, y=daily["units"].rolling(7, center=True).mean(), name="7-day average", mode="lines",
                    line=dict(color=P["series"][0], width=2.5), hovertemplate="%{y:,.0f}")
    ev = A.demand_events(D)
    for _, e in (ev[ev["type"] == "Spike"] if not ev.empty else ev).iterrows():
        fig.add_vrect(x0=e["start"] - pd.Timedelta(hours=12), x1=e["end"] + pd.Timedelta(hours=12), fillcolor=P["series"][3], opacity=0.16, line_width=0)
    shown = [e for _, e in events().iterrows() if daily.index.min() <= e["date"] <= daily.index.max()]
    for i, e in enumerate(shown):                                  # three staggered rows so neighbouring festivals never collide
        fig.add_vline(x=e["date"], line=dict(color=P["axis"], width=1, dash="dot"))
        late = (e["date"] - daily.index.min()) / (daily.index.max() - daily.index.min()) > 0.85
        fig.add_annotation(x=e["date"], y=1.0, yref="paper", yshift=-2 - 13 * (i % 3), text=e["event"], showarrow=False, yanchor="top",
                           xanchor="right" if late else "left", xshift=-4 if late else 4, font=dict(size=10, color=P["muted"]))
    st.subheader("Units per day")
    show(style(fig, 380, ytitle="Units / day"))
    st.caption("Shaded bands are detected demand spikes (≥40% above the weekday-adjusted 4-week norm). Festival labels come from `data/reference/events.csv`.")

    c1, c2 = st.columns(2)
    cat = A.summary(D, "category")
    with c1:
        st.subheader("Units by category")
        show(hbar(cat["units"], color=[CAT_COLOR[c] for c in cat.index[::-1]], height=max(200, 28 * len(cat) + 44)))
    with c2:
        st.subheader("Units by region")
        show(hbar(A.summary(D[D["region"] != "Unknown"], "region")["units"], height=max(200, 28 * len(cat) + 44)))
    key_points(INS.overview(D))
    ai_brief({"category summary": cat, "state summary (top 12)": A.state_table(D).head(12), "demand spikes and slumps": ev,
              "category seasonal index by month (100 = own average)": A.seasonal_profile(D)},
             "Give an executive read-out of demand for the selected period.", "overview")

elif page == "Demand calendar":
    st.title("What is in demand, where, in which month")
    if WEATHER.empty:
        st.warning("No weather loaded, so future months cannot be predicted. Run `python -m demand.weather` once.")
        st.stop()
    c = st.columns([2, 3, 3, 2, 1])
    lvl_label = c[0].selectbox("Place", list(ML.PLACE_LEVELS))
    level = ML.PLACE_LEVELS[lvl_label]
    place = None
    by_units = lambda col, frame: frame.groupby(col)["quantity"].sum().sort_values(ascending=False).index.tolist()
    if level == "district":
        st_pick = c[1].selectbox("State", [s for s in by_units("state", D_ALL) if s != "UNKNOWN"])
        place = c[2].selectbox("District", [x for x in by_units("district", D_ALL[D_ALL["state"] == st_pick]) if x != "UNKNOWN"])
    elif level:
        place = c[1].selectbox(lvl_label, [x for x in by_units(level, D_ALL) if x.upper() != "UNKNOWN"])
    item_label = c[3].selectbox("Show", ["Category", "SKU"])
    top = c[4].number_input("Top", 3, 15, 5)
    item = "category" if item_label == "Category" else "main_sku"
    base = D_ALL[D_ALL["category"].isin(f_cats)] if f_cats else D_ALL
    pred_state = ml_predict_state(_mtime())
    if f_cats:
        pred_state = pred_state[pred_state["category"].isin(f_cats)]
    cal = ML.demand_calendar(base, pred_state, level, place, item, int(top))
    if cal.empty:
        st.warning("No sales for this place.")
        st.stop()
    where = place or "All India"
    here_units = (base if level is None else base[base[level] == place])["quantity"].sum()
    st.caption(f"**{where}** · {here_units:,.0f} units in the {N_MONTHS} months of history ({HIST_SPAN})"
               + (" · district location from PIN code via the India Post directory" if level == "district" else ""))

    def grid(kind):
        g = cal[cal["kind"] == kind]
        label = g["item"].map(lambda v: f"{v} · {str(SKU_TITLE.get(v, ''))[:38]}" if item == "main_sku" else v)
        cell = label + "  (" + g["units"].map("{:,.0f}".format) + ")"
        t = g.assign(cell=cell).pivot(index="rank", columns="month", values="cell")
        t.columns = [pd.Period(m).strftime("%b %y") for m in t.columns]
        return t

    st.subheader("Actual: top sellers each month")
    table(grid("actual"))
    st.subheader("Predicted: expected top sellers each month")
    fut = grid("predicted")
    table(fut.iloc[:, :16])
    notes = ["Future months come from the ML model (weather normals + festival calendar, 0% growth assumed); units in brackets."]
    if level == "district":
        notes.append("The model predicts at state level; a district gets its historical share of its state, category by category.")
    if item == "main_sku":
        notes.append("SKUs are the category prediction split by each SKU's share over the last 12 weeks (only SKUs sold in the last 28 days), "
                     "so the *order* of SKUs inside a category stays as it is today; what changes month to month is the category mix. "
                     "New launches and stock-outs will change it.")
    st.caption(" ".join(notes))

    pm = ML.place_monthly(base, pred_state, level, place)
    cats = [c_ for c_ in CATS if c_ in pm.columns]
    if True:                                                     # full-width rows: 23 months need the room
        st.subheader("Units per month")
        tot = pm[cats].sum(axis=1)
        fig = go.Figure()
        a, p_ = tot[pm["kind"] == "actual"], tot[pm["kind"] == "predicted"].iloc[:16]
        fig.add_scatter(x=a.index, y=a.values, name="Actual", mode="lines+markers", line=dict(color=P["ink2"], width=2), marker=dict(size=6))
        fig.add_scatter(x=p_.index, y=p_.values, name="Predicted", mode="lines+markers", line=dict(color=P["series"][0], width=2), marker=dict(size=6))
        show(mark_forecast_start(style(fig, 320, ytitle="Units / month"), p_.index[0]))
        st.subheader("When each category runs hot or cold")
        keep = [c_ for c_ in cats if pm[c_].sum() >= max(60, 0.004 * pm[cats].sum().sum())]
        yr = pm.loc[pm.index[:N_MONTHS + 12], keep]
        idx = (yr.div(yr.mean()) * 100).T
        idx.columns = [pd.Period(m).strftime("%b %y") for m in idx.columns]
        show(heat(idx, diverging_mid=100))
    st.caption(f"Heatmap: index 100 = that category's average month in this place; the first {N_MONTHS} columns are actual, the rest predicted.")

    key_points(INS.calendar(pm, where), "What to expect and when to order")
    ai_brief({f"top {item_label} per month, actual and predicted ({where})": cal,
              f"units by month and category ({where})": pm.iloc[:N_MONTHS + 12]},
             f"For {where}: say which products are in demand in which months, what to stock and when (production lead time about 6-8 weeks), "
             "which months need the most and the least inventory, and what is uncertain. Treat 'predicted' rows as model estimates.", "calendar")

elif page == "Most & least ordered":
    st.title("Most and least ordered")
    c = st.columns([2, 2, 2, 2])
    level = c[0].selectbox("Item", list(ITEM_LEVELS))
    contexts = {"Month": "month", "Week": "week", "Date": "date", "Season": "season", "Temperature band": "temp_band", "Rain": "rain_band"}
    ctx_label = c[1].selectbox("By", list(contexts))
    scope_label = c[2].selectbox("Within", ["All India", "Region", "State", "District"],
                                 help="District comes from the PIN code. Pick a state in the sidebar first to keep the table readable.")
    metric_label = c[3].selectbox("Rank by", ["Units", "Over-index (lift)"], disabled=ctx_label == "Date",
                                  help="Units names the biggest sellers. Over-index shows what is distinctively strong or weak in that "
                                       "region / season / climate: 100 = sells there exactly as it does everywhere.")
    item, context = ITEM_LEVELS[level], contexts[ctx_label]
    scope = {"All India": None, "Region": "region", "State": "state", "District": "district"}[scope_label]
    metric = "lift" if metric_label.startswith("Over") and ctx_label != "Date" else "units"
    c = st.columns([2, 2, 4])
    min_units = c[0].number_input("Item must have sold at least (units, in selection)", 1, 5000, 300 if metric == "lift" else 50, step=50)
    n = c[1].number_input("Show top / bottom", 1, 25, 5)
    if item == "sku":
        c[2].warning("Raw SKUs split one listing across aliases (%-OOS, -NEW, %-8EYT): a SKU can look like it died when sales just moved to its alias.", icon="⚠️")
    if context == "date" and scope in ("state", "district"):
        st.info("Date x State / District is too fine to be meaningful (most cells are 0-2 units). Use Region, or Month x State / District.")
        st.stop()

    rk = ranked(_mtime(), FILT, item, context, scope, int(n), int(min_units), metric)
    if rk.empty:
        st.warning("Nothing qualifies. Lower the minimum units.")
        st.stop()
    ex = A.extremes(rk, item)
    st.subheader(f"Winner and laggard per {ctx_label.lower()}" + (f" and {scope_label.lower()}" if scope else ""))
    table(with_title(with_title(ex, item, f"most_{item}"), item, f"least_{item}"), hide_index=True, height=380)
    st.caption("An item only competes in periods between its first and last sale, so unlaunched or discontinued items are not counted as "
               "'least ordered'. Ties at the bottom go to the bigger product. `zero_sellers` = competing items that sold nothing in that cell.")

    st.subheader("Look inside one cell")
    keys = [k for k in (scope, context) if k]
    labels = rk[keys].astype(str).agg(" · ".join, axis=1)
    cells = labels.drop_duplicates().tolist()
    pick = st.selectbox("Cell", cells, index=len(cells) - 1)
    sel = rk[labels == pick]
    c1, c2 = st.columns(2)
    for col, side, label in ((c1, "Most", "Most ordered"), (c2, "Least", "Least ordered")):
        with col:
            st.markdown(f"**{label}**")
            table(with_title(sel[sel["side"] == side][["rank", item, "units", "lift", "item_total_units"]], item), hide_index=True)

    st.subheader("When and where each item peaks")
    pk = A.item_peaks(D, item, int(min_units))
    table(with_title(pk, item), hide_index=True, height=380)
    st.caption("Best / weakest month are by units within the item's selling life. Over-indexed region / season / temperature band show where it sells "
               "disproportionately (lift 150 = 50% more than its overall share would predict).")
    ai_brief({"winner and laggard per cell": ex.tail(60), "item peaks (top 40)": pk.head(40)},
             f"Explain which {level} are most and least ordered by {ctx_label.lower()}" + (f" within each {scope_label.lower()}" if scope else "") +
             ", what pattern that shows, and what to do about the laggards.", "ranks")

elif page == "Forecast & suggestions":
    st.title("Forecast and suggestions")
    levels = {"Total": None, "Category": "category", "Region": "region", "Curtain type": "curtain_type", "SKU (aliases merged)": "main_sku"}
    c = st.columns([2, 2, 4])
    lvl = c[0].selectbox("Forecast level", list(levels))
    horizon = c[1].slider("Weeks ahead", 4, 16, 12)
    by = levels[lvl]
    try:
        fc = forecasts(_mtime(), FILT, by, horizon)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    hist = A.weekly_matrix(D.assign(_all="All") if by is None else D.dropna(subset=[by]), by or "_all")
    order = hist.iloc[:, -8:].sum(axis=1).sort_values(ascending=False).index
    have = set(fc["group"])
    groups = [g for g in order if g in have]
    grp = c[2].selectbox(lvl, groups, format_func=lambda g: f"{g} · {str(SKU_TITLE.get(g, ''))[:70]}" if by == "main_sku" else str(g)) if by else "All"
    one, h = fc[fc["group"] == grp].set_index("week"), hist.loc[grp].iloc[-30:]

    fig = go.Figure()
    fig.add_scatter(x=list(one.index) + list(one.index[::-1]), y=list(one["high"]) + list(one["low"][::-1]), fill="toself", line=dict(width=0),
                    fillcolor="rgba(42,120,214,0.15)", name="80% band", hoverinfo="skip")
    fig.add_scatter(x=h.index, y=h.values, name="Actual", mode="lines", line=dict(color=P["ink2"], width=2))
    fig.add_scatter(x=one.index, y=one["baseline"], name="Baseline forecast", mode="lines", line=dict(color=P["series"][0], width=2))
    if (one["festive_multiplier"] != 1).any():
        fig.add_scatter(x=one.index, y=one["festive_scenario"], name="Festive scenario (2025 uplift)", mode="lines",
                        line=dict(color=P["series"][1], width=2, dash="dot"))
    show(style(fig, 380, ytitle="Units / week"))
    k = st.columns(4)
    k[0].metric(f"Baseline, next {horizon} wks", f"{one['baseline'].sum():,.0f}")
    k[1].metric("Festive scenario", f"{one['festive_scenario'].sum():,.0f}")
    k[2].metric(f"Last {horizon} wks actual", f"{hist.loc[grp].iloc[-horizon:].sum():,.0f}")
    k[3].metric("Method", one["method"].iloc[0])
    key_points(INS.forecast(one.reset_index(), float(hist.loc[grp].iloc[-horizon:].sum()), horizon))
    st.caption("Baseline = damped-trend smoothing of recent weeks with an 80% band. The festive scenario applies the uplift seen at the same "
               "distance from Diwali in 2025 (Diwali 2026: 8 Nov, from `events.csv`), including the sale opening four weeks before Diwali week. "
               "It rests on one observation of one festive season. Treat it as the upside case, not a prediction.")

    if by:
        st.subheader(f"All {lvl.lower()} forecasts · next {horizon} weeks")
        tot = fc.groupby("group").agg(baseline=("baseline", "sum"), low=("low", "sum"), high=("high", "sum"),
                                      festive_scenario=("festive_scenario", "sum"), method=("method", "first"))
        tot[f"last_{horizon}w_actual"] = hist.iloc[:, -horizon:].sum(axis=1)
        tot["change_pct"] = ((tot["baseline"] / tot[f"last_{horizon}w_actual"].clip(lower=1) - 1) * 100).round(1)
        tot = tot.round(0).sort_values("baseline", ascending=False).reset_index().rename(columns={"group": by})
        table(with_title(tot, by), hide_index=True, height=320)

    with st.expander("How accurate is this? (backtest on your own history)"):
        if st.toggle("Run the backtest", key="run_bt"):                  # an expander's body always executes, so gate the work
            table(backtests(_mtime(), FILT, by), hide_index=True)
        st.markdown("The model is run as of four past dates and scored on the 8 weeks that followed. **wape_pct** is the week-by-week error, "
                    "**period_total_error_pct** the error on the 8-week total, **bias_pct** above 0 means it over-forecast. "
                    "At SKU level most items sell a handful of units a week, so weekly error is large: plan SKUs on multi-week totals, and "
                    "trust the top sellers' forecasts far more than the tail's.")

    st.subheader("Suggested actions")
    key_points(INS.actions(D, lifecycle(_mtime()), forecasts(_mtime(), FILT, "category", 12)), "Action list")
    st.caption("Above: rule-based, free and instant. Below: Claude reads the forecast, lifecycle, regional over-index and climate sensitivity "
               f"and writes a reasoned plan. Runs through: {_ai_label()}.")
    if st.button("✨ Ask Claude for a 12-week plan", type="primary"):
        from demand import ai
        lc = lifecycle(_mtime())
        cols = ["main_sku", "title", "category", "stage", "units_last_4w", "growth_vs_category_pct_wk", "recent_vs_peak_share", "stockout_suspected", "longest_gap_days"]
        tables = {
            "forecast by category (12-week totals)": F.forecast(D, "category").groupby("group")[["baseline", "festive_scenario"]].sum(),
            "forecast by region (12-week totals)": F.forecast(D, "region").groupby("group")[["baseline", "festive_scenario"]].sum(),
            "category: last 8 weeks direction": A.trend_table(D, "category"),
            "growth-stage products": lc[lc["stage"] == "Growth"].nlargest(20, "units_last_4w")[cols],
            "largest declining products": lc[lc["stage"] == "Decline"].nlargest(20, "units_total")[cols],
            "category over-index by region (100 = national rate)": A.category_index_by_state(D.assign(state=D["region"]), "category", 0),
            "category index by season (100 = own average)": A.season_index(D_ALL),
            "weather sensitivity by category": A.weather_sensitivity(D_ALL, WEATHER),
            "SKU peaks (top 30)": A.item_peaks(D, "main_sku", 300).head(30),
        }
        try:
            with st.spinner("Claude is drafting the plan …"):
                st.session_state["suggestions"] = ai.suggestions(tables)
        except Exception as e:
            st.session_state["suggestions"] = "⚠️ " + ai.friendly_error(e)
    if out := st.session_state.get("suggestions"):
        with st.container(border=True):
            st.markdown(out)

elif page == "Products":
    st.title("What is selling")
    levels = {"Product (ASIN)": "asin", "Design family": "family", "Category": "category", "Curtain type": "curtain_type",
              "Curtain length": "curtain_length", "Pack size": "pack_size"}
    level = st.segmented_control("Level", list(levels), default="Product (ASIN)") or "Product (ASIN)"
    by = levels[level]
    if by == "asin":
        ptab = A.product_table(D, PRODUCTS)
        view = ptab[["asin", "title", "category", "family", "units", "revenue", "asp", "unit_share_pct", "last_4w", "prev_4w", "growth_4w_pct"]]
    else:
        ptab = A.summary(D.dropna(subset=[by]), by).reset_index()
        view = ptab
    table(view, hide_index=True, height=380, column_config={
        "revenue": st.column_config.NumberColumn("revenue ₹", format="localized"), "units": st.column_config.NumberColumn(format="localized"),
        "unit_share_pct": st.column_config.ProgressColumn("unit share %", min_value=0, max_value=float(view["unit_share_pct"].max()), format="%.2f"),
        "growth_4w_pct": st.column_config.NumberColumn("4w growth %", format="%+.1f"), "title": st.column_config.TextColumn(width="large")})

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Concentration")
        pa = A.pareto(D)
        n80 = int((pa["cum_share_pct"] < 80).sum() + 1)
        fig = go.Figure(go.Scatter(x=pa["rank"], y=pa["cum_share_pct"], mode="lines", line=dict(color=P["series"][0], width=2),
                                   hovertemplate="Top %{x:,} products: %{y:.1f}% of units<extra></extra>"))
        fig.add_hline(y=80, line=dict(color=P["axis"], width=1, dash="dot"))
        show(style(fig, 300, legend=False, ytitle="Cumulative % of units", xtitle="Products ranked by units"))
        st.caption(f"**{n80:,}** of {len(pa):,} products make 80% of units; {int((D.groupby('asin')['quantity'].sum() < 10).sum()):,} sold fewer than 10.")
    with c2:
        st.subheader(f"Movers · last {cfg.TREND_WEEKS} weeks")
        tr = A.trend_table(D, by if by != "asin" else "asin", min_units=80)
        if tr.empty:
            st.info("Select at least 8 complete weeks to see movers.")
        else:
            if by == "asin":
                tr = tr.join(PRODUCTS.set_index("asin")["title"])
            movers = pd.concat([tr[tr["direction"] == "Rising"].head(8), tr[tr["direction"] == "Falling"].tail(8)])
            table(movers.reset_index(), hide_index=True, height=300,
                         column_config={"growth_pct_per_week": st.column_config.NumberColumn("% / week", format="%+.1f")})

    st.subheader("Product drill-down")
    opts = A.product_table(D, PRODUCTS).head(500)
    pick = st.selectbox("Product", opts["asin"], format_func=lambda a: f"{a} · {opts.set_index('asin').at[a, 'title'][:90]}")
    dp = D[D["asin"] == pick]
    w = A.weekly_matrix(dp.assign(k="Units"), "k").T
    w = w.reindex(A.complete_weeks(D), fill_value=0)
    c1, c2 = st.columns([2, 1])
    with c1:
        show(lines(w, {"Units": P["series"][0]}, 280, ytitle="Units / week"))
    with c2:
        show(hbar(dp.groupby("state")["quantity"].sum().sort_values(ascending=False).head(8), height=280))
    lc_row = lifecycle(_mtime()).set_index("asin").loc[pick]
    st.caption(f"Stage **{lc_row['stage']}** · SKUs: {PRODUCTS.set_index('asin').at[pick, 'skus']} · "
               f"median price ₹{dp['unit_price'].median():,.0f} · {'⚠️ possible stock-out gap of ' + str(int(lc_row['longest_gap_days'])) + ' days' if lc_row['stockout_suspected'] else 'no long sales gaps'}")

elif page == "Geography":
    st.title("Where it sells")
    stt = A.state_table(D)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Units by state")
        show(hbar(stt["units"].head(15)))
    with c2:
        st.subheader("Top cities")
        show(hbar(D.groupby("city")["quantity"].sum().sort_values(ascending=False).head(15)))
    st.subheader("State table")
    table(stt.reset_index()[["state", "region", "units", "revenue", "asp", "unit_share_pct", "asins", "growth_4w_pct"]], hide_index=True,
                 height=300, column_config={"growth_4w_pct": st.column_config.NumberColumn("4w growth %", format="%+.1f"),
                                                             "revenue": st.column_config.NumberColumn("revenue ₹", format="localized")})
    st.subheader("Districts")
    st_pick = st.selectbox("State", [s for s in stt.index if s != "UNKNOWN"], key="geo_state")
    ds = D[(D["state"] == st_pick) & (D["district"] != "UNKNOWN")]
    dt = A.summary(ds, "district")
    dt["share_of_state_pct"] = (dt["units"] / dt["units"].sum() * 100).round(1)
    top_cat = ds.groupby(["district", "category"])["quantity"].sum().reset_index().sort_values("quantity", ascending=False).drop_duplicates("district")
    dt["top_category"] = top_cat.set_index("district")["category"]
    best_sku = ds.groupby(["district", "main_sku"])["quantity"].sum().reset_index().sort_values("quantity", ascending=False).drop_duplicates("district")
    dt["top_sku"] = best_sku.set_index("district")["main_sku"]
    c1, c2 = st.columns([2, 3])
    with c1:
        show(hbar(dt["units"].head(12)))
    with c2:
        table(dt.reset_index()[["district", "units", "share_of_state_pct", "revenue", "asp", "growth_4w_pct", "top_category", "top_sku"]],
                     hide_index=True, height=380,
                     column_config={"growth_4w_pct": st.column_config.NumberColumn("4w growth %", format="%+.1f"),
                                    "revenue": st.column_config.NumberColumn("revenue ₹", format="localized")})
    big_d = dt.index[dt["units"] >= 400][:15]
    if len(big_d) >= 2:
        di = A.category_index_by_state(ds[ds["district"].isin(big_d)].assign(state=lambda x: x["district"]), "category", 0)
        di = di.loc[:, ds.groupby("category")["quantity"].sum().reindex(di.columns) >= 150]
        st.markdown(f"**What each district of {st_pick.title()} over- and under-buys** (index 100 = the state's own rate)")
        show(heat(di.loc[[x for x in big_d if x in di.index]], diverging_mid=100))
    st.caption("District = the India Post district of the order's PIN code (buyer-typed city names have 9,500 spellings and are not used). "
               "98.8% of orders match the directory exactly; 1.2% use the district of neighbouring PINs.")

    st.subheader("What each state over- and under-buys")
    dim = st.segmented_control("Compare by", ["category", "curtain_type", "curtain_length"], default="category", format_func=pretty) or "category"
    idx = A.category_index_by_state(D.dropna(subset=[dim]), dim, min_state_units=3000)
    idx = idx.loc[:, D.groupby(dim)["quantity"].sum().loc[idx.columns] >= 700]
    show(heat(idx.loc[stt.index.intersection(idx.index)], diverging_mid=100))
    st.caption("Index 100 = the state buys this group at the national rate. 130 = 30% more than its size predicts (red), 70 = 30% less (blue).")
    key_points(INS.geography(D))
    ai_brief({"state table": stt.head(20), f"{dim} index by state": idx}, "Explain the geographic pattern of demand and where to focus inventory placement and ads.", "geo")

elif page == "Trends":
    st.title("When demand rises and falls")
    dim = st.segmented_control("Break down by", ["category", "region", "curtain_type", "curtain_length"], default="category", format_func=pretty) or "category"
    wk = top_n_plus_other(D.dropna(subset=[dim]), dim)
    colors = {c: (CAT_COLOR.get(c) if dim == "category" else None) or (OTHER if str(c).startswith("Other ") else P["series"][i % 8]) for i, c in enumerate(wk.columns)}
    mode = st.segmented_control("Scale", ["Indexed (own average = 100)", "Units per week"], default="Indexed (own average = 100)",
                                help="Curtains are ~80% of units, so on a units scale every other line is flat along the bottom. "
                                     "Indexed puts each group on its own average so their shapes can be compared.") or "Indexed (own average = 100)"
    plot = wk if mode == "Units per week" else wk / wk.mean() * 100
    show(lines(plot, colors, 400, ytitle="Units / week" if mode == "Units per week" else "Index"))
    st.caption("Only complete Monday–Sunday weeks. Use the indexed view to compare the shape of small and large groups on one axis.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Day-of-week pattern")
        wd = A.weekday_profile(D)
        fig = go.Figure(go.Bar(x=wd.index, y=wd.values, marker=dict(color=P["series"][0], cornerradius=4), text=[f"{v:.0f}" for v in wd.values],
                               textposition="outside", hovertemplate="%{x}: %{y:.0f}<extra></extra>"))
        fig.add_hline(y=100, line=dict(color=P["axis"], width=1))
        show(style(fig, 300, legend=False, ytitle="Index (avg day = 100)"))
    with c2:
        st.subheader(f"Direction over the last {cfg.TREND_WEEKS} weeks")
        tr = A.trend_table(D.dropna(subset=[dim]), dim)
        table(tr.reset_index(), hide_index=True, height=300,
                     column_config={"growth_pct_per_week": st.column_config.NumberColumn("% / week", format="%+.1f")})
    st.subheader("Detected demand events")
    ev = A.demand_events(D)
    if not ev.empty:
        ev = ev.assign(start=ev["start"].dt.strftime("%d %b %Y"), end=ev["end"].dt.strftime("%d %b %Y"))
    if ev.empty:
        st.info("No spikes or slumps in this selection.")
    else:
        table(ev, hide_index=True)
    key_points(INS.trends(D, dim))
    ai_brief({"weekly units": wk.tail(16), "recent direction": tr, "events": ev, "weekday index": wd.to_frame("index")},
             "Explain what is rising, what is falling, and whether the moves look seasonal, event-driven or structural.", "trends")

elif page == "Season & climate":
    st.title("Season and climate")
    st.info(f"History is {N_MONTHS} months ({HIST_SPAN}): each calendar month has been seen once, so the seasonal shape is a single observation. "
            "The weather model below does not have that limit: it compares states with each other inside the same week.", icon="ℹ️")
    dim = st.segmented_control("Group", ["category", "curtain_type", "curtain_length"], default="category", format_func=pretty) or "category"
    dd = D_ALL.dropna(subset=[dim])                       # climate needs the full period and every state
    if f_cats:
        dd = dd[dd["category"].isin(f_cats)]
    big = dd.groupby(dim)["quantity"].sum()
    dd = dd[dd[dim].isin(big[big >= 700].index)]

    st.subheader("Seasonal shape · units per day by month")
    show(heat(A.seasonal_profile(dd, dim), diverging_mid=100))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By Indian season")
        show(heat(A.season_index(dd, dim), diverging_mid=100))
    with c2:
        st.subheader("Demand by temperature band")
        tr = A.temperature_response(dd, WEATHER, dim)
        if tr.empty:
            st.info("Weather not loaded. Run `python -m demand.weather`.")
        else:
            show(heat(tr, diverging_mid=100))
    st.caption("Index 100 = normal for that group. Temperature bands are weekly mean daily-max at each state's main city, net of state size and national week effects.")

    st.subheader("Weather sensitivity")
    ws = A.weather_sensitivity(dd, WEATHER, dim)
    if ws.empty:
        st.info("Not enough weather or volume for a regression on this selection.")
    else:
        eff = ws["per_1C_hotter_pct"].sort_values(ascending=False)
        show(hbar(eff, color=[P["series"][7] if v > 0 else P["series"][0] for v in eff.iloc[::-1]], fmt="+.1f", xtitle="% change in weekly units per +1°C"))
        table(ws.reset_index(), hide_index=True)
        st.caption("Two-way fixed effects (state + week) on log weekly units across the larger states. |t| ≥ 2.5 moderate, ≥ 4 strong; "
                   "standard errors are not clustered, so treat borderline values with caution.")

    st.subheader("State explorer")
    c1, c2 = st.columns(2)
    stt = c1.selectbox("State", A.state_table(D_ALL).index[:20])
    grp = c2.selectbox(dim.replace("_", " ").title(), big.sort_values(ascending=False).index)
    sw = A.weekly_matrix(dd[(dd["state"] == stt) & (dd[dim] == grp)].assign(k="Units"), "k").T.reindex(A.complete_weeks(D_ALL), fill_value=0)
    show(lines(sw, {"Units": P["series"][0]}, 230, ytitle=f"{grp} units / week"))
    if not WEATHER.empty:
        wx = WEATHER[WEATHER["state"] == stt].set_index("date").resample("W-MON", label="left", closed="left").agg({"temp_max": "mean", "rain_mm": "sum"})
        wx = wx.reindex(sw.index)
        c1, c2 = st.columns(2)
        with c1:
            show(lines(wx[["temp_max"]].rename(columns={"temp_max": "Max temp °C"}), {"Max temp °C": P["series"][7]}, 200, ytitle="°C"))
        with c2:
            show(lines(wx[["rain_mm"]].rename(columns={"rain_mm": "Rain mm"}), {"Rain mm": P["series"][2]}, 200, ytitle="mm / week"))
    key_points(INS.climate(dd, ws, dim))
    ai_brief({"seasonal index by month": A.seasonal_profile(dd, dim), "index by season": A.season_index(dd, dim), "weather sensitivity": ws,
              "demand index by temperature band": tr},
             "Explain how season and weather affect each group, which effects are trustworthy, and what that implies for inventory timing by region.", "climate")

elif page == "Lifecycle":
    st.title("Product lifecycle")
    lc = lifecycle(_mtime())
    if f_cats:
        lc = lc[lc["category"].isin(f_cats)]
    ss = A.stage_summary(lc)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Products per stage")
        show(hbar(ss["asins"], color=[STAGE_COLOR[s] for s in ss.index[::-1]]))
    with c2:
        st.subheader("Share of last 4 weeks' units")
        show(hbar(ss["share_of_recent_units_pct"], color=[STAGE_COLOR[s] for s in ss.index[::-1]], fmt=".1f"))
    with st.expander("How stages are assigned"):
        st.markdown(f"""
- **Dormant** – no sale in the last {cfg.DORMANT_DAYS} days.  **Launch** – first sale within the last {cfg.LAUNCH_DAYS} days.
- **Sporadic** – under {cfg.SPORADIC_UNITS_26W} units in 26 weeks; too little signal for a trend.
- **Growth / Decline** – the product's *share of its category* moved by ≥ {cfg.GROWTH_PCT_WK:g}% per week over the last {cfg.TREND_WEEKS} weeks
  (and the slope is statistically distinguishable from noise), or it now sells under 40% of its peak share. Using share keeps a
  category-wide seasonal dip from marking every product as declining.  **Mature** – everything else.
- Products already on sale when the data begins have an unknown true age (`launch_date_known = False`).
- ⚠️ *Stock-out suspected* marks steady sellers with a ≥{cfg.STOCKOUT_GAP_DAYS}-day zero-sales gap that later resumed. Until FBA inventory is
  loaded, a "Decline" on such a product may be supply, not demand. Thresholds live in `demand/config.py`.""")

    pick_stage = st.pills("Stage", A.STAGES, default="Growth") or "Growth"
    cols = ["asin", "title", "category", "stage", "units_last_4w", "units_total", "growth_vs_category_pct_wk", "growth_pct_wk", "trend_t_stat",
            "recent_vs_peak_share", "age_days", "launch_date_known", "days_since_sale", "stockout_suspected", "longest_gap_days"]
    rows = lc[lc["stage"] == pick_stage].sort_values("units_total" if pick_stage == "Dormant" else "units_last_4w", ascending=False)
    table(rows[cols], hide_index=True, height=360, column_config={
        "title": st.column_config.TextColumn(width="large"), "growth_vs_category_pct_wk": st.column_config.NumberColumn("vs category %/wk", format="%+.1f"),
        "growth_pct_wk": st.column_config.NumberColumn("units %/wk", format="%+.1f")})

    if not rows.empty:
        pick = st.selectbox("Sales curve", rows["asin"].head(300), format_func=lambda a: f"{a} · {rows.set_index('asin').at[a, 'title'][:90]}")
        w = A.weekly_matrix(D_ALL[D_ALL["asin"] == pick].assign(k="Units"), "k").T.reindex(A.complete_weeks(D_ALL), fill_value=0)
        show(lines(w, {"Units": STAGE_COLOR[pick_stage]}, 260, ytitle="Units / week"))
    key_points(INS.lifecycle(lc))
    ai_brief({"stage summary": ss, "top growth products": lc[lc["stage"] == "Growth"].nlargest(15, "units_last_4w")[cols],
              "largest declining products": lc[lc["stage"] == "Decline"].nlargest(15, "units_total")[cols],
              "recent launches": lc[lc["stage"] == "Launch"].nlargest(10, "units_last_4w")[cols]},
             "Assess the health of the product portfolio by lifecycle stage and recommend actions per stage.", "lifecycle")

elif page == "Future years (ML)":
    st.title("Prediction for the coming months and years")
    if WEATHER.empty:
        st.warning("No weather loaded. Run `python -m demand.weather` once (needs internet), then reload.")
        st.stop()
    c = st.columns([2, 4, 2])
    until = c[0].selectbox("Predict until", ["2027-12-31", "2028-12-31"], index=1, format_func=lambda v: f"Dec {v[:4]}")
    growth = c[1].slider("Assumed underlying growth, % per year", -30, 50, 0, 5,
                         help="Year-on-year growth cannot be measured until a month repeats (September 2026 will be the first), so it is your input, "
                              "not the model's. One hint from the data: ordinary days in Sep 2025 and in Aug 2026 both ran at about 1,100 units. "
                              "0 = the business repeats its 2025-26 level under normal weather and the festival calendar.")
    weather_note = ML.load_normals(WEATHER)[1]
    pred = ml_predict(_mtime(), until, "revert", float(growth))
    if f_cats:
        pred = pred[pred["category"].isin(f_cats)]
    if f_regions:
        pred = pred[pred["region"].isin(f_regions)]
    pred = pred[pred["region"] != "Unknown"]
    hist = D_ALL[(D_ALL["region"] != "Unknown") & D_ALL["category"].isin(pred["category"].unique()) & D_ALL["region"].isin(pred["region"].unique())]
    hist_m = hist.groupby(["region", "month"])["quantity"].sum().unstack("region")
    pred_m = pred.groupby(["region", "month"])["units"].sum().unstack("region")
    regions = hist_m.sum().sort_values(ascending=False).index.tolist()
    REGION_COLOR = {r: P["series"][i] for i, r in enumerate(sorted(D_ALL["region"].unique()))}

    st.subheader("Units per month, all selected regions")
    fig = go.Figure()
    fig.add_scatter(x=hist_m.index, y=hist_m.sum(axis=1), name="Actual", mode="lines+markers", line=dict(color=P["ink2"], width=2), marker=dict(size=6))
    fig.add_scatter(x=pred_m.index, y=pred_m.sum(axis=1), name="Predicted", mode="lines+markers", line=dict(color=P["series"][0], width=2), marker=dict(size=6))
    show(mark_forecast_start(style(fig, 360, ytitle="Units / month"), pred_m.index[0]))

    st.subheader("By region")
    fig = go.Figure()
    for r in regions:
        fig.add_scatter(x=hist_m.index, y=hist_m[r], name=r, legendgroup=r, mode="lines", line=dict(color=REGION_COLOR[r], width=2))
        fig.add_scatter(x=pred_m.index, y=pred_m[r], name=f"{r} (predicted)", legendgroup=r, showlegend=False, mode="lines",
                        line=dict(color=REGION_COLOR[r], width=2, dash="dot"))
    show(mark_forecast_start(style(fig, 420, ytitle="Units / month"), pred_m.index[0]))
    st.caption("Solid = actual, dotted = predicted. The model works from weather and the festival calendar, not from 'same month last year': "
               "September 2025 held the opening of the festive sale, September 2026 will not (Diwali moves from 20 Oct to 8 Nov), "
               "so the two Septembers are predicted to differ.")

    yearly = pred.assign(period=pred["month"].str[:4]).groupby(["region", "period"]).agg(units=("units", "sum"), revenue=("revenue", "sum"))
    months_in = pred.groupby(pred["month"].str[:4])["month"].nunique()
    yu = yearly["units"].unstack("period").round(0)
    yu.columns = [f"{p} ({months_in[p]} mo)" for p in yu.columns]
    yu.loc["ALL"] = yu.sum()
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Predicted units by year")
        table(yu.style.format("{:,.0f}"))
        st.caption(f"For scale: the {N_MONTHS} months of history total {hist['quantity'].sum():,.0f} units for this selection.")
    with c2:
        st.subheader("Seasonal shape by region (first full year)")
        full = [p for p, n in months_in.items() if n == 12]
        if full:
            fy = pred[pred["month"].str[:4] == full[0]].groupby(["region", "month"])["units"].sum().unstack("month")
            fy.columns = [pd.Period(m).strftime("%b") for m in fy.columns]
            show(heat((fy.div(fy.mean(axis=1), axis=0) * 100).loc[[r for r in regions if r in fy.index]], diverging_mid=100, height=280))
            st.caption("Index 100 = that region's average month. Shows which months each region runs hot or cold.")

    st.subheader("Region x category detail")
    detail = pred.pivot_table(index=["region", "category"], columns="month", values="units", aggfunc="sum").round(0)
    table(detail, height=320)
    st.download_button("Download prediction (CSV)", pred.to_csv(index=False).encode(), "homemonde_prediction_by_region_month.csv", "text/csv")

    st.subheader("What the model has learned")
    dr = ml_drivers(_mtime())
    t1, t2, t3 = st.tabs(["Temperature", "Rain", "Around Diwali"])
    with t1:
        show(heat(dr["temperature"], diverging_mid=100))
        st.caption("Demand index of each category as the weekly mean daily-max temperature changes, other things typical (100 = the category's average over this range).")
    with t2:
        show(heat(dr["rain"], diverging_mid=100))
        st.caption("Index versus a dry week.")
    with t3:
        show(heat(dr["diwali"], diverging_mid=100))
        st.caption("Index versus an ordinary week, from the single festive season in the data (Diwali 20 Oct 2025): the lift starts four weeks before Diwali week, "
                   "when the Amazon festive sale opened on 22 Sep, peaks one to two weeks before Diwali, and demand dips right after. Future Diwali dates come from `events.csv`: 8 Nov 2026, 29 Oct 2027, 17 Oct 2028.")
    st.warning("One year cannot separate 'sells in the heat' from 'was launched or advertised in summer'. Mattress Protector and Sofa/Chair Cover took off "
               "in May-July 2026, so the model reads them as strongly heat- and rain-driven. That may be true (waterproof protectors in the monsoon) or an "
               "artefact of timing; believe it only after a second summer. The curtain, bedsheet, comforter and duvet-cover responses agree with the "
               "separate state-versus-state regression on the Season & climate page and are better founded.", icon="⚠️")

    with st.expander("How accurate is it? (validation)"):
        if st.toggle("Run the validation", key="run_mlval"):
            blocked, tail = ml_validation(_mtime())
            st.markdown("**Test 1: 4-week blocks held out across the year** (conditions inside the range the model has seen)")
            table(blocked)
            st.markdown(f"**Test 2: the last 12 weeks held out** ({tail['holdout']}): the model must predict a monsoon it has never seen")
            table(tail["wape"])
            st.markdown("Figures are WAPE: total absolute error as a % of actual units (lower is better). Read together: the model explains part of the "
                        "seasonal movement (test 1) but adds nothing over a plain average when pushed into unseen conditions (test 2). "
                        "Treat months beyond the first half-year as a reasoned scenario, not a measured forecast.")

    st.subheader("Written outlook")
    st.markdown(ML.outlook_text(pred, hist, growth, weather_note))
    ai_brief({"predicted units by region and year": yu, "predicted units by region and month": pred_m.round(0),
              "predicted units by category and year": pred.assign(y=pred["month"].str[:4]).pivot_table(index="category", columns="y", values="units", aggfunc="sum").round(0),
              "actual units by region and month (history)": hist_m, "learned temperature response (index)": dr["temperature"],
              "learned Diwali response (index)": dr["diwali"], "lifecycle stage summary": A.stage_summary(lifecycle(_mtime()))},
             f"Write the theoretical demand outlook for the coming years, region by region. Assumptions in force: underlying growth {growth:+d}% "
             f"a year, future weather = {weather_note} Festive uplift as seen in the two weeks before Diwali 2025. Explain the "
             "reasoning behind each region's seasonal shape (climate and festival timing, including Diwali moving from November 2026 to October "
             "in 2027 and 2028), what it means for production and inventory timing, and be explicit about which statements are well supported "
             f"and which are theoretical given only {N_MONTHS} months of history (no month has repeated yet).", "outlook")

elif page == "AI suggestions":
    st.title("AI suggestions")
    from demand import ai, plays
    st.caption(f"Each plan is written by Claude from a curated set of evidence tables (shown under the plan), in a fixed format so plans can be "
               f"compared week to week. Plans are saved, so they are still here when you come back. Runs through: {_ai_label()}.")
    lc_all = lifecycle(_mtime())

    def show_plan(key: str, label: str, build, button: str):
        """One suggestion block: saved plan if there is one, a button to (re)generate, the evidence underneath."""
        saved = ai.load_note(key)
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"#### {label}" + (f"  \n<span class='card-note'>generated {saved[1]}</span>" if saved else ""), unsafe_allow_html=True)
        go_now = c2.button(("↻ Regenerate" if saved else button), key=f"go_{key}", type="secondary" if saved else "primary", width="stretch")
        if go_now:
            tables, goal = build()
            try:
                with st.status("Claude is building the plan (usually 40-90 seconds) …", expanded=False) as status:
                    text = ai.plan(tables, goal, key)
                    status.update(label="Plan ready", state="complete")
                st.rerun()                               # redraw from the saved note so the timestamp and buttons are current
            except Exception as e:
                st.error(ai.friendly_error(e))
        if saved:
            with st.container(border=True):
                st.markdown(saved[0])
            cc = st.columns([1, 1, 4])
            cc[0].download_button("Download (.md)", saved[0].encode("utf-8"), f"{key}.md", "text/markdown", key=f"dl_{key}")
            with st.expander("Evidence Claude was given"):
                for name, df in build()[0].items():
                    st.markdown(f"**{name}**")
                    table(df, height=min(300, 38 * (len(df) + 1) + 4))
        else:
            st.info("No plan yet. Click the button to have Claude write one.")

    t1, t2, t3, t4, t5, t6 = st.tabs(["This week's priorities", "Festive season plan", "Plan for a place", "Check a product", "Fabric purchase plan",
                                      "Outsourced stock plan"])
    with t1:
        show_plan("weekly_priorities", "What to act on this week", lambda: plays.weekly_priorities(D_ALL, lc_all, PRODUCTS), "✨ Write this week's priorities")
    with t2:
        show_plan("festive_plan", "Festive stock plan (Diwali 8 Nov 2026)", lambda: plays.festive_plan(D_ALL, lc_all, PRODUCTS), "✨ Write the festive plan")
    with t3:
        c = st.columns([1, 2, 2])
        lv = c[0].selectbox("Level", ["State", "Region", "District"], key="pl_level")
        by_units = lambda col, frame: [x for x in frame.groupby(col)["quantity"].sum().sort_values(ascending=False).index if x.upper() != "UNKNOWN"]
        if lv == "District":
            stp = c[1].selectbox("State", by_units("state", D_ALL), key="pl_state")
            plc = c[2].selectbox("District", by_units("district", D_ALL[D_ALL["state"] == stp]), key="pl_dist")
        else:
            plc = c[1].selectbox(lv, by_units(lv.lower(), D_ALL), key="pl_place")
        show_plan(f"place_{lv.lower()}_{plc}", f"Plan for {plc.title() if lv != 'District' else plc}",
                  lambda: plays.place_plan(D_ALL, lc_all, PRODUCTS, ml_predict_state(_mtime()), lv.lower(), plc), "✨ Write the plan for this place")
    with t4:
        top_skus = D_ALL.groupby("main_sku")["quantity"].sum().sort_values(ascending=False).head(600).index
        sku = st.selectbox("SKU (aliases merged)", top_skus, format_func=lambda k: f"{k} · {str(SKU_TITLE.get(k, ''))[:80]}", key="pc_sku")
        show_plan(f"product_{sku}", f"Verdict on {sku}", lambda: plays.product_check(D_ALL, lc_all, PRODUCTS, sku), "✨ Check this product")
    with t5:
        if cfg.FABRIC_PARQUET.exists():
            show_plan("fabric_plan", "Fabric to buy for the next 12 weeks", lambda: plays.fabric_plan(D_ALL, lc_all, PRODUCTS), "✨ Write the fabric purchase plan")
        else:
            st.info("Load the fabric sheet first (see the Data page).")
    with t6:
        if cfg.OUTSOURCE_PARQUET.exists():
            show_plan("outsource_plan", "Outsourced range: what to reorder, run down and clear",
                      lambda: plays.outsource_plan(D_ALL, lc_all, PRODUCTS), "✨ Write the outsourced stock plan")
        else:
            st.info("Load the outsource sheet first (see the Data page).")

    st.divider()
    key_points(INS.actions(D_ALL, lc_all, forecasts(_mtime(), FILT, "category", 12)), "Rule-based action list (no AI, always current)")

elif page == "AI analyst":
    st.title("Ask the analyst")
    from demand import ai
    st.caption(f"Ask in plain language. Claude queries your cleaned data (sales, products, lifecycle, weather, forecast, prediction) and shows "
               f"every query it ran. Runs through: {_ai_label()}.")
    if "analyst" not in st.session_state or st.session_state.get("analyst_mtime") != _mtime():
        with st.spinner("Preparing the analyst's database …"):
            st.session_state.analyst, st.session_state.analyst_mtime, st.session_state.chat = ai.Analyst(), _mtime(), []
    chat = st.session_state.chat

    view = [f"period {start} to {end}"] + ([f"categories {', '.join(f_cats)}"] if f_cats else []) + ([f"regions {', '.join(f_regions)}"] if f_regions else []) \
        + ([f"states {', '.join(f_states)}"] if f_states else []) + ([f"districts {', '.join(f_districts)}"] if f_districts else [])
    filtered = bool(f_cats or f_regions or f_states or f_districts or (start, end) != (lo, hi))
    use_view = st.toggle("Scope questions to my sidebar filters", value=filtered, disabled=not filtered,
                         help="When on, Claude is told which period / category / place you are looking at and answers within it.")

    def render(turn):
        text, followups = ai.split_followups(turn["text"])
        st.markdown(text)
        if turn.get("queries"):
            with st.expander(f"How this was worked out · {len(turn['queries'])} quer{'y' if len(turn['queries']) == 1 else 'ies'}"):
                for q in turn["queries"]:
                    st.code(q, language="sql")
                frame = ai.query_frame(turn["queries"][-1])
                if frame is not None and len(frame):
                    st.markdown("**Rows behind the last query**")
                    table(frame, hide_index=True, height=min(320, 38 * (len(frame) + 1) + 4))
        return followups

    if not chat:
        st.markdown("##### Start with one of these, or type your own question below")
        cols = st.columns(3)
        for i, (label, q) in enumerate(ai.STARTERS.items()):
            if cols[i % 3].button(label, key=f"starter_{i}", width="stretch"):
                st.session_state.pending_q = q
                st.rerun()

    followups = []
    for i, turn in enumerate(chat):
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.markdown(turn["text"])
            else:
                followups = render(turn)

    typed = st.chat_input("Ask about products, places, months, weather, lifecycle, forecast …")
    question = typed or st.session_state.pop("pending_q", None)
    if question:
        chat.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            queries = []
            with st.status("Thinking …", expanded=True) as status:
                def on_query(sql):
                    queries.append(sql)
                    status.update(label=f"Running query {len(queries)} …")
                    st.code(sql, language="sql")
                try:
                    asked = (f"[User's current view: {'; '.join(view)}]\n" if use_view and filtered else "") + question
                    answer = st.session_state.analyst.ask(asked, on_query)
                except Exception as e:
                    answer = "⚠️ " + ai.friendly_error(e)
                status.update(label=f"Answered · {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}", state="complete", expanded=False)
            turn = {"role": "assistant", "text": answer, "queries": queries}
            chat.append(turn)
        st.rerun()                                    # redraw from history so follow-up buttons appear under the new answer

    if followups:
        st.markdown("<div class='card-note'>Suggested next questions</div>", unsafe_allow_html=True)
        for i, q in enumerate(followups):                         # one per row so the whole question is readable
            if st.button("↳ " + q, key=f"fu_{len(chat)}_{i}"):
                st.session_state.pending_q = q
                st.rerun()
    if chat:
        c = st.columns([1, 1, 5])
        transcript = "\n\n".join(("**You:** " if t["role"] == "user" else "**Analyst:** ") + ai.split_followups(t["text"])[0] for t in chat)
        c[0].download_button("Download chat", transcript.encode("utf-8"), "analyst_chat.md", "text/markdown")
        if c[1].button("New conversation"):
            st.session_state.pop("analyst")
            st.rerun()

elif page == "Fabric consumption":
    st.title("Fabric consumption")
    if not cfg.FABRIC_PARQUET.exists():
        st.warning("No fabric sheet loaded. Put `Fabric Consumption*.xlsx` in the project folder and run `python -m demand.ingest`.")
        st.stop()
    pf = fabric_map(_mtime(), _fab_mtime())
    dm = FB.with_metres(D, pf)
    cov = FB.coverage(D, pf)
    known = dm.dropna(subset=["metres"])
    if known.empty:
        st.info("The products in this view consume no in-house fabric (outsourced, or not on the fabric sheet). Change the filters to see fabric consumption.")
        st.stop()
    total_m = known["metres"].sum()
    inferred_m = known.loc[known["fabric_source"] == "inferred: size rule", "metres"].sum()
    k = st.columns(5)
    k[0].metric("Fabric consumed", f"{total_m:,.0f} m")
    k[1].metric("From the sheet", f"{total_m - inferred_m:,.0f} m")
    k[2].metric("By size rule (code unknown)", f"{inferred_m:,.0f} m")
    k[3].metric("Units with a fabric figure", f"{cov.loc['ALL', 'known_%']:.1f}%")
    k[4].metric("Fabric codes in use", f"{known.loc[~known['fabric_code'].isin([FB.UNKNOWN, '(none)']), 'fabric_code'].nunique()}")
    st.caption("Metres = units sold x the sheet's metres per unit (metres per piece x pieces per SKU). Products the sheet does not list get metres "
               "from its own size rule (curtain metres depend only on length; cushion, pillow and bolster covers on cover size) and are shown "
               "separately as **code unknown**. Outsourced products (see *Outsourced stock*) are bought in finished and consume no in-house fabric.")
    key_points(INS.fabric(dm, fabric_requirement(_mtime(), _fab_mtime(), FILT, 12), cov))

    FAB_TABS = ["By fabric code", "By type & family", "Over time", "Requirement (next 12 weeks)", "Long range (ML)", "By place", "Coverage & sheet",
                "SKUs without fabric data"]
    _tab = st.query_params.get("tab", "")                        # ?page=Fabric consumption&tab=SKUs without fabric data opens that tab
    tabs = st.tabs(FAB_TABS, default=_tab if _tab in FAB_TABS else None)
    with tabs[0]:
        bf = FB.by_fabric(dm)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("Top 15 fabric codes, metres")
            show(hbar(bf["metres"].head(15)))
        with c2:
            st.subheader("All fabric codes")
            table(bf.reset_index(), hide_index=True, height=520, column_config={
                "metres": st.column_config.NumberColumn("metres", format="localized"), "growth_4w_%": st.column_config.NumberColumn("4 wk growth %", format="%+.1f")})
        st.subheader("Look up a fabric code")
        code = st.selectbox("Fabric code", bf.index.tolist(), key="fab_code")
        x = known[known["fabric_code"] == code]
        c1, c2 = st.columns([3, 2])
        with c1:
            w = A.weekly_matrix(x.assign(quantity=x["metres"], k="m"), "k").T.reindex(A.complete_weeks(D), fill_value=0)
            show(lines(w.rename(columns={"m": "metres / week"}), {"metres / week": P["series"][0]}, 260, ytitle="Metres / week"))
        with c2:
            prods = x.groupby(["main_sku"]).agg(units=("quantity", "sum"), metres=("metres", "sum")).sort_values("metres", ascending=False).round(0)
            prods["title"] = prods.index.map(SKU_TITLE).astype(str).str[:60]
            table(prods.reset_index(), hide_index=True, height=260)
        st.caption(f"{code}: {x['fabric_type'].iloc[0]} · {x['fabric_family'].iloc[0]} · colours: {', '.join(sorted(x['colour'].dropna().unique()))} · "
                   f"{x['asin'].nunique()} products")
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Metres by fabric type")
            show(hbar(FB.by_fabric(dm, "fabric_type")["metres"]))
        with c2:
            st.subheader("Metres by fabric family")
            show(hbar(FB.by_fabric(dm, "fabric_family")["metres"]))
        st.subheader("Metres by product type")
        table(FB.by_fabric(dm, "product_type").reset_index(), hide_index=True, height=380,
              column_config={"growth_4w_%": st.column_config.NumberColumn("4 wk growth %", format="%+.1f")})
    with tabs[2]:
        dim = st.segmented_control("Series", ["fabric_type", "fabric_family", "fabric_code"], default="fabric_type", format_func=pretty, key="fab_dim")
        mm = FB.monthly(dm, dim, top=None)                       # every row of the series, not only the largest
        _known_k = known if dim != "fabric_code" else known[~known["fabric_code"].isin([FB.UNKNOWN, "(none)"])]
        wkm = A.weekly_matrix(_known_k.assign(quantity=_known_k["metres"]), dim).T
        months = [pd.Period(c).strftime("%b %y") for c in mm.columns]
        mm.columns = months
        if dim != "fabric_code":
            top = wkm.sum().sort_values(ascending=False).index[:8]
            show(lines(wkm[top], {c: P["series"][i % 8] for i, c in enumerate(top)}, 380, ytitle="Metres / week"))
            st.subheader("Metres by month")
            mm.insert(0, "total_metres", mm.sum(axis=1))
            table(mm.reset_index(), hide_index=True, column_config={c: st.column_config.NumberColumn(c, format="localized") for c in months}
                  | {"total_metres": st.column_config.NumberColumn("Total metres", format="localized")})
        else:
            # every fabric code on the sheet, with its details, including codes nothing in this view consumed
            sheet = FB.load_fabric()
            first = lambda s: s.dropna().mode().iat[0] if s.notna().any() else None
            info = sheet.groupby("fabric_code").agg(fabric_type=("fabric_type", first), fabric_family=("fabric_family", first),
                                                    colours=("colour", lambda s: ", ".join(sorted(s.dropna().unique()))), skus_on_sheet=("sku", "size"))
            allc = info.join(bf[["units", "products"]]).join(mm, how="left")
            allc[["units", "products"] + months] = allc[["units", "products"] + months].fillna(0)
            allc.insert(4, "total_metres", allc[months].sum(axis=1))
            f = st.columns([2, 2, 2, 2, 2])
            p_type = f[0].multiselect("Fabric type", sorted(allc["fabric_type"].dropna().unique()), placeholder="All types", key="fot_type")
            p_fam = f[1].multiselect("Fabric family", sorted(allc["fabric_family"].dropna().unique()), placeholder="All families", key="fot_fam")
            find = f[2].text_input("Search code or colour", placeholder="e.g. FAB-0030 or Beige", key="fot_find").strip().upper()
            order = f[3].selectbox("Order by", ["Total metres (high to low)", "Fabric code (A to Z)", f"Latest month ({months[-1]})", "Fabric type, then metres"], key="fot_order")
            f[4].markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)      # line the toggle up with the inputs beside it
            used_only = f[4].toggle("Consumed only", value=False, key="fot_used", help="Hide fabric codes with no consumption in this view.")
            view = allc
            if p_type:
                view = view[view["fabric_type"].isin(p_type)]
            if p_fam:
                view = view[view["fabric_family"].isin(p_fam)]
            if find:
                view = view[view.index.str.upper().str.contains(find, regex=False) | view["colours"].str.upper().str.contains(find, regex=False)]
            if used_only:
                view = view[view["total_metres"] > 0]
            view = (view.sort_index() if order.startswith("Fabric code") else view.sort_values(months[-1], ascending=False) if order.startswith("Latest")
                    else view.sort_values(["fabric_type", "total_metres"], ascending=[True, False]) if order.startswith("Fabric type")
                    else view.sort_values("total_metres", ascending=False))
            top = [c for c in view.index if c in wkm.columns and view.at[c, "total_metres"] > 0][:8] if not order.startswith("Total") else \
                  [c for c in view.index if c in wkm.columns][:8]
            if top:
                with card("Metres per week", f"the first {len(top)} fabric codes of the table below (a chart of all {len(view)} would be unreadable)"):
                    show(lines(wkm[top], {c: P["series"][i % 8] for i, c in enumerate(top)}, 360, ytitle="Metres / week"))
            st.subheader(f"Metres by month, all fabric codes ({len(view)} of {len(allc)})")
            st.markdown(f"<div class='card-note'>{view['total_metres'].sum():,.0f} m in this view · {int((view['total_metres'] > 0).sum())} codes consumed · "
                        f"{int((view['total_metres'] == 0).sum())} on the sheet with no consumption here</div>", unsafe_allow_html=True)
            table(view.reset_index(), hide_index=True, height=min(720, 35 * (len(view) + 1) + 4), column_config={
                "fabric_code": st.column_config.TextColumn("Fabric code", pinned=True), "colours": st.column_config.TextColumn("Colours", width="medium"),
                "skus_on_sheet": st.column_config.NumberColumn("SKUs on sheet"), "units": st.column_config.NumberColumn("Units sold", format="localized"),
                "products": st.column_config.NumberColumn("Products sold"), "total_metres": st.column_config.NumberColumn("Total metres", format="localized")}
                | {c: st.column_config.NumberColumn(c, format="localized") for c in months})
            st.download_button("Download this table (CSV)", view.round(0).to_csv().encode(), "fabric_codes_metres_by_month.csv", "text/csv")
            st.caption("One row per fabric code on the fabric sheet, with its type, family, colours and metres consumed each month. The sidebar filters "
                       "(period, category, place, sourcing) change the metres; the filters above choose which codes are listed.")
    with tabs[3]:
        horizon = st.slider("Weeks ahead", 4, 16, 12, key="fab_h")
        req = fabric_requirement(_mtime(), _fab_mtime(), FILT, horizon)
        if req.empty:
            st.info("Need at least 16 complete weeks of history to forecast.")
        else:
            coded = req[req.index != FB.UNKNOWN]
            c = st.columns(3)
            c[0].metric(f"Baseline, next {horizon} wks (coded)", f"{coded['baseline_m'].sum():,.0f} m")
            c[1].metric("Festive scenario (coded)", f"{coded['festive_m'].sum():,.0f} m")
            c[2].metric("Code unknown (new ranges)", f"{req.loc[FB.UNKNOWN, 'festive_m'] if FB.UNKNOWN in req.index else 0:,.0f} m",
                        help="Products whose metres come from the size rule because they are not in the fabric sheet yet.")
            st.subheader("Metres needed per fabric code")
            table(req.reset_index().rename(columns={"key": "fabric_code"}), hide_index=True, height=480,
                  column_config={"baseline_m": st.column_config.NumberColumn("baseline m", format="localized"),
                                 "festive_m": st.column_config.NumberColumn("festive m", format="localized"),
                                 f"last_{horizon}w_actual_m": st.column_config.NumberColumn(f"last {horizon} wk actual m", format="localized")})
            st.download_button("Download requirement (CSV)", req.to_csv().encode(), "fabric_requirement.csv", "text/csv")
            st.caption(f"Product-level weekly forecast (baseline and the Diwali-aligned festive scenario) x metres per unit. With {cfg.PRODUCTION_WEEKS} weeks "
                       "of production, fabric for the festive weeks must already be in hand; treat the festive column as the upside case.")
    with tabs[4]:
        st.subheader("Metres per month by fabric code, next 12 months (ML prediction)")
        pred = ml_predict(_mtime(), "2027-12-31", "revert", 0.0)
        lr = FB.monthly_requirement(pred, D, pf)
        lr.columns = ["fabric type"] + [pd.Period(c).strftime("%b %y") for c in lr.columns[1:]]
        table(lr.head(25))
        st.caption("Category-level ML prediction (climate normals + festival calendar, 0% growth) split across products by their last-12-week share, "
                   "then x metres per unit. The product mix inside each category is held at today's; the month-to-month shape comes from the model.")
    with tabs[5]:
        lvl = st.segmented_control("Level", ["region", "state", "district"], default="region", format_func=pretty, key="fab_place")
        g = known[known[lvl].str.upper() != "UNKNOWN"].groupby(lvl)["metres"].sum().sort_values(ascending=False)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader(f"Metres by {lvl}")
            show(hbar(g.head(15)))
        with c2:
            st.subheader(f"Fabric type mix by {lvl} (metres)")
            piv = known[known[lvl].isin(g.head(10).index)].pivot_table(index=lvl, columns="fabric_type", values="metres", aggfunc="sum", fill_value=0)
            show(heat((piv.div(piv.sum(axis=1), axis=0) * 100).round(0).loc[g.head(10).index.intersection(piv.index)], diverging_mid=None, height=330))
    with tabs[6]:
        st.subheader("Units with a fabric figure, by category")
        table(cov.reset_index())
        miss = pf[pf["fabric_source"].isin(["not in sheet", "inferred: size rule"])].copy()
        miss["units_year"] = miss["asin"].map(D_ALL.groupby("asin")["quantity"].sum()).fillna(0)
        miss = miss[miss["units_year"] > 0].sort_values("units_year", ascending=False)
        st.subheader(f"Products not in the sheet ({len(miss):,}) - add these to the sheet to sharpen the numbers")
        table(miss[["main_sku", "asin", "category", "fabric_source", "size", "pieces", "metres_per_unit", "units_year", "title"]].head(300), hide_index=True, height=380,
              column_config={"title": st.column_config.TextColumn(width="large")})
        st.download_button("Download the full list (CSV)", miss.to_csv(index=False).encode(), "products_missing_from_fabric_sheet.csv", "text/csv")
        st.caption(f"Sheet loaded: {cfg.FABRIC_PARQUET.stat().st_mtime and pd.Timestamp(cfg.FABRIC_PARQUET.stat().st_mtime, unit='s'):%d %b %Y %H:%M} · "
                   f"{len(FB.load_fabric()):,} SKUs · {FB.load_fabric()['fabric_code'].nunique()} fabric codes. Update the workbook and run `python -m demand.ingest` to refresh.")
    with tabs[7]:
        ms = missing_skus(_mtime())
        if f_cats:
            ms = ms[ms["category"].isin(f_cats)]
        k = st.columns(4)
        k[0].metric("SKUs without fabric data", f"{len(ms):,}", help="Every SKU name; a listing sold under two SKUs counts twice.")
        k[1].metric("Listings (ASINs)", f"{ms['asin'].nunique():,}")
        k[2].metric("Units sold in the year", f"{ms['units_year'].sum():,.0f}", help=f"{ms['units_year'].sum() / max(D_ALL['quantity'].sum(), 1) * 100:.1f}% of all units")
        k[3].metric("Selling in the last 8 weeks", f"{int((ms['units_8w'] > 0).sum()):,}")
        c1, c2, c3 = st.columns([5, 6, 6])
        with c1:
            with card("By category", "SKUs and units in the year"):
                bc = ms.groupby("category").agg(skus=("sku", "size"), units_year=("units_year", "sum")).sort_values("units_year", ascending=False)
                table(bc.reset_index(), hide_index=True, height=248, column_config={"skus": st.column_config.NumberColumn("SKUs"),
                                                                                    "units_year": st.column_config.NumberColumn("Units", format="localized")})
        with c2:
            with card("By priority", "1 = 100+ units in the last 8 weeks, 2 = 20-99, 3 = 1-19"):
                bp = ms.groupby(["priority", "fabric_data"]).size().unstack(fill_value=0).reindex(FB.PRIORITY, fill_value=0)
                bp = bp.rename(columns={"Estimated (no fabric code)": "Estimated"})
                bp["SKUs"] = bp.sum(axis=1)
                bp["Units (8 wk)"] = ms.groupby("priority")["units_8w"].sum().reindex(FB.PRIORITY, fill_value=0)
                table(bp.reset_index(), hide_index=True, height=248)
        with c3:
            with card("Largest ranges", "units in the year, top 8 design ranges"):
                br = ms.groupby("sku_range")["units_year"].sum().sort_values(ascending=False).head(8)
                show(hbar(br, height=236))
        f = st.columns([2, 2, 2, 3])
        pick_cat = f[0].multiselect("Category", bc.index.tolist(), placeholder="All categories", key="ms_cat")
        pick_pri = f[1].multiselect("Priority", FB.PRIORITY, placeholder="All priorities", key="ms_pri")
        pick_kind = f[2].multiselect("Fabric data", sorted(ms["fabric_data"].unique()), placeholder="Estimated and none", key="ms_kind")
        find = f[3].text_input("Search SKU, range or ASIN", placeholder="e.g. HM162-173 or B0GG8RZDJ6", key="ms_find").strip().upper()
        view = ms
        if pick_cat:
            view = view[view["category"].isin(pick_cat)]
        if pick_pri:
            view = view[view["priority"].isin(pick_pri)]
        if pick_kind:
            view = view[view["fabric_data"].isin(pick_kind)]
        if find:
            view = view[view["sku"].str.upper().str.contains(find, regex=False) | view["asin"].str.upper().str.contains(find, regex=False)]
        st.markdown(f"<div class='card-note'>Showing {len(view):,} of {len(ms):,} SKUs · {view['units_year'].sum():,.0f} units in the year · "
                    f"{view['units_8w'].sum():,.0f} in the last 8 weeks</div>", unsafe_allow_html=True)
        table(view[["sku", "category", "fabric_data", "priority", "units_year", "units_8w", "estimated_size", "estimated_metres_per_unit", "asin", "title"]],
              hide_index=True, height=520, row_height=28, column_config={
                  "sku": st.column_config.TextColumn("SKU", pinned=True), "fabric_data": st.column_config.TextColumn("Fabric data"),
                  "units_year": st.column_config.NumberColumn("Units (year)", format="localized"), "units_8w": st.column_config.NumberColumn("Units (8 wk)", format="localized"),
                  "estimated_size": st.column_config.TextColumn("Est. size"), "estimated_metres_per_unit": st.column_config.NumberColumn("Est. m / unit", format="%.2f"),
                  "asin": st.column_config.TextColumn("ASIN"), "title": st.column_config.TextColumn("Title", width="large")})
        b = st.columns([2, 2, 6])
        b[0].download_button("Download this view (CSV)", view.to_csv(index=False).encode(), "skus_without_fabric_data.csv", "text/csv", width="stretch")
        b[1].download_button("Download SKU names (TXT)", "\n".join(view["sku"]).encode(), "skus_without_fabric_data.txt", "text/plain", width="stretch")
        st.caption("**Estimated** = the tool works out metres from the sheet's size rule (curtain length, cover size) but has no fabric code. "
                   "**No data** = no fabric figure at all. Outsourced products are left out: they are bought in finished and need no fabric data. "
                   "Add these SKUs to the fabric sheet and run `python -m demand.ingest`; they then drop off this list.")
    ai_brief({"12-week fabric requirement": fabric_requirement(_mtime(), _fab_mtime(), FILT, 12).head(25), "top fabric codes": FB.by_fabric(dm).head(20),
              "metres by fabric type": FB.by_fabric(dm, "fabric_type"), "coverage": cov},
             "Explain the fabric consumption picture: which fabrics matter most, what is rising or falling, what to buy for the next 12 weeks, "
             "and how much of the picture is missing because products are not in the sheet.", "fabric")

elif page == "Outsourced stock":
    st.title("Outsourced stock")
    if not cfg.OUTSOURCE_PARQUET.exists():
        st.warning("No outsource sheet loaded. Put `Outsource*.xlsx` in the project folder and run `python -m demand.ingest`.")
        st.stop()
    om, ps = sourcing_tables(_mtime())
    sc = stock_cover(_mtime())
    if f_cats:
        sc = sc[sc["category"].isin(f_cats)]
    if sc.empty:
        st.warning("No outsourced SKUs in the selected categories.")
        st.stop()
    snap = pd.Timestamp(sc["snapshot"].iloc[0])
    share = SRC.sourcing_summary(D, ps)
    STATUS_COLOR = dict(zip(SRC.STATUS_ORDER, [P["series"][7], P["series"][1], P["series"][3], P["series"][2], P["series"][0], P["series"][6], OTHER]))
    at_risk = sc[sc["status"].isin(["Out of stock", "Critical", "Reorder now"])]
    slow = sc[sc["status"].isin(["Overstock", "Idle stock"])]

    k = st.columns(6)
    k[0].metric("SKUs on the sheet", f"{len(sc):,}")
    k[1].metric("Ready stock", f"{sc['ready'].sum():,.0f}")
    k[2].metric("Virtual quantity", f"{sc['virtual'].sum():,.0f}")
    k[3].metric("SKUs at risk", f"{len(at_risk):,}", help="Selling SKUs that are out of stock, under 2 weeks of cover, or run out within the lead time + 2 weeks.")
    k[4].metric("Demand, 12 wks", f"{sc['fc_12w_festive'].sum():,.0f}", help=f"Festive scenario; baseline {sc['fc_12w_baseline'].sum():,.0f} units.")
    k[5].metric("Units to reorder", f"{sc['reorder_festive'].sum():,.0f}", help=f"Festive scenario; baseline {sc['reorder_baseline'].sum():,.0f} units.")
    st.caption(f"Stock snapshot of **{snap:%d %b %Y}** against demand. **Ready** = finished stock in hand; **virtual** = quantity kept available to sell "
               f"(it can exceed ready when the maker holds or can supply more); cover is measured on the larger of the two. Sales data ends "
               f"{D_ALL['date'].max():%d %b %Y}, so demand from the snapshot on is the product-level forecast. Stock already at Amazon fulfilment "
               f"centres is not on the sheet. Lead time assumed: {cfg.OUTSOURCE_LEAD_WEEKS} weeks; reorder covers lead time + {cfg.OUTSOURCE_REVIEW_WEEKS} weeks. "
               + ("Category filter applied. " if f_cats else "") + "Place and period filters do not apply to stock.")
    key_points(INS.outsource(sc, share, cfg.OUTSOURCE_LEAD_WEEKS))

    SHOW = ["sku", "category", "status", "ready", "virtual", "available", "units_8w", "weekly_rate", "fc_12w_baseline", "fc_12w_festive", "cover_weeks",
            "stockout_date", "reorder_baseline", "reorder_festive", "demand_basis", "stage", "title"]
    SHOW = [c for c in SHOW if c in sc.columns]
    STOCK_CFG = {"cover_weeks": st.column_config.NumberColumn("Cover (weeks)", format="%.1f", help="Weeks until available stock runs out along the festive forecast"),
                 "stockout_date": st.column_config.DateColumn("Runs out", format="DD MMM YYYY"),
                 "fc_12w_baseline": st.column_config.NumberColumn("Forecast 12 wk"), "fc_12w_festive": st.column_config.NumberColumn("Forecast 12 wk (festive)"),
                 "units_8w": st.column_config.NumberColumn("Sold last 8 wk"), "weekly_rate": st.column_config.NumberColumn("Forecast / week", format="%.1f"),
                 "reorder_baseline": st.column_config.NumberColumn("Reorder"), "reorder_festive": st.column_config.NumberColumn("Reorder (festive)"),
                 "title": st.column_config.TextColumn("Title", width="large"), "last_sale": st.column_config.DateColumn("Last sale", format="DD MMM YYYY")}

    def stock_table(frame: pd.DataFrame, cols=SHOW, height=460):
        x = frame[cols].copy()
        if "status" in x:
            x["status"] = x["status"].astype(str)
        if "title" in x:
            x["title"] = x["title"].astype(str).str[:70].replace("nan", "")
        table(x, hide_index=True, height=height, column_config=STOCK_CFG)

    tabs = st.tabs(["Stock health", "Reorder list", "Overstock & idle", "Ready vs virtual", "In-house vs outsourced", "Mapping & sheet"])
    with tabs[0]:
        ss = SRC.status_summary(sc)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("SKUs by stock status")
            fig = hbar(ss["skus"], height=300)
            fig.update_traces(marker_color=[STATUS_COLOR[x] for x in ss.index[::-1]])
            show(fig)
        with c2:
            st.subheader("Available stock against the next 12 weeks' demand, by category")
            bc = SRC.by_category(sc)
            bc = bc[(bc["available"] > 0) | (bc["fc_12w_festive"] > 0)].iloc[::-1]
            fig = go.Figure()
            fig.add_bar(y=bc.index, x=bc["ready"], name="Ready stock", orientation="h", marker=dict(color=P["series"][0], cornerradius=3),
                        hovertemplate="%{x:,.0f}")
            fig.add_bar(y=bc.index, x=bc["virtual"], name="Virtual quantity", orientation="h", marker=dict(color=P["seq"][1], cornerradius=3),
                        hovertemplate="%{x:,.0f}")
            fig.add_bar(y=bc.index, x=bc["fc_12w_festive"], name="Demand, next 12 weeks (festive)", orientation="h",
                        marker=dict(color=P["series"][1], cornerradius=3), hovertemplate="%{x:,.0f}")
            fig = style(fig, max(300, 58 * len(bc) + 70), xtitle="Units")
            fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08, hovermode="y unified")
            fig.update_xaxes(showgrid=True, gridcolor=P["grid"], tickformat="~s", showline=False)
            fig.update_yaxes(showgrid=False, tickformat=None, ticksuffix="  ", rangemode="normal", tickfont=dict(color=P["ink2"], size=12))
            show(fig)
        st.subheader("What each status holds")
        table(ss.reset_index().astype({"status": str}), hide_index=True,
              column_config={"fc_12w_festive": st.column_config.NumberColumn("Forecast 12 wk (festive)"), "units_8w": st.column_config.NumberColumn("Sold last 8 wk")})
        st.subheader("By category")
        table(SRC.by_category(sc).reset_index(), hide_index=True,
              column_config={"cover_weeks": st.column_config.NumberColumn("Cover (weeks)", format="%.1f"),
                             "fc_12w_baseline": st.column_config.NumberColumn("Forecast 12 wk"), "fc_12w_festive": st.column_config.NumberColumn("Forecast 12 wk (festive)"),
                             "units_8w": st.column_config.NumberColumn("Sold last 8 wk"), "reorder_festive": st.column_config.NumberColumn("Reorder (festive)")})
    with tabs[1]:
        st.subheader(f"Selling SKUs short of cover ({len(at_risk)})")
        if at_risk.empty:
            st.success("Every selling SKU is covered beyond the lead time.")
        else:
            c = st.columns(3)
            c[0].metric("Demand at risk, next 12 wks", f"{at_risk['fc_12w_festive'].sum():,.0f} units")
            c[1].metric("Available for it", f"{at_risk['available'].sum():,.0f} units")
            c[2].metric("To reorder (festive / baseline)", f"{at_risk['reorder_festive'].sum():,.0f} / {at_risk['reorder_baseline'].sum():,.0f}")
            top = at_risk.sort_values("fc_12w_festive", ascending=False).head(15).set_index("sku").iloc[::-1]
            fig = go.Figure()
            fig.add_bar(y=top.index, x=top["available"], name="Available now", orientation="h", marker=dict(color=P["series"][0], cornerradius=3),
                        hovertemplate="%{x:,.0f}")
            fig.add_bar(y=top.index, x=top["fc_12w_festive"], name="Demand, next 12 weeks (festive)", orientation="h",
                        marker=dict(color=P["series"][1], cornerradius=3), hovertemplate="%{x:,.0f}")
            fig = style(fig, 30 * len(top) * 2 // 2 + 110, xtitle="Units")
            fig.update_layout(barmode="group", bargap=0.3, hovermode="y unified")
            fig.update_xaxes(showgrid=True, gridcolor=P["grid"], showline=False)
            fig.update_yaxes(showgrid=False, tickformat=None, ticksuffix="  ", rangemode="normal", tickfont=dict(color=P["ink2"], size=11))
            with card("Largest exposures", "15 at-risk SKUs with the most forecast demand"):
                show(fig)
            stock_table(at_risk.sort_values(["status", "fc_12w_festive"], ascending=[True, False]))
            st.download_button("Download reorder list (CSV)", at_risk[SHOW].to_csv(index=False).encode(), "outsource_reorder_list.csv", "text/csv")
            st.caption(f"Reorder = forecast demand over the lead time ({cfg.OUTSOURCE_LEAD_WEEKS} weeks) plus {cfg.OUTSOURCE_REVIEW_WEEKS} weeks, minus what is "
                       "available. Per-product forecasts are rough (about 70% weekly error in backtests), so read the quantities as order-of-magnitude "
                       "and the ranking as the message. Where sales stopped before the data ends while the SKU was still selling, the forecast reads "
                       "that as no demand, so the rate of its last selling weeks is used instead (see *Demand basis*). An 'Out of stock' SKU may still "
                       "have units at Amazon.")
    with tabs[2]:
        st.subheader(f"Stock that is not moving ({len(slow)} SKUs, {slow['available'].sum():,.0f} units)")
        if slow.empty:
            st.success("No overstocked or idle SKUs.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                with card("Overstock by category", "units available in SKUs with more than 26 weeks of cover"):
                    g = sc[sc["status"] == "Overstock"].groupby("category")["available"].sum().sort_values(ascending=False)
                    if len(g):
                        show(hbar(g, color=P["series"][0]))
                    else:
                        st.caption("None.")
            with c2:
                with card("Idle stock by category", "units available in SKUs with no sales in the last 8 weeks"):
                    g = sc[sc["status"] == "Idle stock"].groupby("category")["available"].sum().sort_values(ascending=False)
                    if len(g):
                        show(hbar(g, color=P["series"][6]))
                    else:
                        st.caption("None.")
            stock_table(slow.sort_values("available", ascending=False), [c for c in SHOW if not c.startswith("reorder") and c != "stockout_date"] + ["last_sale"])
            st.caption("Overstock is judged against the forecast, which for a slow product is small and uncertain: a SKU that sells 2 a week with 80 in "
                       "hand shows 40 weeks of cover. Idle SKUs with a listing are candidates to reprice, bundle or relist; those with no sales at all "
                       "in the data may simply not be listed on Amazon.")
    with tabs[3]:
        rel = np.select([(sc["ready"] == 0) & (sc["virtual"] == 0), sc["ready"] == sc["virtual"], (sc["ready"] == 0) & (sc["virtual"] > 0),
                         (sc["virtual"] == 0) & (sc["ready"] > 0), sc["virtual"] > sc["ready"]],
                        ["Both zero", "Virtual = ready", "Virtual only (nothing in hand)", "Ready only (virtual is 0)", "Virtual above ready"], "Virtual below ready")
        rt = sc.assign(relation=rel).groupby("relation").agg(skus=("sku", "size"), ready=("ready", "sum"), virtual=("virtual", "sum"),
                                                             units_8w=("units_8w", "sum"), fc_12w_festive=("fc_12w_festive", "sum")).sort_values("skus", ascending=False)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("How the two columns relate")
            show(hbar(rt["skus"], height=300))
        with c2:
            st.subheader("Stock and demand in each group")
            table(rt.reset_index(), hide_index=True, column_config={"fc_12w_festive": st.column_config.NumberColumn("Forecast 12 wk (festive)"),
                                                                    "units_8w": st.column_config.NumberColumn("Sold last 8 wk")})
        dep = sc[(sc["virtual"] > sc["ready"]) & (sc["fc_12w_festive"] > sc["ready"])].copy()
        dep["demand_beyond_ready"] = (dep[["fc_12w_festive", "virtual"]].min(axis=1) - dep["ready"]).clip(lower=0).round(0)
        st.subheader(f"Demand that depends on the maker ({len(dep)} SKUs, {dep['demand_beyond_ready'].sum():,.0f} units)")
        stock_table(dep.sort_values("demand_beyond_ready", ascending=False),
                    ["sku", "category", "ready", "virtual", "demand_beyond_ready", "fc_12w_festive", "cover_weeks_ready_only", "cover_weeks", "status", "title"], 380)
        unl = sc[(sc["virtual"] == 0) & (sc["ready"] > 0)]
        st.subheader(f"Stock in hand that is not offered ({len(unl)} SKUs, {unl['ready'].sum():,.0f} units)")
        stock_table(unl.sort_values("ready", ascending=False), ["sku", "category", "ready", "virtual", "units_8w", "units_year", "last_sale", "match", "title"], 320)
        st.caption("Reading used here: ready is physical finished stock, virtual is the quantity the business keeps available to sell. If virtual is "
                   "instead *extra* stock on top of ready, set `OUTSOURCE_AVAILABLE=sum` in `.env` and cover will be measured on ready + virtual.")
    with tabs[4]:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Units per week by sourcing")
            wk = A.weekly_matrix(D, "sourcing").T
            order = [x for x in (SRC.IN_HOUSE, SRC.OUTSOURCED, SRC.UNCLASSIFIED, SRC.BOTH) if x in wk.columns]
            show(lines(wk[order], dict(zip(order, [P["series"][0], P["series"][1], OTHER, P["series"][3]])), 340, ytitle="Units / week"))
        with c2:
            st.subheader("Share of units outsourced, by category")
            sh = share.drop(index="ALL")
            sh = sh.loc[sh["outsourced_%"] >= 0.5, "outsourced_%"].sort_values(ascending=False) if "outsourced_%" in sh else pd.Series(dtype=float)
            if len(sh):
                show(hbar(sh, color=P["series"][1], fmt=".0f", height=340, xtitle="% of the category's units"))
                st.caption("Categories not shown are made in-house.")
            else:
                st.caption("No outsourced sales in this view.")
        st.subheader("Units by category and sourcing")
        table(share.reset_index().rename(columns={"category": "category"}), hide_index=True,
              column_config={"outsourced_%": st.column_config.NumberColumn("Outsourced %", format="%.1f")})
        mo = D.pivot_table(index="sourcing", columns="month", values="quantity", aggfunc="sum", fill_value=0)
        st.subheader("Units by month")
        mo.columns = [pd.Period(c).strftime("%b %y") for c in mo.columns]
        table(mo.style.format("{:,.0f}"))
        st.caption("Use the **Sourcing** filter in the sidebar to see any other page (geography, season, lifecycle, forecast) for outsourced or "
                   "in-house products only. This tab follows all sidebar filters.")
    with tabs[5]:
        mt = om["match"].value_counts().rename("skus").to_frame()
        mt["ready"], mt["virtual"] = om.groupby("match")["ready"].sum(), om.groupby("match")["virtual"].sum()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("How sheet SKUs were tied to listings")
            table(mt.reset_index(), hide_index=True)
        with c2:
            st.subheader("How products were classified")
            pc = ps.assign(units=ps["asin"].map(D_ALL.groupby("asin")["quantity"].sum()).fillna(0))
            pc["basis"] = pc["sourcing_basis"].str.replace(r" \(SKU prefix .*\)", "", regex=True)
            table(pc.groupby(["sourcing", "basis"]).agg(products=("asin", "size"), units_year=("units", "sum")).reset_index(), hide_index=True)
        loose = om[om["match"] == "same SKU, different punctuation"]
        if len(loose):
            st.subheader(f"Matched after ignoring punctuation ({len(loose)}) - worth aligning the sheet's spelling with Seller Central")
            table(loose[["sku", "sold_as", "asin", "ready", "virtual"]].rename(columns={"sku": "sku_in_sheet", "sold_as": "sku_on_amazon"}), hide_index=True, height=260)
        nos = om[om["asin"].isna()]
        st.subheader(f"Sheet SKUs with no sales in the data ({len(nos)}, {SRC.available(nos).sum():,.0f} units available)")
        table(nos[["sku", "prefix", "ready", "virtual"]].sort_values("virtual", ascending=False), hide_index=True, height=260)
        both = ps[ps["sourcing"] == SRC.BOTH]
        if len(both):
            st.subheader(f"On both sheets ({len(both)}) - made in-house or bought in?")
            table(both[["main_sku", "asin", "category"]], hide_index=True)
        unc = ps[ps["sourcing"] == SRC.UNCLASSIFIED].assign(units_year=lambda t: t["asin"].map(D_ALL.groupby("asin")["quantity"].sum()).fillna(0))
        unc = unc.groupby("prefix").agg(products=("asin", "size"), units_year=("units_year", "sum"), category=("category", lambda x: x.mode().iloc[0]),
                                        example=("main_sku", "first")).sort_values("units_year", ascending=False)
        st.subheader(f"Ranges on neither sheet ({int(unc['products'].sum())} products, {unc['units_year'].sum():,.0f} units in the year)")
        table(unc.reset_index().rename(columns={"prefix": "sku_range"}), hide_index=True, height=260)
        ch = SRC.stock_changes(SRC.load_history())
        if len(ch):
            st.subheader(f"Stock change since the previous snapshot ({pd.Timestamp(ch.attrs['from']):%d %b} to {pd.Timestamp(ch.attrs['to']):%d %b %Y})")
            table(ch.reset_index(), hide_index=True, height=300)
        st.download_button("Download the full stock table (CSV)", sc.astype({"status": str}).to_csv(index=False).encode(), "outsource_stock.csv", "text/csv")
        st.caption(f"Snapshots kept: {SRC.load_history()['snapshot'].nunique()}. Each time the workbook is updated and `python -m demand.ingest` is run, the new "
                   "stock is added to the history, so stock movement and sell-through build up over time.")
    ai_brief({"stock by status": SRC.status_summary(sc), "stock by category": SRC.by_category(sc),
              "at-risk SKUs (top 25 by forecast)": at_risk.sort_values("fc_12w_festive", ascending=False)[SHOW[:-1]].head(25).astype({"status": str}),
              "overstock and idle (top 20 by stock)": slow.sort_values("available", ascending=False)[SHOW[:-1]].head(20).astype({"status": str}),
              "units by category and sourcing": share},
             "Explain the outsourced stock position: what will run out and when, what is over-bought, what to reorder first, and how much of "
             "the conclusion rests on the forecast and on the reading of 'virtual quantity'.", "outsource")


elif page == "Data":
    st.title("Data")
    d_all = ORDERS
    c = st.columns(4)
    c[0].metric("Order lines stored", f"{len(d_all):,}")
    c[1].metric("Demand lines", f"{int(d_all['is_demand'].sum()):,}")
    c[2].metric("Cancelled lines", f"{int(d_all['is_cancelled'].sum()):,}")
    c[3].metric("Non-Amazon (MCF) lines", f"{int((~d_all['is_amazon']).sum()):,}")
    st.markdown(f"**Claude features run through:** {_ai_label()}  \n"
                "**Anthropic API key:** " + ("found (used only if `AI_BACKEND=api` is set, or on a computer without Claude Code)"
                                             if cfg.ANTHROPIC_API_KEY else "not set (only needed for `AI_BACKEND=api`)"))
    st.markdown("**Fabric sheet:** " + (f"loaded ({len(FB.load_fabric()):,} SKUs, {FB.load_fabric()['fabric_code'].nunique()} fabric codes)" if cfg.FABRIC_PARQUET.exists()
                                        else "not loaded. Put `Fabric Consumption*.xlsx` in the project folder and run `python -m demand.ingest`."))
    if cfg.OUTSOURCE_PARQUET.exists():
        _o = SRC.load_outsource()
        st.markdown(f"**Outsource sheet:** loaded ({len(_o):,} SKUs, ready {_o['ready'].sum():,.0f}, virtual {_o['virtual'].sum():,.0f}, "
                    f"stock as of {pd.Timestamp(_o['snapshot'].iloc[0]):%d %b %Y}; {SRC.load_history()['snapshot'].nunique()} snapshot(s) kept)")
    else:
        st.markdown("**Outsource sheet:** not loaded. Put `Outsource*.xlsx` in the project folder and run `python -m demand.ingest`.")
    st.subheader("Reconciliation with the Excel workbook")
    rec = d_all.assign(month=d_all["date"].dt.strftime("%Y-%m")).groupby("month").apply(lambda g: pd.Series({
        "rows in sheet": len(g), "units in sheet (sum of quantity)": g["quantity"].sum(),
        "- non-Amazon / MCF units": g.loc[~g["is_amazon"], "quantity"].sum(),
        "- cancelled rows": int(g["is_cancelled"].sum()),
        "= demand units used": g.loc[g["is_demand"], "quantity"].sum(),
        "demand revenue ₹": g["revenue"].sum()}), include_groups=False)
    rec.loc["TOTAL"] = rec.sum()
    table(rec.style.format("{:,.0f}"))
    st.caption("'Units in sheet' is what a SUM of the quantity column gives in Excel for that month's sheet. The tool's demand figure is lower only by "
               "the non-Amazon (MCF) rows; cancelled rows already carry quantity 0. Verified row-for-row against the workbook: 0 difference in every month.")

    st.subheader("Add a new report")
    st.markdown("Download **Reports → Fulfilment / Orders → All Orders** from Seller Central for the new month and upload it here "
                f"(or drop it into `{cfg.RAW_DIR}` and run `python -m demand.ingest`). Re-uploading a month is safe: newer order statuses replace older ones.")
    up = st.file_uploader("All Orders report", type=["xlsx", "txt", "tsv", "csv"],
                          help="On the hosted copy (Streamlit Community Cloud) an upload lasts only until the app next restarts. "
                               "To update it for good, ingest locally, commit `data/store/` and push.")
    if up and st.button("Ingest file"):
        from demand.ingest import ingest
        from demand.weather import update_weather
        cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
        (cfg.RAW_DIR / up.name).write_bytes(up.getbuffer())
        with st.status("Ingesting …", expanded=True):
            ingest(log=st.write)
            update_weather(log=st.write)
        st.cache_data.clear()
        st.rerun()
    st.subheader("Lines per month and source file")
    m = d_all.assign(month=d_all["date"].dt.strftime("%Y-%m")).pivot_table(index="month", columns="source_file", values="quantity", aggfunc="size", fill_value=0)
    table(m)
    st.subheader("Weather coverage")
    if WEATHER.empty:
        st.warning("No weather loaded. Run `python -m demand.weather`.")
    else:
        st.write(f"{WEATHER['state'].nunique()} states · {WEATHER['date'].min():%d %b %Y} – {WEATHER['date'].max():%d %b %Y} · source: Open-Meteo archive")
