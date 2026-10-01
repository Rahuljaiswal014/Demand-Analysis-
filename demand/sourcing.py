"""Sourcing and outsourced stock: where each product comes from, and whether the bought-in ones have enough stock.

Two sheets define sourcing, and they barely overlap:
  * "Fabric Consumption*.xlsx"  - products HomeMonde manufactures (one row per SKU with its fabric)      -> In-house
  * "Outsource*.xlsx"           - finished goods bought from outside makers, with their stock            -> Outsourced
A product in neither sheet is classified by its range: if every classified product sharing its SKU prefix has the same
sourcing, it inherits it (flagged `by range`); otherwise it stays Unclassified.

The outsource sheet is a point-in-time stock snapshot (Item sku, Ready stock, Virtual quantities). Every ingest appends the
snapshot to a history file, so stock trends and sell-through between snapshots build up as the sheet is updated.

  ready    = finished stock physically in hand
  virtual  = quantity available to sell as maintained by the business (it can exceed ready when the maker holds or can
             supply more, and is often 0 while ready stock exists)
  available = how the two combine, set by config.OUTSOURCE_AVAILABLE: "max" (default, virtual already includes ready),
              "sum" (virtual is extra to ready), "ready" or "virtual".
"""
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics as A
from . import config as cfg
from . import forecast as F

IN_HOUSE, OUTSOURCED, BOTH, UNCLASSIFIED = "In-house", "Outsourced", "Both sheets (check)", "Unclassified"
STATUS_ORDER = ["Out of stock", "Critical", "Reorder now", "Healthy", "Overstock", "Idle stock", "Inactive"]
STATUS_HELP = {
    "Out of stock": "selling, nothing available",
    "Critical": "under 2 weeks of cover",
    "Reorder now": "cover is shorter than the lead time plus 2 weeks",
    "Healthy": "covered beyond the lead time, up to 26 weeks",
    "Overstock": "more than 26 weeks of cover",
    "Idle stock": "stock in hand but no sales in the last 8 weeks",
    "Inactive": "no stock and no recent sales",
}


def _norm(sku) -> str:
    """SKU key that survives the differences between the sheets: case, '&' vs '_', a trailing '#', spaces."""
    return re.sub(r"[^A-Z0-9]", "", str(sku).upper())


def _prefix(sku) -> str:
    m = re.match(r"[A-Za-z]+", str(sku))
    return m.group(0).upper() if m else ""


# ------------------------------------------------------------------ the sheet
def find_sheet() -> Path | None:
    for folder in (cfg.ROOT, cfg.RAW_DIR):
        hits = sorted(folder.glob("Outsource*.xls*"))
        if hits:
            return hits[-1]
    return None


def read_sheet(path: Path) -> pd.DataFrame:
    from .ingest import _copy_locked
    try:
        raw = pd.read_excel(path, dtype=str)
    except PermissionError:                                   # open in Excel: read a copy of the saved file
        with tempfile.TemporaryDirectory() as tmp:
            raw = pd.read_excel(_copy_locked(path, Path(tmp)), dtype=str)
    raw.columns = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in raw.columns]
    want = {"item sku": "sku", "ready stock": "ready", "virtual quantities": "virtual"}
    missing = [c for c in want if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: columns not found: {missing} (have {list(raw.columns)})")
    o = raw[list(want)].rename(columns=want).dropna(how="all")
    o["sku"] = o["sku"].astype(str).str.strip()
    o = o[~o["sku"].isin(["", "nan", "None"])]
    for c in ("ready", "virtual"):
        o[c] = pd.to_numeric(o[c], errors="coerce").fillna(0).clip(lower=0)
    # the same SKU listed twice is summed rather than dropped
    o = o.groupby(o["sku"].str.upper(), as_index=False).agg(sku=("sku", "first"), ready=("ready", "sum"), virtual=("virtual", "sum"))
    return o[["sku", "ready", "virtual"]]


def update_outsource(log=print) -> pd.DataFrame | None:
    """Rebuild outsource.parquet when the sheet is new or changed, and append the snapshot to the history."""
    path = find_sheet()
    if path is None:
        return None
    sig = {"size": path.stat().st_size, "mtime": int(path.stat().st_mtime), "name": path.name}
    if cfg.OUTSOURCE_PARQUET.exists() and cfg.OUTSOURCE_MANIFEST.exists() and json.loads(cfg.OUTSOURCE_MANIFEST.read_text()) == sig:
        return pd.read_parquet(cfg.OUTSOURCE_PARQUET)
    log(f"reading {path.name} ...")
    o = read_sheet(path)
    o["snapshot"] = pd.Timestamp(path.stat().st_mtime, unit="s").normalize()
    o.to_parquet(cfg.OUTSOURCE_PARQUET, index=False)
    hist = pd.read_parquet(cfg.OUTSOURCE_HISTORY) if cfg.OUTSOURCE_HISTORY.exists() else pd.DataFrame(columns=o.columns)
    hist = pd.concat([hist[hist["snapshot"] != o["snapshot"].iloc[0]], o], ignore_index=True)
    hist.to_parquet(cfg.OUTSOURCE_HISTORY, index=False)
    cfg.OUTSOURCE_MANIFEST.write_text(json.dumps(sig))
    log(f"outsource sheet: {len(o):,} SKUs, ready {int(o['ready'].sum()):,}, virtual {int(o['virtual'].sum()):,}, "
        f"snapshot {o['snapshot'].iloc[0]:%d %b %Y} ({hist['snapshot'].nunique()} snapshot(s) in history)")
    return o


def load_outsource() -> pd.DataFrame:
    return pd.read_parquet(cfg.OUTSOURCE_PARQUET) if cfg.OUTSOURCE_PARQUET.exists() else pd.DataFrame(columns=["sku", "ready", "virtual", "snapshot"])


def load_history() -> pd.DataFrame:
    return pd.read_parquet(cfg.OUTSOURCE_HISTORY) if cfg.OUTSOURCE_HISTORY.exists() else pd.DataFrame(columns=["sku", "ready", "virtual", "snapshot"])


def available(o: pd.DataFrame) -> pd.Series:
    mode = cfg.OUTSOURCE_AVAILABLE
    if mode == "sum":
        return o["ready"] + o["virtual"]
    if mode in ("ready", "virtual"):
        return o[mode]
    return o[["ready", "virtual"]].max(axis=1)


# ------------------------------------------------------------------ mapping to products
def match(out: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Attach the ASIN each outsource SKU sells under. Exact SKU first, then the punctuation-blind key; where several
    listings share a key, the one with the most units wins."""
    o = out.copy()
    s = orders.assign(key=orders["sku"].str.upper())
    units = s.groupby(["key", "asin"])["quantity"].sum().reset_index().sort_values("quantity", ascending=False)
    exact = units.drop_duplicates("key").set_index("key")["asin"]
    units["nkey"] = units["key"].map(_norm)
    loose = units.drop_duplicates("nkey").set_index("nkey")
    o["asin"] = o["sku"].str.upper().map(exact)
    o["match"] = np.where(o["asin"].notna(), "exact SKU", None)
    todo = o["asin"].isna()
    nk = o.loc[todo, "sku"].map(_norm)
    o.loc[todo, "asin"] = nk.map(loose["asin"])
    o.loc[todo, "sold_as"] = nk.map(loose["key"])
    o.loc[todo & o["asin"].notna(), "match"] = "same SKU, different punctuation"
    o["match"] = o["match"].fillna("no sales in the data")
    o["prefix"] = o["sku"].map(_prefix)
    return o


def product_sourcing(products: pd.DataFrame, fab: pd.DataFrame, out_matched: pd.DataFrame) -> pd.DataFrame:
    """One row per ASIN: sourcing and what it rests on."""
    p = products[["asin", "main_sku", "skus", "category"]].copy()
    keys = (p["main_sku"].astype(str) + ", " + p["skus"].astype(str)).str.upper().str.split(", ")
    fab_asin, fab_sku = set(fab["asin"].dropna()) if len(fab) else set(), set(fab["sku"]) if len(fab) else set()
    out_asin = set(out_matched["asin"].dropna())
    out_key = set(out_matched["sku"].map(_norm))
    in_fab = p["asin"].isin(fab_asin) | keys.map(lambda ks: bool(set(ks) & fab_sku))
    in_out = p["asin"].isin(out_asin) | keys.map(lambda ks: any(_norm(k) in out_key for k in ks))
    p["sourcing"] = np.select([in_fab & in_out, in_fab, in_out], [BOTH, IN_HOUSE, OUTSOURCED], UNCLASSIFIED)
    p["sourcing_basis"] = np.select([in_fab & in_out, in_fab, in_out], ["both sheets", "fabric sheet", "outsource sheet"], "")
    # by range: a prefix whose classified products all share one sourcing passes it to its unclassified members
    p["prefix"] = p["main_sku"].map(_prefix)
    known = p[p["sourcing"].isin([IN_HOUSE, OUTSOURCED])]
    g = known.groupby("prefix")["sourcing"].agg(["nunique", "first", "size"])
    rule = g[(g["nunique"] == 1) & (g["size"] >= 5)]["first"]
    # outsource-only prefixes that never sold still define a range
    extra = pd.Series(OUTSOURCED, index=sorted(set(out_matched["prefix"]) - set(g.index)))
    rule = pd.concat([rule, extra[~extra.index.isin(rule.index)]])
    todo = p["sourcing"] == UNCLASSIFIED
    inherited = p.loc[todo, "prefix"].map(rule)
    p.loc[todo & inherited.notna(), "sourcing"] = inherited.dropna()
    p.loc[todo & inherited.notna(), "sourcing_basis"] = "by range (SKU prefix " + p.loc[todo & inherited.notna(), "prefix"] + ")"
    p.loc[p["sourcing"] == UNCLASSIFIED, "sourcing_basis"] = "in neither sheet, range is mixed or new"
    return p.drop(columns=["skus"])


def sourcing_summary(d: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    """Units, revenue and products per sourcing, by category. `d` may already carry a `sourcing` column."""
    x = d if "sourcing" in d.columns else d.merge(ps[["asin", "sourcing"]], on="asin", how="left")
    t = x.pivot_table(index="category", columns="sourcing", values="quantity", aggfunc="sum", fill_value=0)
    t = t.reindex(columns=[c for c in (IN_HOUSE, OUTSOURCED, BOTH, UNCLASSIFIED) if c in t.columns])
    t["units"] = t.sum(axis=1)
    t = t.sort_values("units", ascending=False)
    t.loc["ALL"] = t.sum()
    if OUTSOURCED in t:
        t["outsourced_%"] = (t[OUTSOURCED] / t["units"] * 100).round(1)
    return t


# ------------------------------------------------------------------ stock against demand
def stock_cover(d: pd.DataFrame, out_matched: pd.DataFrame, products: pd.DataFrame, lifecycle: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per outsourced SKU: stock, recent sales, forecast from the snapshot date, weeks of cover, status, reorder.

    Demand is the product-level weekly forecast (baseline and festive scenario) for the weeks from the stock snapshot on.
    Cover walks that forecast until the available stock is used up, so a festive peak shortens it as it should."""
    o = out_matched.copy()
    o["available"] = available(o)
    snap = pd.Timestamp(o["snapshot"].iloc[0]) if "snapshot" in o and len(o) else d["date"].max()
    end = d["date"].max()
    gap_weeks = max(0, int(np.ceil((snap - end).days / 7)))
    horizon = gap_weeks + 16
    fc = F.forecast(d, "asin", horizon)
    fc = fc[fc["week"] >= A.week_start(pd.Series([snap])).iloc[0]] if not fc.empty else fc
    weeks = sorted(fc["week"].unique())[:16] if not fc.empty else []
    base = fc.pivot_table(index="group", columns="week", values="baseline", aggfunc="sum").reindex(columns=weeks).fillna(0) if weeks else pd.DataFrame()
    fest = fc.pivot_table(index="group", columns="week", values="festive_scenario", aggfunc="sum").reindex(columns=weeks).fillna(0) if weeks else pd.DataFrame()

    recent = d[d["date"] > end - pd.Timedelta(weeks=8)].groupby("asin")["quantity"].sum()
    last4 = d[d["date"] > end - pd.Timedelta(weeks=4)].groupby("asin")["quantity"].sum()
    year = d.groupby("asin")["quantity"].sum()
    last_sale = d.groupby("asin")["date"].max()
    o["units_year"] = o["asin"].map(year).fillna(0)
    o["units_8w"] = o["asin"].map(recent).fillna(0)
    o["units_4w"] = o["asin"].map(last4).fillna(0)
    o["last_sale"] = o["asin"].map(last_sale)

    def cover(row_fc: np.ndarray, stock: float) -> float:
        """Weeks until `stock` is used up along the weekly forecast; beyond the horizon the last 4 weeks' average carries on."""
        row_fc = np.asarray(row_fc, dtype=float)
        if stock <= 0:
            return 0.0
        cum = np.cumsum(row_fc)
        if len(cum) and cum[-1] >= stock:
            i = int(np.searchsorted(cum, stock))
            prev = cum[i - 1] if i else 0.0
            return i + (stock - prev) / max(row_fc[i], 1e-9)
        tail = row_fc[-4:].mean() if len(row_fc) else 0.0
        return np.inf if tail <= 0 else len(cum) + (stock - (cum[-1] if len(cum) else 0)) / tail

    # A product whose sales stopped before the data ends while it was still selling well has most likely run out, and the
    # forecast (which follows the last weeks) reads that as no demand. For those the rate of their last selling weeks is used.
    # The rate is the average over the 8 weeks up to the product's own last sale, quiet weeks included.
    own = d[d["asin"].isin(set(o["asin"].dropna()))]
    own = own[own["date"] > own["asin"].map(last_sale) - pd.Timedelta(weeks=8)]
    selling_rate = own.groupby("asin")["quantity"].sum() / 8
    lead, review = cfg.OUTSOURCE_LEAD_WEEKS, cfg.OUTSOURCE_REVIEW_WEEKS
    rows = []
    for r in o.itertuples():
        b = base.loc[r.asin].to_numpy() if len(base) and r.asin in base.index else np.zeros(len(weeks))
        f = fest.loc[r.asin].to_numpy() if len(fest) and r.asin in fest.index else np.zeros(len(weeks))
        basis = "forecast" if b.sum() > 0 else "none"
        rate = float(selling_rate.get(r.asin, 0.0))
        stopped = pd.notna(r.last_sale) and (end - r.last_sale).days >= 10
        if stopped and r.units_8w > 0 and rate >= 1 and b[:12].mean() < 0.5 * rate:
            n = max(len(weeks), 16)
            b, f, basis = np.full(n, rate), np.full(n, rate), "last selling rate (sales stopped: stock-out suspected)"
        span = min(len(b), lead + review)
        rows.append({"demand_basis": basis, "fc_12w_baseline": b[:12].sum(), "fc_12w_festive": f[:12].sum(), "weekly_rate": b[:12].mean() if len(b) else 0.0,
                     "cover_weeks": cover(f, r.available), "cover_weeks_ready_only": cover(f, r.ready),
                     "reorder_baseline": max(0.0, b[:span].sum() - r.available), "reorder_festive": max(0.0, f[:span].sum() - r.available)})
    o = pd.concat([o.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    selling = (o["units_8w"] > 0) | (o["fc_12w_baseline"] >= 1)
    o["status"] = np.select(
        [~selling & (o["available"] > 0), ~selling, o["available"] <= 0, o["cover_weeks"] < 2, o["cover_weeks"] < lead + 2, o["cover_weeks"] > 26],
        ["Idle stock", "Inactive", "Out of stock", "Critical", "Reorder now", "Overstock"], "Healthy")
    o["stockout_date"] = [snap + pd.Timedelta(weeks=float(w)) if np.isfinite(w) and w < 60 and s in ("Critical", "Reorder now", "Healthy") else pd.NaT
                          for w, s in zip(o["cover_weeks"], o["status"])]
    info = products.set_index("asin")[["main_sku", "title", "category"]]
    o = o.join(info, on="asin")
    o["category"] = o["category"].fillna(o["prefix"].map(_category_of_prefix(products)))
    if lifecycle is not None and len(lifecycle):
        o = o.join(lifecycle.drop_duplicates("asin").set_index("asin")[["stage"]], on="asin")
    for c in ("fc_12w_baseline", "fc_12w_festive", "reorder_baseline", "reorder_festive"):
        o[c] = o[c].round(0)
    o["weekly_rate"] = o["weekly_rate"].round(1)
    o["cover_weeks"] = o["cover_weeks"].replace(np.inf, np.nan).round(1)
    o["cover_weeks_ready_only"] = o["cover_weeks_ready_only"].replace(np.inf, np.nan).round(1)
    o["status"] = pd.Categorical(o["status"], STATUS_ORDER, ordered=True)
    return o.sort_values(["status", "fc_12w_festive"], ascending=[True, False]).reset_index(drop=True)


def _category_of_prefix(products: pd.DataFrame) -> pd.Series:
    p = products.assign(prefix=products["main_sku"].map(_prefix))
    return p.groupby("prefix")["category"].agg(lambda s: s.value_counts().index[0])


def status_summary(sc: pd.DataFrame) -> pd.DataFrame:
    g = sc.groupby("status", observed=True).agg(skus=("sku", "size"), ready=("ready", "sum"), virtual=("virtual", "sum"), available=("available", "sum"),
                                                units_8w=("units_8w", "sum"), fc_12w_festive=("fc_12w_festive", "sum"))
    g["what it means"] = [STATUS_HELP[s] for s in g.index]
    return g


def by_category(sc: pd.DataFrame) -> pd.DataFrame:
    g = sc.groupby("category").agg(skus=("sku", "size"), ready=("ready", "sum"), virtual=("virtual", "sum"), available=("available", "sum"),
                                   units_8w=("units_8w", "sum"), fc_12w_baseline=("fc_12w_baseline", "sum"), fc_12w_festive=("fc_12w_festive", "sum"),
                                   reorder_festive=("reorder_festive", "sum"))
    g["cover_weeks"] = (g["available"] / (g["fc_12w_festive"] / 12).replace(0, np.nan)).round(1)
    g["at_risk_skus"] = sc[sc["status"].isin(["Out of stock", "Critical", "Reorder now"])].groupby("category").size()
    return g.fillna({"at_risk_skus": 0}).sort_values("fc_12w_festive", ascending=False)


def stock_changes(hist: pd.DataFrame) -> pd.DataFrame:
    """Change in stock between the two most recent snapshots (empty until a second snapshot exists)."""
    snaps = sorted(hist["snapshot"].unique())
    if len(snaps) < 2:
        return pd.DataFrame()
    a = hist[hist["snapshot"] == snaps[-2]].set_index("sku")
    b = hist[hist["snapshot"] == snaps[-1]].set_index("sku")
    t = b[["ready", "virtual"]].join(a[["ready", "virtual"]], lsuffix="_now", rsuffix="_before", how="outer").fillna(0)
    t["ready_change"], t["virtual_change"] = t["ready_now"] - t["ready_before"], t["virtual_now"] - t["virtual_before"]
    t.attrs["from"], t.attrs["to"] = snaps[-2], snaps[-1]
    return t[(t["ready_change"] != 0) | (t["virtual_change"] != 0)].sort_values("ready_change")


# ------------------------------------------------------------------ one-call builders
def build(products: pd.DataFrame, orders: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(outsource sheet matched to ASINs, sourcing per product) from the stores on disk. Matching uses every order line,
    cancelled ones included, because a SKU that only ever had cancelled orders still identifies its listing."""
    from . import fabric as FB
    orders = A.load_orders() if orders is None else orders
    om = match(load_outsource(), orders)
    return om, product_sourcing(products, FB.load_fabric(), om)


def outsourced_asins(products: pd.DataFrame, orders: pd.DataFrame | None = None) -> set:
    if not cfg.OUTSOURCE_PARQUET.exists():
        return set()
    return set(build(products, orders)[1].query("sourcing == @OUTSOURCED")["asin"])
