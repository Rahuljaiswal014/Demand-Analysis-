"""Suggestion "plays": for each kind of advice the business asks for, the evidence tables Claude should reason over
and the goal it is given. Keeping the evidence curated (rather than letting the model wander through the database) is
what makes the suggestions specific, comparable from run to run, and checkable against the dashboard."""
import pandas as pd

from . import analytics as A
from . import config as cfg
from . import forecast as F
from . import insights as INS
from . import predict as ML
from . import fabric as FB
from . import sourcing as SRC

LC_COLS = ["main_sku", "title", "category", "stage", "units_last_4w", "units_total", "growth_vs_category_pct_wk", "recent_vs_peak_share",
           "stockout_suspected", "longest_gap_days", "age_days"]


def _short(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(title=df["title"].astype(str).str[:55]) if "title" in df else df


def festive_leaders(d: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    """SKUs that led the 2025 festive window (sale opening to the day before Diwali week), with their lift over their own
    quiet-season weekly rate. This is the only direct evidence of what sells in the festive sale."""
    diwali = [w for w in F._diwali_weeks() if d["date"].min() <= w <= d["date"].max()]
    if not diwali:
        return pd.DataFrame()
    start, end = diwali[-1] - pd.Timedelta(weeks=4), diwali[-1] - pd.Timedelta(days=1)
    quiet = d[(d["date"] >= diwali[-1] + pd.Timedelta(weeks=6)) & (d["date"] < diwali[-1] + pd.Timedelta(weeks=18))]
    fest = d[d["date"].between(start, end)]
    t = pd.DataFrame({"festive_units_4wk": fest.groupby("main_sku")["quantity"].sum(),
                      "quiet_units_per_4wk": (quiet.groupby("main_sku")["quantity"].sum() / 3).round(0)}).fillna(0)
    t["lift_x"] = (t["festive_units_4wk"] / t["quiet_units_per_4wk"].clip(lower=5)).round(1)
    t["category"] = d.drop_duplicates("main_sku").set_index("main_sku")["category"]
    t["still_selling"] = t.index.isin(d.loc[d["date"] > d["date"].max() - pd.Timedelta(days=28), "main_sku"].unique())
    return t.sort_values("festive_units_4wk", ascending=False).head(top)


def weekly_priorities(d, lc, products):
    movers = A.trend_table(d, "main_sku", min_units=120)
    return {
        "category direction, last 8 weeks": A.trend_table(d, "category"),
        "region summary with 4-week growth": A.summary(d[d["region"] != "Unknown"], "region")[["units", "unit_share_pct", "last_4w", "prev_4w", "growth_4w_pct"]],
        "fastest rising SKUs (8 weeks, statistically clear)": movers[movers["direction"] == "Rising"].head(12),
        "fastest falling SKUs (8 weeks, statistically clear)": movers[movers["direction"] == "Falling"].tail(12),
        "growth-stage products": _short(lc[lc["stage"] == "Growth"].nlargest(15, "units_last_4w")[LC_COLS]),
        "big sellers with a suspected stock-out": _short(lc[lc["stockout_suspected"]].nlargest(12, "units_total")[LC_COLS + ["gap_ended"]]),
        "next 12 weeks by category (baseline and festive scenario)": F.forecast(d, "category").groupby("group")[["baseline", "festive_scenario"]].sum().round(0),
    }, ("Set this week's priorities for the demand and inventory team. Separate what needs action now from what only needs watching. "
        "A falling SKU with a suspected stock-out is a supply question, not a demand one: say so.")


def festive_plan(d, lc, products):
    fc_cat, fc_reg = F.forecast(d, "category"), F.forecast(d, "region")
    weekly = fc_cat.groupby("week")[["baseline", "festive_scenario"]].sum()
    weekly["festive_multiplier"] = weekly["festive_scenario"] / weekly["baseline"]     # of the total; an average of category multipliers would not reconcile
    weekly = weekly.round({"baseline": 0, "festive_scenario": 0, "festive_multiplier": 2})
    return {
        "total by week, next 12 weeks (festive_multiplier > 1 marks the sale weeks)": weekly,
        "by category, 12-week totals": fc_cat.groupby("group")[["baseline", "festive_scenario"]].sum().round(0),
        "by region, 12-week totals": fc_reg[fc_reg["group"] != "Unknown"].groupby("group")[["baseline", "festive_scenario"]].sum().round(0),
        "SKUs that led the 2025 festive window (lift_x = festive 4 weeks vs their own quiet-season 4 weeks)": festive_leaders(d),
        "category index by month last year (100 = own average)": A.seasonal_profile(d),
        "products growing now (candidates the 2025 list cannot contain)": _short(lc[lc["stage"].isin(["Growth", "Launch"])].nlargest(12, "units_last_4w")[LC_COLS]),
    }, ("Build the festive-season stock plan. Diwali 2026 is on 8 Nov; in 2025 the Amazon festive sale opened four weeks before Diwali week "
        "and demand dipped for about three weeks after it. Say what to build, how much above a normal week, and the date by which stock must "
        f"be at Amazon (lead times: {cfg.PRODUCTION_WEEKS} weeks production, {cfg.INBOUND_WEEKS} weeks inbound). Flag SKUs on the 2025 list that "
        "are no longer selling, and newer products that deserve a festive allocation although they have no festive history.")


def place_plan(d, lc, products, pred_state, level: str, place: str):
    here = d[d[level] == place]
    pm = ML.place_monthly(d, pred_state, level, place)
    nat = d.groupby("category")["quantity"].sum() / d["quantity"].sum()
    loc = here.groupby("category")["quantity"].sum() / here["quantity"].sum()
    idx = (loc / nat * 100).round(0).rename("index_vs_national").to_frame().assign(units=here.groupby("category")["quantity"].sum())
    top = here.groupby("main_sku").agg(units=("quantity", "sum"), last_8w=("quantity", lambda s: s[here.loc[s.index, "date"] > d["date"].max() - pd.Timedelta(weeks=8)].sum()))
    top = top.nlargest(20, "units").join(products.drop_duplicates("main_sku").set_index("main_sku")[["title", "category"]])
    lift = A.rank_cells(here.assign(_p=place), "main_sku", "season", None, 4, max(40, int(here["quantity"].sum() * 0.002)), "lift")
    sub = {"district": None, "state": "district", "region": "state"}[level]
    tables = {
        f"{place}: units by month and category, actual then predicted": pm.iloc[:A.complete_weeks(d).shape[0] // 4 + 8],
        f"{place}: category mix versus national (index 100 = national rate)": idx.sort_values("units", ascending=False),
        f"{place}: top 20 SKUs": _short(top),
        f"{place}: SKUs that over-index by season here": lift[lift["side"] == "Most"][["season", "main_sku", "units", "lift"]] if not lift.empty else lift,
    }
    if sub:
        tables[f"{place}: by {sub}"] = A.summary(here[here[sub].str.upper() != "UNKNOWN"], sub)[["units", "unit_share_pct", "growth_4w_pct"]].head(15)
    return tables, (f"Write the demand and stocking plan for {place} for the next three months: what this place buys differently from the rest of "
                    "India, which products to position close to it and when, and where inside it the demand sits. Predicted months are model "
                    "estimates allocated from state level; treat them as direction, not as exact counts.")


def fabric_plan(d, lc, products):
    fab = FB.load_fabric()
    pf = FB.product_fabric(products, fab, SRC.outsourced_asins(products))
    dm = FB.with_metres(d, pf)
    req = FB.requirement(d, pf)
    top_codes = FB.by_fabric(dm).head(20)
    end = d["date"].max()
    recent = dm[dm["date"] > end - pd.Timedelta(weeks=8)]
    unknown = (recent[recent["fabric_source"] == "inferred: size rule"].groupby("main_sku").agg(units_8w=("quantity", "sum"), metres_8w=("metres", "sum"))
               .sort_values("metres_8w", ascending=False).head(15).round(0))
    return {
        "12-week fabric requirement per fabric code (metres): baseline, festive scenario, last 12 weeks actual": req.head(30),
        "fabric codes by metres consumed in the year, with 4-week momentum": top_codes,
        "metres by fabric type": FB.by_fabric(dm, "fabric_type"),
        "metres by month and fabric code (top 12)": FB.monthly(dm),
        "coverage: units with a known fabric, by category": FB.coverage(d, pf),
        "biggest products whose metres are inferred by size rule and fabric code is unknown (last 8 weeks)": unknown,
    }, (f"Write the fabric purchase plan for the next 12 weeks: how many metres of each fabric code to have in hand and by when, given "
        f"{cfg.PRODUCTION_WEEKS} weeks of production before dispatch, with the festive scenario as the upside case. Call out codes whose "
        "consumption is rising or falling fast, the share of demand whose fabric code is unknown (new ranges missing from the sheet: say "
        "which SKUs to add), and any single-code concentration risk.")


def outsource_plan(d, lc, products):
    om, ps = SRC.build(products)
    sc = SRC.stock_cover(d, om, products, lc)
    cols = ["sku", "category", "ready", "virtual", "available", "units_8w", "fc_12w_baseline", "fc_12w_festive", "cover_weeks", "status",
            "reorder_baseline", "reorder_festive", "stage"]
    cols = [c for c in cols if c in sc.columns]
    risk = sc[sc["status"].isin(["Out of stock", "Critical", "Reorder now"])].sort_values("fc_12w_festive", ascending=False)
    over = sc[sc["status"] == "Overstock"].sort_values("available", ascending=False)
    idle = sc[sc["status"] == "Idle stock"].sort_values("available", ascending=False)
    gap = sc[(sc["virtual"] > sc["ready"]) & (sc["fc_12w_festive"] > sc["ready"])].assign(
        promised_beyond_ready=lambda t: t["virtual"] - t["ready"]).sort_values("fc_12w_festive", ascending=False)
    snap = pd.Timestamp(sc["snapshot"].iloc[0])
    return {
        "outsourced stock by status (SKUs, stock, last 8 weeks units, 12-week festive forecast)": SRC.status_summary(sc),
        "outsourced stock by category": SRC.by_category(sc),
        "at risk: selling SKUs that are out of stock or short of cover (top 30 by forecast)": risk[cols].head(30).set_index("sku"),
        "overstock: more than 26 weeks of cover (top 15 by stock)": over[cols].head(15).set_index("sku"),
        "idle stock: stock in hand, no sales in the last 8 weeks (top 15 by stock)": idle[cols + ["last_sale"]].head(15).set_index("sku"),
        "demand that depends on the maker: forecast exceeds ready stock, virtual quantity covers the rest (top 15)":
            gap[["sku", "category", "ready", "virtual", "promised_beyond_ready", "fc_12w_festive", "cover_weeks_ready_only"]].head(15).set_index("sku"),
        "share of units outsourced, by category (whole year)": SRC.sourcing_summary(d, ps),
    }, (f"Write the purchase and stock plan for the outsourced (bought-in) range from the stock snapshot of {snap:%d %b %Y}: which SKUs to "
        f"reorder now and how many units (lead time assumed {cfg.OUTSOURCE_LEAD_WEEKS} weeks, cover target lead time + {cfg.OUTSOURCE_REVIEW_WEEKS} "
        "weeks, festive scenario as the upside case), which to stop buying and run down, and what to do with idle stock (reprice, bundle, "
        "relist). Rank by units at risk. State the three caveats: sales data ends before the snapshot so cover rests on the forecast; stock "
        "already at Amazon fulfilment centres is not in the sheet; 'virtual quantity' is read as the quantity kept available to sell.")


def product_check(d, lc, products, sku: str):
    x = d[d["main_sku"] == sku]
    row = lc[lc["main_sku"] == sku]
    monthly = x.groupby("month").agg(units=("quantity", "sum"), median_price=("unit_price", "median"), promo_share=("has_promo", "mean")).round(2)
    cat = x["category"].mode().iloc[0]
    share = (x.groupby("month")["quantity"].sum() / d[d["category"] == cat].groupby("month")["quantity"].sum() * 100).round(2)
    monthly["share_of_category_pct"] = share
    peers = d[(d["family"] == x["family"].mode().iloc[0])].groupby("main_sku")["quantity"].sum().nlargest(8).rename("units").to_frame()
    return {
        f"{sku}: lifecycle facts": _short(row[LC_COLS + ["first_sale", "last_sale", "peak_week", "gap_ended"]]).T,
        f"{sku}: by month (units, median price, promotion share, share of {cat})": monthly,
        f"{sku}: by region": A.summary(x[x["region"] != "Unknown"], "region")[["units", "unit_share_pct", "growth_4w_pct"]],
        f"{sku}: top districts": x[x["district"] != "UNKNOWN"].groupby("district")["quantity"].sum().nlargest(10).rename("units").to_frame(),
        f"{sku}: by temperature band of the day (units)": x.groupby("temp_band")["quantity"].sum().rename("units").to_frame(),
        "sibling SKUs in the same design family (year units)": peers,
    }, (f"Give a verdict on SKU {sku}: is it healthy, seasonal, supply-constrained or fading, and what should be done with it over the next "
        "three months (stock level, price or promotion, where to position it)? Separate demand signals from possible stock-outs, and price "
        "effects from season.")
