"""Written key points and suggested actions computed straight from the numbers: no AI, no cost, instant, and always
consistent with the tables on screen. Each function returns markdown bullets. Claude's write-ups sit on top of these
for depth; these are the floor that is always there."""
import pandas as pd

from . import analytics as A
from . import config as cfg

UNKNOWN_KEY = "(unknown: new ranges, by size rule)"
MIN_UNITS = 400          # groups smaller than this are too noisy to call out as "growing" or "shrinking"


def _pct(v) -> str:
    return f"{v:+.0f}%"


def _mon(m, fmt: str = "%b %Y") -> str:
    return pd.Period(m).strftime(fmt)


def _place(label: str) -> str:
    """'BANGALORE, KA' -> 'Bangalore, KA'"""
    name, _, abbr = str(label).rpartition(", ")
    return f"{name.title()}, {abbr}" if name else str(label).title()


def _md(points: list[str]) -> str:
    return "\n".join(f"- {p}" for p in points if p)


def _movers(t: pd.DataFrame, min_units: int = MIN_UNITS):
    """Biggest riser and faller by 4-week growth among groups with enough volume."""
    if "growth_4w_pct" not in t:
        return None, None
    big = t[(t["prev_4w"] >= min_units / 4) & t["growth_4w_pct"].notna()]
    if big.empty:
        return None, None
    return big["growth_4w_pct"].idxmax(), big["growth_4w_pct"].idxmin()


def overview(d: pd.DataFrame) -> str:
    cat, reg = A.summary(d, "category"), A.summary(d[d["region"] != "Unknown"], "region")
    total = A.summary(d.assign(_a="all"), "_a").iloc[0]
    pts = [f"**{cat.index[0]}** is {cat['unit_share_pct'].iloc[0]:.0f}% of units; the next two are {cat.index[1]} "
           f"({cat['unit_share_pct'].iloc[1]:.0f}%) and {cat.index[2]} ({cat['unit_share_pct'].iloc[2]:.0f}%)."]
    if "growth_4w_pct" in total and pd.notna(total["growth_4w_pct"]):
        pts.append(f"The last 4 weeks ran **{_pct(total['growth_4w_pct'])}** against the 4 weeks before "
                   f"({total['last_4w']:,.0f} vs {total['prev_4w']:,.0f} units).")
    up, down = _movers(cat)
    if up is not None and up != down:
        best = cat.at[up, "growth_4w_pct"]
        lead = f"Fastest-growing category: **{up}** ({_pct(best)})" if best > 0.5 else f"No category grew over the last 4 weeks; **{up}** held up best ({_pct(best)})"
        pts.append(f"{lead}; weakest: **{down}** ({_pct(cat.at[down, 'growth_4w_pct'])}).")
    pts.append(f"**{reg.index[0]}** leads the regions with {reg['unit_share_pct'].iloc[0]:.0f}% of units"
               + (f"; over the last 4 weeks {reg['growth_4w_pct'].idxmax()} did best ({_pct(reg['growth_4w_pct'].max())}) and "
                  f"{reg['growth_4w_pct'].idxmin()} worst ({_pct(reg['growth_4w_pct'].min())})." if "growth_4w_pct" in reg else "."))
    ev = A.demand_events(d)
    if not ev.empty:
        top = ev[ev["type"] == "Spike"].nlargest(1, "units")
        if not top.empty:
            e = top.iloc[0]
            pts.append(f"Biggest demand spike: **{e['start']:%d %b} to {e['end']:%d %b %Y}**, peaking {e['peak_vs_normal_pct']:.0f}% above a normal day "
                       f"({e['units']:,.0f} units in {e['days']} days).")
    return _md(pts)


def geography(d: pd.DataFrame) -> str:
    dd = d[d["state"] != "UNKNOWN"]
    st = A.state_table(dd)
    top3 = st.head(3)
    pts = [f"Three states make {top3['unit_share_pct'].sum():.0f}% of units: " + ", ".join(f"**{s.title()}** ({v:.0f}%)" for s, v in top3["unit_share_pct"].items()) + "."]
    dist = dd[dd["district"] != "UNKNOWN"].groupby("district")["quantity"].sum().sort_values(ascending=False)
    if len(dist):
        pts.append(f"By district, **{_place(dist.index[0])}** alone is {dist.iloc[0] / dist.sum() * 100:.0f}% of units; the top 10 districts are "
                   f"{dist.head(10).sum() / dist.sum() * 100:.0f}%. Inventory placed near these cities serves most of the demand.")
    up, down = _movers(st, 1200)
    if up is not None and up != down:
        pts.append(f"Among larger states, **{up.title()}** grew most over the last 4 weeks ({_pct(st.at[up, 'growth_4w_pct'])}) and "
                   f"**{down.title()}** fell most ({_pct(st.at[down, 'growth_4w_pct'])}).")
    idx = A.category_index_by_state(dd, "category", 3000)
    vol = dd.groupby("category")["quantity"].sum()
    idx = idx.loc[:, vol.reindex(idx.columns) >= 2000]
    if not idx.empty:
        s = idx.stack()
        hi = s[s >= 150].sort_values(ascending=False)
        hi = hi[~hi.index.get_level_values(1).duplicated()].head(3)          # one example per category, not three of the same
        if len(hi):
            pts.append("Strong local tastes: " + "; ".join(f"**{st_.title()}** buys {cat_} at {v / 100:.1f}x the national rate" for (st_, cat_), v in hi.items()) + ".")
    return _md(pts)


def trends(d: pd.DataFrame, by: str = "category") -> str:
    tr = A.trend_table(d.dropna(subset=[by]), by)
    pts = []
    if not tr.empty:
        rising, falling = tr[tr["direction"] == "Rising"], tr[tr["direction"] == "Falling"]
        pts.append(("Rising over the last 8 weeks: " + ", ".join(f"**{i}** ({v:+.1f}%/wk)" for i, v in rising["growth_pct_per_week"].head(4).items()) + ".")
                   if len(rising) else "Nothing is rising with statistical confidence over the last 8 weeks.")
        if len(falling):
            worst = falling.sort_values("growth_pct_per_week").head(4)
            pts.append("Falling: " + ", ".join(f"**{i}** ({v:+.1f}%/wk)" for i, v in worst["growth_pct_per_week"].items())
                       + ". Check the Season & climate page before reading this as lost demand: several categories dip every monsoon.")
    wd = A.weekday_profile(d)
    pts.append(f"**{wd.idxmax()}** is the strongest day ({wd.max() - 100:+.0f}% vs an average day) and **{wd.idxmin()}** the weakest ({wd.min() - 100:+.0f}%): "
               "time deals and ad budgets to the weekend.")
    return _md(pts)


def climate(d: pd.DataFrame, ws: pd.DataFrame, by: str = "category") -> str:
    pts = []
    season = A.season_index(d, by)
    vol = d.groupby(by)["quantity"].sum()
    for grp in vol[vol >= 2000].sort_values(ascending=False).index[:6]:
        if grp in season.index:
            row = season.loc[grp].dropna()
            if row.max() >= 125:
                pts.append(f"**{grp}** peaks in **{row.idxmax()}** (index {row.max():.0f}, where 100 is its own average) and is weakest in {row.idxmin()} ({row.min():.0f}).")
    if ws is not None and not ws.empty:
        strong = ws[ws["t_temp"].abs() >= 2.5].sort_values("per_1C_hotter_pct")
        if len(strong):
            pts.append("Temperature, comparing states within the same week: " + "; ".join(
                f"**{i}** {v:+.1f}% per +1°C" for i, v in strong["per_1C_hotter_pct"].items()) + ".")
    pts.append("Every month has been seen only once, so treat the seasonal shape as one observation; the temperature effects are better founded "
               "because they compare 20 states inside the same weeks.")
    return _md(pts)


def lifecycle(lc: pd.DataFrame) -> str:
    ss = A.stage_summary(lc)
    pts = []
    if "Growth" in ss.index and "Decline" in ss.index:
        pts.append(f"**{int(ss.at['Growth', 'asins'])} products are growing** and carry {ss.at['Growth', 'share_of_recent_units_pct']:.0f}% of recent units; "
                   f"**{int(ss.at['Decline', 'asins'])} are declining** and still carry {ss.at['Decline', 'share_of_recent_units_pct']:.0f}%.")
    g = lc[lc["stage"] == "Growth"].nlargest(3, "units_last_4w")
    if len(g):
        pts.append("Biggest growers: " + "; ".join(f"`{r.main_sku}` ({r.units_last_4w:,.0f} units in 4 wks, {r.growth_vs_category_pct_wk:+.0f}%/wk vs its category)" for r in g.itertuples()) + ".")
    dcl = lc[lc["stage"] == "Decline"].nlargest(3, "units_last_4w")
    if len(dcl):
        pts.append("Largest decliners: " + "; ".join(f"`{r.main_sku}` (now at {r.recent_vs_peak_share * 100:.0f}% of its peak share)" for r in dcl.itertuples()) + ".")
    so = lc[lc["stockout_suspected"] & lc["stage"].isin(["Decline", "Mature", "Growth"])].nlargest(3, "units_total")
    if len(so):
        pts.append("Possible stock-outs among big sellers (a zero-sales gap that later resumed): " + "; ".join(
            f"`{r.main_sku}` ({int(r.longest_gap_days)} days, ended {pd.Timestamp(r.gap_ended):%d %b})" for r in so.itertuples()) + ". A 'decline' here may be supply, not demand.")
    if "Dormant" in ss.index:
        pts.append(f"{int(ss.at['Dormant', 'asins']):,} products have not sold for {cfg.DORMANT_DAYS}+ days: candidates to delist or clear, after checking they are not simply out of stock.")
    return _md(pts)


def forecast(fc: pd.DataFrame, hist_last: float, horizon: int) -> str:
    base, fest = fc["baseline"].sum(), fc["festive_scenario"].sum()
    pts = [f"Baseline for the next {horizon} weeks: **{base:,.0f} units**, {_pct((base / max(hist_last, 1) - 1) * 100)} against the last {horizon} weeks ({hist_last:,.0f})."]
    lift = fc[fc["festive_multiplier"] > 1.15]
    if len(lift):
        pts.append(f"Festive scenario: **{fest:,.0f} units**. The lift falls in the weeks of {lift['week'].min():%d %b} to {lift['week'].max():%d %b}, peaking at "
                   f"{lift['festive_multiplier'].max():.1f}x a normal week. Stock for those weeks has to be dispatched to Amazon by {lift['week'].min() - pd.Timedelta(weeks=cfg.INBOUND_WEEKS):%d %b}.")
    dip = fc[fc["festive_multiplier"] < 0.9]
    if len(dip):
        pts.append(f"Expect a post-festive dip from {dip['week'].min():%d %b} (about {dip['festive_multiplier'].mean():.1f}x normal): do not read it as a loss of demand.")
    return _md(pts)


def calendar(pm: pd.DataFrame, where: str) -> str:
    """pm: month x category units with a 'kind' column (from predict.place_monthly)."""
    cats = [c for c in pm.columns if c != "kind"]
    fut = pm[pm["kind"] == "predicted"].iloc[:12]
    if fut.empty:
        return ""
    tot = fut[cats].sum(axis=1)
    pts = [f"In **{where}** the model expects the next 12 months to peak in **{_mon(tot.idxmax())}** ({tot.max():,.0f} units) and bottom out in "
           f"**{_mon(tot.idxmin())}** ({tot.min():,.0f}), a {tot.max() / max(tot.min(), 1):.1f}x swing."]
    big = fut[cats].sum().sort_values(ascending=False)
    for c in big[big >= max(50, 0.01 * big.sum())].index[:5]:
        s = fut[c]
        if s.max() >= 1.3 * s.mean():
            due = pd.Period(s.idxmax()).start_time - pd.Timedelta(weeks=cfg.PRODUCTION_WEEKS + cfg.INBOUND_WEEKS)
            when = f"that order is due by {due:%d %b}" if due > pd.Timestamp.today() else "that order should already be placed"
            pts.append(f"**{c}**: strongest in {_mon(s.idxmax(), '%b')} ({s.max():,.0f}), about {s.max() / max(s.mean(), 1):.1f}x its average month. "
                       f"With {cfg.PRODUCTION_WEEKS} weeks of production and {cfg.INBOUND_WEEKS} of inbound, {when}.")
    return _md(pts)


def actions(d: pd.DataFrame, lc: pd.DataFrame, fc_cat: pd.DataFrame) -> str:
    """The rule-based action list shown on the Forecast & suggestions page."""
    out = []
    g = lc[(lc["stage"] == "Growth")].nlargest(6, "units_last_4w")
    if len(g):
        out.append("**Protect availability (growing products)**\n" + _md(
            [f"`{r.main_sku}` · {str(r.title)[:60]}: {r.units_last_4w:,.0f} units in the last 4 weeks, gaining {r.growth_vs_category_pct_wk:+.0f}%/wk on its category." for r in g.itertuples()]))
    lift = fc_cat.groupby("group").agg(base=("baseline", "sum"), fest=("festive_scenario", "sum"))
    lift = lift[lift["base"] >= 300].assign(x=lambda t: t["fest"] / t["base"]).sort_values("x", ascending=False).head(4)
    if len(lift) and lift["x"].max() > 1.05:
        out.append("**Festive build (next 12 weeks, if 2025's pattern repeats)**\n" + _md(
            [f"**{c}**: plan {r.fest:,.0f} units instead of the baseline {r.base:,.0f} ({(r.x - 1) * 100:+.0f}%)." for c, r in lift.iterrows() if r.x > 1.05]))
    dd = d[d["region"] != "Unknown"]
    idx = A.category_index_by_state(dd.assign(state=dd["region"]), "category", 0)
    vol = dd.groupby("category")["quantity"].sum()
    idx = idx.loc[:, vol.reindex(idx.columns) >= 3000]
    picks = [f"**{reg}** over-buys {row.idxmax()} ({row.max() / 100:.1f}x the national rate) and under-buys {row.idxmin()} ({row.min() / 100:.1f}x)."
             for reg, row in idx.iterrows() if row.max() >= 115]
    if picks:
        out.append("**Regional placement**\n" + _md(picks))
    so = lc[lc["stockout_suspected"] & (lc["units_last_4w"] >= 40)].nlargest(5, "units_total")
    if len(so):
        out.append("**Check stock before trusting the trend**\n" + _md(
            [f"`{r.main_sku}`: no sales for {int(r.longest_gap_days)} days up to {pd.Timestamp(r.gap_ended):%d %b}, then sales resumed. Stage shown as {r.stage}." for r in so.itertuples()]))
    dec = lc[(lc["stage"] == "Decline") & ~lc["stockout_suspected"]].nlargest(5, "units_total")
    if len(dec):
        out.append("**Slow movers to review (declining, no stock-out signal)**\n" + _md(
            [f"`{r.main_sku}` · {str(r.title)[:60]}: {r.units_total:,.0f} units in the year, now at {r.recent_vs_peak_share * 100:.0f}% of its peak share. "
             "Reprice, refresh the listing, or run down stock." for r in dec.itertuples()]))
    return "\n\n".join(out)


def fabric(dm: pd.DataFrame, req: pd.DataFrame, cov: pd.DataFrame) -> str:
    """Key points for the fabric page. dm = with_metres(demand), req = requirement(), cov = coverage()."""
    known = dm.dropna(subset=["metres"])
    if known.empty:
        return "- No fabric sheet loaded."
    total = known["metres"].sum()
    inferred = known.loc[known["fabric_source"] == "inferred: size rule", "metres"].sum()
    pts = [f"Sales in this view consumed **{total:,.0f} m** of fabric ({cov.loc['ALL', 'known_%']:.0f}% of units have a fabric figure; "
           f"{inferred / total * 100:.0f}% of the metres are by size rule for products the sheet does not list)."]
    bt = known.groupby("fabric_type")["metres"].sum().sort_values(ascending=False)
    pts.append("By type (of the metres with a known fabric): " + ", ".join(f"**{k}** {v / bt.sum() * 100:.0f}%" for k, v in bt.head(4).items()) + ".")
    codes = known[~known["fabric_code"].isin([UNKNOWN_KEY, "(none)"])].groupby("fabric_code")["metres"].sum().sort_values(ascending=False)
    if len(codes):
        top3 = codes.head(3)
        pts.append(f"Three codes carry {top3.sum() / codes.sum() * 100:.0f}% of the coded metres: "
                   + ", ".join(f"**{k}** ({v:,.0f} m)" for k, v in top3.items()) + ": a supply problem on **{}** alone would hit {:.0f}% of the coded range.".format(top3.index[0], top3.iloc[0] / codes.sum() * 100))
    if req is not None and not req.empty:
        coded = req[req.index != UNKNOWN_KEY]
        if len(coded):
            pts.append(f"Next 12 weeks, coded fabrics: **{coded['baseline_m'].sum():,.0f} m** baseline, **{coded['festive_m'].sum():,.0f} m** in the festive scenario. "
                       f"Largest: {', '.join(f'**{k}** {v:,.0f} m' for k, v in coded['festive_m'].head(3).items())}.")
        if UNKNOWN_KEY in req.index:
            u = req.loc[UNKNOWN_KEY]
            pts.append(f"**{u['festive_m']:,.0f} m** of the festive requirement ({u['festive_m'] / req['festive_m'].sum() * 100:.0f}%) has no fabric code: "
                       f"{int(u['products'])} products (the new ranges) are not in the sheet yet. Adding them is the single biggest accuracy gain.")
    return _md(pts)


def outsource(sc: pd.DataFrame, share: pd.DataFrame, lead_weeks: int) -> str:
    """Key points for the outsourced stock page. sc = sourcing.stock_cover(), share = sourcing.sourcing_summary()."""
    if sc is None or sc.empty:
        return "- No outsource sheet loaded."
    pts = []
    if "Outsourced" in share.columns:
        a = share.loc["ALL"]
        heavy = share.drop(index="ALL")
        heavy = heavy[(heavy["outsourced_%"] >= 40) & (heavy["units"] >= 500)].sort_values("Outsourced", ascending=False)
        pts.append(f"Outsourced products are **{a['outsourced_%']:.1f}%** of units sold ({a['Outsourced']:,.0f}); the rest is made in-house. "
                   + ("Bought-in categories: " + ", ".join(f"**{c}** {r['outsourced_%']:.0f}%" for c, r in heavy.iterrows()) + "." if len(heavy) else ""))
    n = sc["status"].astype(str).value_counts()
    risk = sc[sc["status"].isin(["Out of stock", "Critical", "Reorder now"])]
    pts.append(f"Of {len(sc)} SKUs on the sheet, **{n.get('Out of stock', 0)}** are selling with nothing available, **{n.get('Critical', 0)}** have under "
               f"2 weeks of cover and **{n.get('Reorder now', 0)}** run out inside the {lead_weeks}-week lead time plus 2 weeks. Together they carry "
               f"**{risk['fc_12w_festive'].sum():,.0f} units** of the next 12 weeks' demand against {risk['available'].sum():,.0f} units available.")
    top = risk.sort_values("fc_12w_festive", ascending=False).head(3)
    if len(top):
        pts.append("Largest exposures: " + "; ".join(
            f"`{r.sku}` ({r.available:,.0f} available, {r.fc_12w_festive:,.0f} forecast)" for r in top.itertuples()) + ".")
    slow = sc[sc["status"].isin(["Overstock", "Idle stock"])]
    if len(slow):
        idle = sc[sc["status"] == "Idle stock"]
        pts.append(f"**{slow['available'].sum():,.0f} units** ({slow['available'].sum() / max(sc['available'].sum(), 1) * 100:.0f}% of available stock) sit in "
                   f"{len(slow)} SKUs with more than 26 weeks of cover or no sales; {len(idle)} of them ({idle['available'].sum():,.0f} units) "
                   "have not sold in 8 weeks.")
    dep = sc[(sc["virtual"] > sc["ready"]) & (sc["fc_12w_festive"] > sc["ready"])]
    if len(dep):
        pts.append(f"For {len(dep)} SKUs the ready stock alone does not cover 12 weeks of demand and the virtual quantity makes up the difference: "
                   f"**{(dep[['fc_12w_festive', 'virtual']].min(axis=1) - dep['ready']).clip(lower=0).sum():,.0f} units** of demand depend on the maker delivering.")
    un = sc[sc["asin"].isna()]
    if len(un):
        pts.append(f"{len(un)} sheet SKUs ({un['available'].sum():,.0f} units available) have never appeared in the sales data: not listed yet, "
                   "or listed under a different SKU.")
    return _md(pts)
