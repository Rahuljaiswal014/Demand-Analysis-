"""Demand analytics: what sells, where, when, how weather moves it, and lifecycle stage.

Every function takes the demand frame (orders where is_demand) so the dashboard can pass a
filtered slice. Weeks run Monday-Sunday and only complete weeks are used for trends.
"""
import numpy as np
import pandas as pd

from . import config as cfg

SEASONS = {1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer", 6: "Monsoon",
           7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon", 12: "Winter"}
SEASON_ORDER = ["Summer", "Monsoon", "Post-monsoon", "Winter"]
STAGES = ["Launch", "Growth", "Mature", "Decline", "Dormant", "Sporadic"]


# ---------------------------------------------------------------- loading
def load_orders() -> pd.DataFrame:
    return pd.read_parquet(cfg.ORDERS_PARQUET)


def load_products() -> pd.DataFrame:
    return pd.read_parquet(cfg.PRODUCTS_PARQUET)


def load_weather() -> pd.DataFrame:
    return pd.read_parquet(cfg.WEATHER_PARQUET) if cfg.WEATHER_PARQUET.exists() else pd.DataFrame()


def demand(orders: pd.DataFrame) -> pd.DataFrame:
    return orders[orders["is_demand"]]


# ---------------------------------------------------------------- time helpers
def week_start(dates: pd.Series) -> pd.Series:
    return dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")


def complete_weeks(d: pd.DataFrame) -> pd.DatetimeIndex:
    """Monday starts of weeks fully covered by the data (partial edge weeks distort trends)."""
    lo, hi = d["date"].min(), d["date"].max()
    first = lo if lo.dayofweek == 0 else lo + pd.Timedelta(days=7 - lo.dayofweek)
    last = hi - pd.Timedelta(days=hi.dayofweek + 7) if hi.dayofweek != 6 else hi - pd.Timedelta(days=6)
    return pd.date_range(first, last, freq="7D")


def weekly_matrix(d: pd.DataFrame, by: str) -> pd.DataFrame:
    """rows = `by`, columns = complete weeks, values = units."""
    weeks = complete_weeks(d)
    w = d.assign(week=week_start(d["date"])).groupby([by, "week"])["quantity"].sum().unstack(fill_value=0)
    return w.reindex(columns=weeks, fill_value=0)


def _pct(a, b):
    return np.where(b > 0, (a / np.where(b > 0, b, 1) - 1) * 100, np.nan)


# ---------------------------------------------------------------- what is selling
def summary(d: pd.DataFrame, by: str) -> pd.DataFrame:
    """Units, revenue, ASP, share and 4-week-over-4-week growth for any grouping column."""
    g = d.groupby(by).agg(units=("quantity", "sum"), revenue=("revenue", "sum"),
                          orders=("order_id", "nunique"), asins=("asin", "nunique"))
    g["asp"] = (g["revenue"] / g["units"]).round(0)
    g["revenue"] = g["revenue"].round(0)
    g["unit_share_pct"] = (g["units"] / g["units"].sum() * 100).round(2)
    w = weekly_matrix(d, by)
    if w.shape[1] >= 8:
        g["last_4w"] = w.iloc[:, -4:].sum(axis=1)
        g["prev_4w"] = w.iloc[:, -8:-4].sum(axis=1)
        g["growth_4w_pct"] = np.round(_pct(g["last_4w"].to_numpy(float), g["prev_4w"].to_numpy(float)), 1)
    return g.sort_values("units", ascending=False)


def product_table(d: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    cols = ["asin", "title", "category", "family", "curtain_type", "curtain_length_ft", "pack_size", "main_sku"]
    return summary(d, "asin").drop(columns=["asins"]).reset_index().merge(products[cols], on="asin", how="left")


def pareto(d: pd.DataFrame) -> pd.DataFrame:
    s = d.groupby("asin")["quantity"].sum().sort_values(ascending=False)
    return pd.DataFrame({"rank": np.arange(1, len(s) + 1), "cum_share_pct": (s.cumsum() / s.sum() * 100).to_numpy()})


# ---------------------------------------------------------------- where
def state_table(d: pd.DataFrame) -> pd.DataFrame:
    t = summary(d, "state")
    t["region"] = d.groupby("state")["region"].first()
    return t


def category_index_by_state(d: pd.DataFrame, by: str = "category", min_state_units: int = 1500) -> pd.DataFrame:
    """Over/under-index: 100 = the state buys this group at the national rate, 130 = 30% more."""
    m = d.pivot_table(index="state", columns=by, values="quantity", aggfunc="sum", fill_value=0)
    m = m[m.sum(axis=1) >= min_state_units]
    state_mix = m.div(m.sum(axis=1), axis=0)
    national_mix = m.sum() / m.sum().sum()
    return (state_mix / national_mix * 100).round(0)


# ---------------------------------------------------------------- when
def daily_series(d: pd.DataFrame) -> pd.DataFrame:
    s = d.groupby("date").agg(units=("quantity", "sum"), revenue=("revenue", "sum"))
    return s.reindex(pd.date_range(s.index.min(), s.index.max()), fill_value=0).rename_axis("date")


def weekday_profile(d: pd.DataFrame) -> pd.Series:
    s = daily_series(d)["units"]
    p = s.groupby(s.index.dayofweek).mean()
    p.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return (p / p.mean() * 100).round(1)


def demand_events(d: pd.DataFrame, threshold: float = 1.4) -> pd.DataFrame:
    """Spike / slump days versus a weekday-adjusted 4-week rolling median, merged into events."""
    s = daily_series(d)["units"].astype(float)
    wd = s.groupby(s.index.dayofweek).transform("median") / s.median()
    base = (s / wd).rolling(29, center=True, min_periods=10).median() * wd
    ratio = s / base
    flag = np.where(ratio >= threshold, "Spike", np.where(ratio <= 1 / threshold, "Slump", ""))
    out, run = [], None
    for day, f, r, u in zip(s.index, flag, ratio, s):
        if f and run and run["type"] == f and (day - run["end"]).days == 1:
            run.update(end=day, units=run["units"] + u, peak_ratio=max(run["peak_ratio"], r) if f == "Spike" else min(run["peak_ratio"], r))
        else:
            if run:
                out.append(run)
            run = {"type": f, "start": day, "end": day, "units": u, "peak_ratio": r} if f else None
    if run:
        out.append(run)
    ev = pd.DataFrame(out)
    if ev.empty:
        return ev
    ev["days"] = (ev["end"] - ev["start"]).dt.days + 1
    ev["peak_vs_normal_pct"] = ((ev["peak_ratio"] - 1) * 100).round(0)
    return ev.drop(columns="peak_ratio")


def trend_table(d: pd.DataFrame, by: str, weeks: int = cfg.TREND_WEEKS, min_units: int = 40) -> pd.DataFrame:
    """Log-linear weekly growth over the last `weeks` complete weeks, with a direction label."""
    w = weekly_matrix(d, by)
    if w.shape[1] < weeks:
        return pd.DataFrame()
    recent = w.iloc[:, -weeks:]
    recent = recent[recent.sum(axis=1) >= min_units]
    slope, tstat = _loglinear(recent.to_numpy(float))
    t = pd.DataFrame({"units_recent": recent.sum(axis=1), "growth_pct_per_week": np.round((np.exp(slope) - 1) * 100, 1),
                      "t_stat": np.round(tstat, 1)}, index=recent.index)
    t["direction"] = np.select([(t["t_stat"] >= 2) & (t["growth_pct_per_week"] > 0),
                                (t["t_stat"] <= -2) & (t["growth_pct_per_week"] < 0)], ["Rising", "Falling"], "Stable")
    return t.sort_values("growth_pct_per_week", ascending=False)


def _loglinear(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise OLS of log(y+0.5) on time. Returns slope and its t-statistic."""
    return _slope(np.log(y + 0.5))


def _slope(ly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = ly.shape[1]
    t = np.arange(n) - (n - 1) / 2
    sxx = (t ** 2).sum()
    slope = ly @ t / sxx
    resid = ly - ly.mean(axis=1, keepdims=True) - np.outer(slope, t)
    se = np.sqrt((resid ** 2).sum(axis=1) / max(n - 2, 1) / sxx)
    return slope, np.divide(slope, se, out=np.zeros_like(slope), where=se > 0)


# ---------------------------------------------------------------- season & climate
def seasonal_profile(d: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    """Units/day by month as an index (100 = that group's own average). <12 months of history: one cycle only."""
    dd = d.assign(month=d["date"].dt.to_period("M"))
    days = dd.groupby("month")["date"].nunique()
    per_day = dd.pivot_table(index=by, columns="month", values="quantity", aggfunc="sum", fill_value=0).div(days, axis=1)
    idx = per_day.div(per_day.mean(axis=1), axis=0) * 100
    idx.columns = [str(c) for c in idx.columns]
    return idx.round(0)


def season_index(d: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    dd = d.assign(season=d["date"].dt.month.map(SEASONS))
    days = dd.groupby("season")["date"].nunique()
    per_day = dd.pivot_table(index=by, columns="season", values="quantity", aggfunc="sum", fill_value=0).div(days, axis=1)
    idx = per_day.div(per_day.mean(axis=1), axis=0) * 100
    return idx.reindex(columns=[s for s in SEASON_ORDER if s in idx.columns]).round(0)


def _state_week_panel(d: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = complete_weeks(d)
    big = d.groupby("state")["quantity"].sum()
    states = big[big >= cfg.CLIMATE_MIN_STATE_UNITS].index.intersection(weather["state"].unique())
    wx = weather[weather["state"].isin(states)].assign(week=lambda x: week_start(x["date"]))
    wx = wx.groupby(["state", "week"]).agg(temp_max=("temp_max", "mean"), rain_mm=("rain_mm", "sum"),
                                           humidity=("humidity", "mean"), n=("date", "size"))
    wx = wx[wx["n"] == 7].drop(columns="n")
    wx = wx[wx.index.get_level_values("week").isin(weeks)]
    dd = d[d["state"].isin(states)].assign(week=week_start(d["date"]))
    return dd, wx


def weather_sensitivity(d: pd.DataFrame, weather: pd.DataFrame, by: str = "category", min_weekly_units: float = 4.0) -> pd.DataFrame:
    """Two-way fixed-effects regression of log weekly units on weather, per group.

    State effects absorb "Karnataka always buys more"; week effects absorb national events
    (festive sales, Prime Day, price changes). What remains is: in a week when one state was
    hotter / wetter than its norm relative to the others, did it buy more of this group?
    """
    if weather.empty:
        return pd.DataFrame()
    dd, wx = _state_week_panel(d, weather)
    rows = []
    for grp, g in dd.groupby(by):
        y = g.groupby(["state", "week"])["quantity"].sum().reindex(wx.index, fill_value=0)
        keep = y.groupby("state").mean() >= min_weekly_units
        keep = keep[keep].index
        if len(keep) < 6:
            continue
        sel = y.index.get_level_values("state").isin(keep)
        yy, xx = np.log1p(y[sel]), wx[sel]
        panel = pd.concat([yy.rename("y"), xx], axis=1).dropna()
        # balanced panel -> the two-way within transform is exact
        full = panel.index.get_level_values("week").value_counts()
        panel = panel[panel.index.get_level_values("week").isin(full[full == len(keep)].index)]
        dm = panel - panel.groupby("state").transform("mean") - panel.groupby("week").transform("mean") + panel.mean()
        X, Y = dm[["temp_max", "rain_mm", "humidity"]].to_numpy(), dm["y"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        n_s, n_w = len(keep), panel.index.get_level_values("week").nunique()
        dof = len(Y) - n_s - n_w - X.shape[1] + 1
        if dof <= 10:
            continue
        sigma2 = ((Y - X @ beta) ** 2).sum() / dof
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
        t = beta / se
        rows.append({by: grp, "states": n_s, "weeks": n_w, "units": int(g["quantity"].sum()),
                     "per_1C_hotter_pct": (np.exp(beta[0]) - 1) * 100, "t_temp": t[0],
                     "per_10mm_rain_pct": (np.exp(beta[1] * 10) - 1) * 100, "t_rain": t[1],
                     "per_10pt_humidity_pct": (np.exp(beta[2] * 10) - 1) * 100, "t_humidity": t[2]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    strongest = out[["t_temp", "t_rain", "t_humidity"]].abs()
    out["weather_signal"] = np.select([strongest.max(axis=1) >= 4, strongest.max(axis=1) >= 2.5], ["Strong", "Moderate"], "None detected")
    return out.round(2).sort_values("units", ascending=False).set_index(by)


def temperature_response(d: pd.DataFrame, weather: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    """Demand index by temperature band, net of state size and national week effects (100 = normal)."""
    if weather.empty:
        return pd.DataFrame()
    dd, wx = _state_week_panel(d, weather)
    bands = pd.cut(wx["temp_max"], [-50, 24, 28, 32, 36, 60], labels=["<24°C", "24-28°C", "28-32°C", "32-36°C", ">36°C"])
    out = {}
    for grp, g in dd.groupby(by):
        y = g.groupby(["state", "week"])["quantity"].sum().reindex(wx.index, fill_value=0).astype(float)
        if y.sum() < 2000:
            continue
        rel = y / y.groupby("state").transform("mean")
        rel = rel / rel.groupby("week").transform("mean")
        out[grp] = rel.groupby(bands, observed=True).mean() * 100
    return pd.DataFrame(out).T.round(0)


# ---------------------------------------------------------------- lifecycle
def lifecycle(d: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Stage per ASIN. Growth/decline is measured on the product's share of its category, so a
    category-wide seasonal dip does not mark every product in it as declining."""
    data_start, data_end = d["date"].min(), d["date"].max()
    w = weekly_matrix(d, "asin")
    cat = products.set_index("asin")["category"].reindex(w.index)
    cat_week = w.groupby(cat).sum()
    share = w / cat_week.reindex(cat.to_numpy()).to_numpy().clip(min=1)

    n = cfg.TREND_WEEKS
    slope_abs, _ = _loglinear(w.iloc[:, -n:].to_numpy(float))
    eps = 0.5 / cat_week.reindex(cat.to_numpy()).iloc[:, -n:].to_numpy().clip(min=1)
    slope_share, t_share = _slope(np.log(share.iloc[:, -n:].to_numpy() + eps))

    roll_share = share.T.rolling(4, min_periods=2).mean().T
    roll_units = w.T.rolling(4, min_periods=2).mean().T

    g = d.groupby("asin")
    lc = pd.DataFrame({
        "first_sale": g["date"].min(), "last_sale": g["date"].max(), "units_total": g["quantity"].sum(),
    }).reindex(w.index)
    lc["age_days"] = (data_end - lc["first_sale"]).dt.days
    lc["launch_date_known"] = (lc["first_sale"] - data_start).dt.days > cfg.CENSOR_DAYS
    lc["days_since_sale"] = (data_end - lc["last_sale"]).dt.days
    lc["units_26w"] = w.iloc[:, -26:].sum(axis=1)
    lc["units_last_4w"] = w.iloc[:, -4:].sum(axis=1)
    lc["peak_4w_avg"] = roll_units.max(axis=1).round(1)
    lc["peak_week"] = roll_units.idxmax(axis=1)
    lc["recent_vs_peak_share"] = (roll_share.iloc[:, -1] / roll_share.max(axis=1).clip(lower=1e-9)).round(2)
    lc["growth_pct_wk"] = np.round((np.exp(slope_abs) - 1) * 100, 1)
    lc["growth_vs_category_pct_wk"] = np.round((np.exp(slope_share) - 1) * 100, 1)

    lc["trend_t_stat"] = np.round(t_share, 1)   # |t| < 1.5 means the slope is indistinguishable from noise

    gr, rp = lc["growth_vs_category_pct_wk"], lc["recent_vs_peak_share"]
    sure = lc["trend_t_stat"].abs() >= 1.5
    lc["stage"] = np.select(
        [lc["days_since_sale"] > cfg.DORMANT_DAYS,
         lc["launch_date_known"] & (lc["age_days"] <= cfg.LAUNCH_DAYS),
         lc["units_26w"] < cfg.SPORADIC_UNITS_26W,
         sure & (gr >= cfg.GROWTH_PCT_WK) & (rp >= 0.7),
         (sure & (gr <= cfg.DECLINE_PCT_WK)) | (rp < 0.4)],
        ["Dormant", "Launch", "Sporadic", "Growth", "Decline"], "Mature")

    lc = lc.join(_stockout_suspects(d, lc))
    lc["stockout_suspected"] = lc["stockout_suspected"].fillna(False).astype(bool)
    keep = ["asin", "title", "category", "family", "curtain_type", "curtain_length_ft", "main_sku"]
    return lc.reset_index().merge(products[keep], on="asin", how="left")


def _stockout_suspects(d: pd.DataFrame, lc: pd.DataFrame) -> pd.DataFrame:
    """Steady sellers (>=1/day) with a long zero-sales gap that later resumed. Proxy until FBA inventory is wired in."""
    life = (lc["last_sale"] - lc["first_sale"]).dt.days + 1
    steady = lc.index[(lc["units_total"] / life.clip(lower=1) >= 1) & (life >= 60)]
    dd = d[d["asin"].isin(steady)].drop_duplicates(["asin", "date"]).sort_values(["asin", "date"])
    gap = dd.groupby("asin")["date"].diff().dt.days - 1
    dd = dd.assign(gap=gap)
    worst = dd.loc[dd.groupby("asin")["gap"].idxmax().dropna()].set_index("asin")
    out = pd.DataFrame({"longest_gap_days": worst["gap"], "gap_ended": worst["date"]})
    out["stockout_suspected"] = out["longest_gap_days"] >= cfg.STOCKOUT_GAP_DAYS
    return out


def stage_summary(lc: pd.DataFrame) -> pd.DataFrame:
    s = lc.groupby("stage").agg(asins=("asin", "size"), units_last_4w=("units_last_4w", "sum"), units_total=("units_total", "sum"))
    s["share_of_recent_units_pct"] = (s["units_last_4w"] / s["units_last_4w"].sum() * 100).round(1)
    return s.reindex([x for x in STAGES if x in s.index])


# ---------------------------------------------------------------- most / least ordered, by date, region and climate
TEMP_BANDS = ["<24°C", "24-28°C", "28-32°C", "32-36°C", ">36°C"]
RAIN_BANDS = ["Dry day", "Light rain (1-10 mm)", "Rainy day (>10 mm)"]
_RANK_COLS = ("side", "rank", "units", "lift", "item_total_units", "competing_items", "zero_sellers")


def add_context(d: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Tag every order line with its calendar and climate context: month, week, season, and the
    temperature / rain band of that day in the buyer's state."""
    d = d.assign(month=d["date"].dt.strftime("%Y-%m"), week=week_start(d["date"]), season=d["date"].dt.month.map(SEASONS))
    if weather.empty:
        return d.assign(temp_max=np.nan, rain_mm=np.nan, temp_band=None, rain_band=None)
    d = d.merge(weather[["state", "date", "temp_max", "rain_mm"]], on=["state", "date"], how="left")
    d["temp_band"] = pd.cut(d["temp_max"], [-50, 24, 28, 32, 36, 60], labels=TEMP_BANDS).astype("object")
    d["rain_band"] = pd.cut(d["rain_mm"], [-1, 1, 10, 1e4], labels=RAIN_BANDS, right=False).astype("object")
    return d


def _exposure(dd: pd.DataFrame, item: str, items: pd.Index, cols: list[str], cells: pd.Index) -> np.ndarray:
    """items x cells: units sold by *all* products in each cell during the days each item was on sale
    (first to last sale). It is the fair denominator for "how did this item do here": a product launched in
    January is not judged against a season, month or cold spell that happened before it existed. 0 = never exposed."""
    life = dd.groupby(item)["date"].agg(["min", "max"]).reindex(items)
    if "date" in cols:                                   # a cell is one day: exposed iff that day is inside the item's life
        total = dd.groupby(cols)["quantity"].sum().reindex(cells).to_numpy(float)
        day = pd.to_datetime(cells.get_level_values("date")).to_numpy()
        alive = (day[None, :] >= life["min"].to_numpy()[:, None]) & (day[None, :] <= life["max"].to_numpy()[:, None])
        return np.where(alive, total[None, :], 0.0)
    daily = dd.groupby(["date"] + cols)["quantity"].sum().unstack(cols, fill_value=0).reindex(columns=cells, fill_value=0)
    cum = np.vstack([np.zeros(daily.shape[1]), daily.cumsum().to_numpy(float)])
    start = daily.index.searchsorted(life["min"].to_numpy())
    stop = daily.index.searchsorted(life["max"].to_numpy(), side="right")
    return cum[stop] - cum[start]


def rank_cells(d: pd.DataFrame, item: str, context: str, scope: str | None = None, n: int = 5,
               min_item_units: int = 50, metric: str = "units") -> pd.DataFrame:
    """Most and least ordered `item` (sku / asin / family / category) in every cell of context x scope.

    context: date | week | month | season | temp_band | rain_band      scope: None | region | state
    metric 'units' ranks by volume. 'lift' ranks by over-index: the item's share of the cell divided by
    its overall share (100 = sells here exactly as everywhere else), which surfaces what is *distinctively*
    strong or weak in a region or climate instead of naming the national best-seller every time.

    Only items with >= min_item_units overall compete, and an item only competes in cells it was exposed to
    (see _exposure), so a not-yet-launched or discontinued item is never "least ordered" and lift compares an
    item with the market during its own selling life. Ties at the bottom (many zeros) are broken towards the bigger item: a large product selling nothing is the news.
    """
    cols = [c for c in (scope, context) if c]
    dd = d.dropna(subset=cols + [item])
    if scope:
        dd = dd[dd[scope].str.upper() != "UNKNOWN"]          # the few rows whose address could not be resolved
    totals = dd.groupby(item)["quantity"].sum()
    keep = totals[totals >= min_item_units].index
    m = dd[dd[item].isin(keep)].groupby([item] + cols)["quantity"].sum().unstack(cols, fill_value=0)
    if m.empty:
        return pd.DataFrame()
    units = m.to_numpy(float)
    size = totals[m.index].to_numpy(float)
    market = _exposure(dd, item, m.index, cols, m.columns)
    active = market > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (units / market) / (size / market.sum(axis=1))[:, None] * 100

    score = np.where(active, lift if metric == "lift" else units, np.nan)
    rows = []
    for j, cell in enumerate(m.columns):
        idx = np.flatnonzero(~np.isnan(score[:, j]))
        if not len(idx):
            continue
        col = score[idx, j]
        top = idx[np.lexsort((-size[idx], -col))][:n]
        bottom = idx[np.lexsort((-size[idx], col))][:n]
        key = dict(zip(cols, cell if scope else (cell,)))
        for side, picks in (("Most", top), ("Least", bottom)):
            for r, i in enumerate(picks, 1):
                rows.append({**key, "side": side, "rank": r, item: m.index[i], "units": int(units[i, j]),
                             "lift": round(lift[i, j], 0), "item_total_units": int(size[i]),
                             "competing_items": len(idx), "zero_sellers": int((units[idx, j] == 0).sum())})
    return pd.DataFrame(rows)


def extremes(ranked: pd.DataFrame, item: str) -> pd.DataFrame:
    """One line per cell: the most and the least ordered item."""
    if ranked.empty:
        return ranked
    keys = [c for c in ranked.columns if c not in _RANK_COLS and c != item]
    first = ranked[ranked["rank"] == 1]
    most = first[first["side"] == "Most"].set_index(keys)[[item, "units", "lift"]].add_prefix("most_")
    least = first[first["side"] == "Least"].set_index(keys)[[item, "units", "lift", "competing_items", "zero_sellers"]].add_prefix("least_")
    return (most.join(least).rename(columns={"least_competing_items": "competing_items", "least_zero_sellers": "zero_sellers"})
                .reset_index())


def item_peaks(d: pd.DataFrame, item: str, min_item_units: int = 50) -> pd.DataFrame:
    """Per item: when and where it sells most and least (best / weakest month, best date, top region and
    state, and the region, season and temperature band it over-indexes in). Expects add_context() columns."""
    totals = d.groupby(item)["quantity"].sum()
    dd = d[d[item].isin(totals[totals >= min_item_units].index)]
    g = dd.groupby(item)
    out = pd.DataFrame({"units": g["quantity"].sum(), "first_sale": g["date"].min(), "last_sale": g["date"].max()})

    def best(col, how="max", lift=False, mask_life=False):
        src = d.dropna(subset=[col])
        p = dd.pivot_table(index=item, columns=col, values="quantity", aggfunc="sum", fill_value=0).astype(float)
        if mask_life or lift:                           # judge the item only against what it was on sale for
            market = _exposure(src, item, p.index, [col], p.columns)
            if lift:                                    # over-index versus the whole market during the item's life
                with np.errstate(divide="ignore", invalid="ignore"):
                    p = (p / market).div(p.sum(axis=1) / market.sum(axis=1), axis=0) * 100
            p = p.where(market > 0)
        pick = p.idxmax(axis=1) if how == "max" else p.idxmin(axis=1)
        return pick, (p.max(axis=1) if how == "max" else p.min(axis=1)).round(0)

    out["best_month"], out["best_month_units"] = best("month", mask_life=True)
    out["weakest_month"], out["weakest_month_units"] = best("month", "min", mask_life=True)
    out["best_date"], out["best_date_units"] = best("date")
    out["top_region"], _ = best("region")
    out["top_state"], _ = best("state")
    for col in ("region", "season", "temp_band"):
        if dd[col].notna().any():
            out[f"overindexed_{col}"], out[f"overindexed_{col}_lift"] = best(col, lift=True)
    out["best_date"] = pd.to_datetime(out["best_date"]).dt.date
    return out.drop(columns=["first_sale", "last_sale"]).sort_values("units", ascending=False).reset_index()
