"""Long-range demand prediction by region, category and month (the ML layer).

Why this model and not a classic seasonal forecast: the history is one year (Sep-25 to Aug-26). Every calendar month
has been seen exactly once, so "same month last year" is a single number per month with trend, season, festival and
one-off events all mixed into it, and no month has repeated yet.
What the data does contain is 36 states living through very different weather in the same weeks, and one Diwali.
So the model learns *why* demand moves rather than *when*:

    weekly units of a category in a state  =  that state-category's normal level
                                              x  f(category, region, temperature, rain, humidity, weeks to Diwali)

f is a gradient-boosted tree model trained on ~20k state x category x week rows. To predict a future month it is
fed that state's climate normals (10-year average weather for those days) and the festival calendar, which is how
it can move the festive peak with Diwali (20 Oct 2025, 8 Nov 2026, 29 Oct 2027) instead of replaying last year's calendar.

What it deliberately does NOT do is extrapolate growth. Year-on-year growth is unobservable until a month repeats
(September 2026 will be the first), so it is an explicit input (`growth_pct_per_year`), not something the model pretends to know.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import analytics as A
from . import config as cfg

FEATURES = ["category_code", "region_code", "state_code", "temp_max", "temp_min", "rain_mm", "humidity", "weeks_to_diwali"]
CATEGORICAL = [0, 1, 2]
NO_FESTIVAL = 99
SCENARIOS = {"Returns to its full-year norm": "revert", "Stays at the current run-rate": "persist"}
# Weeks relative to the Diwali week that carry a festive effect. In 2025 the Amazon festive sale opened on 22 Sep, four weeks
# before the Diwali week (20 Oct), and demand dipped for about a month afterwards. One lead-in week is kept on each side.
FESTIVE_WINDOW = (-5, 4)


@dataclass
class Model:
    gbm: HistGradientBoostingRegressor
    base: pd.Series                 # log normal weekly level per (state, category)
    codes: dict                     # category / region / state -> integer code
    state_region: pd.Series
    last_week: pd.Timestamp


# ------------------------------------------------------------------ features
def _diwali_dates() -> list[pd.Timestamp]:
    ev = pd.read_csv(cfg.EVENTS_CSV, parse_dates=["date"])
    return sorted(ev.loc[ev["event"].str.lower().eq("diwali"), "date"])


def _weeks_to_diwali(weeks: pd.Series) -> np.ndarray:
    """Signed distance in weeks to the nearest Diwali week inside FESTIVE_WINDOW; every other week is 'no festival'."""
    out = np.full(len(weeks), NO_FESTIVAL, dtype=float)
    wk = weeks.to_numpy()
    for d in _diwali_dates():
        k = np.round((wk - np.datetime64(d - pd.Timedelta(days=d.dayofweek))) / np.timedelta64(7, "D"))
        hit = (k >= FESTIVE_WINDOW[0]) & (k <= FESTIVE_WINDOW[1])
        out[hit] = k[hit]
    return out


def _weekly_weather(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.assign(week=A.week_start(weather["date"]))
    g = w.groupby(["state", "week"]).agg(temp_max=("temp_max", "mean"), temp_min=("temp_min", "mean"), rain_mm=("rain_mm", "sum"),
                                         humidity=("humidity", "mean"), n=("date", "size"))
    return g[g["n"] == 7].drop(columns="n").reset_index()


def _normal_weather(normals: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    days = pd.DataFrame({"date": [w + pd.Timedelta(days=i) for w in weeks for i in range(7)]})
    days["week"], days["doy"] = A.week_start(days["date"]), days["date"].dt.dayofyear.clip(upper=365)
    x = days.merge(normals, on="doy")
    return x.groupby(["state", "week"]).agg(temp_max=("temp_max", "mean"), temp_min=("temp_min", "mean"), rain_mm=("rain_mm", "sum"),
                                            humidity=("humidity", "mean")).reset_index()


def load_normals(weather: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Day-of-year climate per state for predicting future months, and a note on where it came from.

    Preferred: 10-year normals downloaded by `python -m demand.weather`. For any state not downloaded yet (the free
    weather API has an hourly quota) the fallback is the one year of actual weather already in the store, smoothed,
    with the missing September days interpolated between August and October."""
    cols = ["temp_max", "temp_min", "rain_mm", "humidity"]
    have = pd.read_parquet(cfg.NORMALS_PARQUET) if cfg.NORMALS_PARQUET.exists() else pd.DataFrame(columns=["state", "doy"] + cols)
    parts = [have[["state", "doy"] + cols]]
    missing = sorted(set(weather["state"]) - set(have["state"]))
    for state in missing:
        w = weather[weather["state"] == state].assign(doy=lambda x: x["date"].dt.dayofyear.clip(upper=365))
        n = w.groupby("doy")[cols].mean().reindex(range(1, 366))
        wrapped = pd.concat([n.iloc[-30:], n, n.iloc[:30]]).reset_index(drop=True)
        wrapped = wrapped.interpolate(limit_direction="both").rolling(15, center=True, min_periods=5).mean()
        n = wrapped.iloc[30:-30].set_axis(range(1, 366)).rename_axis("doy")
        parts.append(n.reset_index().assign(state=state))
    total = weather["state"].nunique()
    if not missing:
        note = "10-year climate normals (2016-2025) for every state."
    elif len(missing) == total:
        note = ("last year's actual weather, smoothed, as the 'normal' year (September interpolated). "
                "Run `python -m demand.weather` to replace it with 10-year normals.")
    else:
        note = (f"10-year normals for {total - len(missing)} states, last year's actual weather for the other {len(missing)}. "
                "Run `python -m demand.weather` again to finish the download.")
    return pd.concat(parts, ignore_index=True), note


def _panel(d: pd.DataFrame) -> pd.DataFrame:
    """Balanced state x category x week panel of units (zeros filled)."""
    weeks = A.complete_weeks(d)
    dd = d[(d["state"] != "UNKNOWN") & d["category"].ne("Other")].assign(week=A.week_start(d["date"]))
    dd = dd[dd["week"].isin(weeks)]
    idx = pd.MultiIndex.from_product([sorted(dd["state"].unique()), sorted(dd["category"].unique()), weeks], names=["state", "category", "week"])
    return dd.groupby(["state", "category", "week"])["quantity"].sum().reindex(idx, fill_value=0).rename("units").reset_index()


def _design(rows: pd.DataFrame, m: "Model | dict") -> pd.DataFrame:
    codes = m.codes if isinstance(m, Model) else m
    x = rows.copy()
    x["category_code"] = x["category"].map(codes["category"])
    x["state_code"] = x["state"].map(codes["state"])
    x["region_code"] = x["region"].map(codes["region"])
    x["weeks_to_diwali"] = _weeks_to_diwali(x["week"])
    return x


# ------------------------------------------------------------------ fit / predict
def fit(d: pd.DataFrame, weather: pd.DataFrame, until: pd.Timestamp | None = None, exclude_weeks=None) -> Model:
    panel = _panel(d)
    if until is not None:
        panel = panel[panel["week"] < until]
    if exclude_weeks is not None:
        panel = panel[~panel["week"].isin(exclude_weeks)]
    state_region = d.drop_duplicates("state").set_index("state")["region"]
    panel["region"] = panel["state"].map(state_region)
    codes = {k: {v: i for i, v in enumerate(sorted(panel[k].unique()))} for k in ("category", "state", "region")}

    base = np.log(panel.groupby(["state", "category"])["units"].mean() + 0.5).rename("base")
    x = _design(panel.merge(_weekly_weather(weather), on=["state", "week"]), codes).join(base, on=["state", "category"])
    # tiny state-category cells are almost all zeros: they add noise, not signal, and get the pooled pattern anyway
    train = x[x["base"] > np.log(1.5)]
    y = np.log(train["units"] + 0.5) - train["base"]
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                                        categorical_features=CATEGORICAL, random_state=0)
    gbm.fit(train[FEATURES], y, sample_weight=np.sqrt(np.exp(train["base"])))
    return Model(gbm, base, codes, state_region, panel["week"].max())


def _predict_weeks(m: Model, wx: pd.DataFrame) -> pd.DataFrame:
    """wx: state, week + weather columns  ->  state, category, region, week, units (structural, before calibration)."""
    return _predict_rows(m, wx.assign(weeks_to_diwali=_weeks_to_diwali(wx["week"])))


def _predict_rows(m: Model, wx: pd.DataFrame) -> pd.DataFrame:
    """wx must already carry weeks_to_diwali."""
    rows = m.base.reset_index().merge(wx, on="state")
    rows["region"] = rows["state"].map(m.state_region)
    for k in ("category", "state", "region"):
        rows[f"{k}_code"] = rows[k].map(m.codes[k])
    rows["units"] = (np.exp(rows["base"] + m.gbm.predict(rows[FEATURES])) - 0.5).clip(lower=0)
    return rows[["state", "category", "region", "week", "units"]]


def _to_months(weekly: pd.DataFrame, keys: list[str], value: str = "units") -> pd.DataFrame:
    """Split each week across the calendar months it touches, by days."""
    parts = []
    for i in range(7):
        day = weekly["week"] + pd.Timedelta(days=i)
        parts.append(weekly.assign(month=day.dt.to_period("M").astype(str), v=weekly[value] / 7))
    return pd.concat(parts).groupby(keys + ["month"])["v"].sum().rename(value).reset_index()


def predict(d: pd.DataFrame, weather: pd.DataFrame, normals: pd.DataFrame, until: str = "2028-12-31", scenario: str = "revert",
            growth_pct_per_year: float = 0.0, model: Model | None = None, level: str = "region") -> pd.DataFrame:
    """Monthly prediction by region x category (level="region") or region x state x category (level="state").
    Columns: region, [state,] category, month, units, revenue."""
    m = model or fit(d, weather)
    future = pd.date_range(m.last_week + pd.Timedelta(weeks=1), pd.Timestamp(until), freq="7D")
    fut = _predict_weeks(m, _normal_weather(normals, future))

    # calibration: how the last 12 weeks actually ran versus what the model says those weeks "should" have been
    recent = pd.date_range(m.last_week - pd.Timedelta(weeks=11), m.last_week, freq="7D")
    fitted = _predict_weeks(m, _weekly_weather(weather)[lambda w: w["week"].isin(recent)]).groupby(["region", "category"])["units"].sum()
    actual = _panel(d)[lambda p: p["week"].isin(recent)].assign(region=lambda p: p["state"].map(m.state_region)).groupby(["region", "category"])["units"].sum()
    ratio = ((actual + 20) / (fitted + 20)).clip(0.4, 2.5).rename("ratio")          # +20 shrinks thin cells towards 1
    fut = fut.join(ratio, on=["region", "category"]).fillna({"ratio": 1.0})
    h = ((fut["week"] - m.last_week).dt.days / 7).to_numpy()
    weight = np.ones_like(h) if scenario == "persist" else 0.5 ** (h / 26)           # run-rate gap halves every 26 weeks
    fut["units"] = fut["units"] * fut["ratio"] ** weight * (1 + growth_pct_per_year / 100) ** (h / 52)

    keys = ["region", "state", "category"] if level == "state" else ["region", "category"]
    out = _to_months(fut.groupby(keys + ["week"])["units"].sum().reset_index(), keys)
    out = out[out["month"] > d["date"].max().to_period("M").strftime("%Y-%m")]
    asp = d[d["date"] > d["date"].max() - pd.Timedelta(weeks=12)].groupby("category").agg(r=("revenue", "sum"), q=("quantity", "sum"))
    out["revenue"] = (out["units"] * out["category"].map(asp["r"] / asp["q"])).round(0)
    return out.assign(units=out["units"].round(1))


# ------------------------------------------------------------------ demand calendar: what is in demand, where, in which month
PLACE_LEVELS = {"All India": None, "Region": "region", "State": "state", "District": "district"}


def demand_calendar(d: pd.DataFrame, pred_state: pd.DataFrame, level: str | None, place: str | None,
                    item: str = "category", top: int = 5) -> pd.DataFrame:
    """Top `item`s (category or main_sku) per month for one place: actual months from the orders, future months from
    the ML prediction. Long table: month, kind ('actual' | 'predicted'), item, units, share_pct, rank.

    The model predicts state x category. Finer cuts are allocated from it and are labelled as such in the UI:
      district  = state prediction x the district's historical share of that state, category by category
      main_sku  = category prediction x the SKU's share of that category in this place over the last 12 weeks,
                  counting only SKUs sold in the last 28 days (a discontinued SKU is not "in demand next March")."""
    here = d if level is None else d[d[level] == place]
    if here.empty:
        return pd.DataFrame()
    hist = here.assign(month=here["date"].dt.strftime("%Y-%m")).groupby(["month", item])["quantity"].sum().rename("units").reset_index()

    # category prediction for this place
    if level == "district":
        state = here["state"].mode().iloc[0]
        in_state = d[d["state"] == state]
        share = (here.groupby("category")["quantity"].sum() / in_state.groupby("category")["quantity"].sum()).fillna(0)
        overall = here["quantity"].sum() / in_state["quantity"].sum()
        thin = in_state.groupby("category")["quantity"].sum() < 30           # too few units for a category-level share
        share = share.where(~thin.reindex(share.index, fill_value=True), overall)
        p = pred_state[pred_state["state"] == state].copy()
        p["units"] = p["units"] * p["category"].map(share).fillna(overall)
    else:
        p = pred_state if level is None else pred_state[pred_state[level] == place]
    fut = p.groupby(["month", "category"])["units"].sum().reset_index()

    if item != "category":
        end = d["date"].max()
        recent = here[here["date"] > end - pd.Timedelta(weeks=12)]
        if recent["quantity"].sum() < 200:                                    # a thin place borrows its state's SKU mix
            recent = d[(d["state"].isin(here["state"].unique())) & (d["date"] > end - pd.Timedelta(weeks=12))]
        alive = recent.loc[recent["date"] > end - pd.Timedelta(days=28), item].unique()
        mix = recent[recent[item].isin(alive)].groupby(["category", item])["quantity"].sum()
        mix = (mix / mix.groupby("category").transform("sum")).rename("mix").reset_index()
        fut = fut.merge(mix, on="category").assign(units=lambda x: x["units"] * x["mix"])[["month", item, "units"]]

    out = pd.concat([hist.assign(kind="actual"), fut.assign(kind="predicted")], ignore_index=True)
    out["share_pct"] = (out["units"] / out.groupby("month")["units"].transform("sum") * 100).round(1)
    out["rank"] = out.groupby("month")["units"].rank(method="first", ascending=False).astype(int)
    out = out[out["rank"] <= top].sort_values(["month", "rank"]).rename(columns={item: "item"})
    return out.assign(units=out["units"].round(0))[["month", "kind", "rank", "item", "units", "share_pct"]]


def place_monthly(d: pd.DataFrame, pred_state: pd.DataFrame, level: str | None, place: str | None) -> pd.DataFrame:
    """month x category units for one place, actual then predicted (same allocation rules as demand_calendar)."""
    cal = demand_calendar(d, pred_state, level, place, "category", top=99)
    if cal.empty:
        return cal
    return cal.pivot_table(index="month", columns="item", values="units", aggfunc="sum", fill_value=0).join(
        cal.drop_duplicates("month").set_index("month")["kind"])


# ------------------------------------------------------------------ validation and explanation
def validate(d: pd.DataFrame, weather: pd.DataFrame, holdout_weeks: int = 12) -> dict:
    """Train without the last `holdout_weeks`, predict them from their real weather, and compare with two naive rules."""
    weeks = A.complete_weeks(d)
    cut = weeks[-holdout_weeks]
    m = fit(d, weather, until=cut)
    wx = _weekly_weather(weather)
    test_wx = wx[wx["week"] >= cut]
    pred = _predict_weeks(m, test_wx)
    # same run-rate calibration the live prediction uses, from the 12 weeks before the cut
    recent = weeks[-holdout_weeks - 12:-holdout_weeks]
    panel = _panel(d).assign(region=lambda p: p["state"].map(m.state_region))
    fitted = _predict_weeks(m, wx[wx["week"].isin(recent)]).groupby(["region", "category"])["units"].sum()
    actual_recent = panel[panel["week"].isin(recent)].groupby(["region", "category"])["units"].sum()
    ratio = ((actual_recent + 20) / (fitted + 20)).clip(0.4, 2.5).rename("ratio")
    pred = pred.join(ratio, on=["region", "category"]).fillna({"ratio": 1.0})
    hh = ((pred["week"] - (cut - pd.Timedelta(weeks=1))).dt.days / 7).to_numpy()
    pred["ml"] = pred["units"] * pred["ratio"] ** (0.5 ** (hh / 26))

    test = panel[panel["week"] >= cut].merge(pred[["state", "category", "week", "ml"]], on=["state", "category", "week"])
    train = panel[panel["week"] < cut]
    test = test.join(train.groupby(["state", "category"])["units"].mean().rename("naive_mean"), on=["state", "category"])
    test = test.join(train[train["week"].isin(recent)].groupby(["state", "category"])["units"].mean().rename("naive_recent"), on=["state", "category"])

    def wape(keys):
        g = test.groupby(keys)[["units", "ml", "naive_mean", "naive_recent"]].sum()
        return {c: round(float((g[c] - g["units"]).abs().sum() / g["units"].sum() * 100), 1) for c in ("ml", "naive_mean", "naive_recent")}

    test["month"] = test["week"].dt.to_period("M").astype(str)
    table = pd.DataFrame({"total, by week": wape(["week"]), "region x week": wape(["region", "week"]), "region x month": wape(["region", "month"]),
                          "category x month": wape(["category", "month"]), "region x category x month": wape(["region", "category", "month"])}).T
    table.columns = ["ML model", "naive: full-history average", "naive: last 12 weeks' average"]
    by_region = test.groupby("region")[["units", "ml"]].sum().assign(error_pct=lambda g: ((g["ml"] / g["units"] - 1) * 100).round(1))
    return {"holdout": f"{cut.date()} to {(weeks[-1] + pd.Timedelta(days=6)).date()}", "wape": table, "by_region": by_region.round(0)}


def validate_blocked(d: pd.DataFrame, weather: pd.DataFrame, block: int = 4, folds: int = 4) -> pd.DataFrame:
    """Hold out 4-week blocks spread through the year (every `folds`-th block), predict them from their weather and
    festival position, and compare with the flat average. The end-of-history holdout in validate() asks the model to
    predict a monsoon it has never seen; this asks the fairer question for a climate-driven model: given conditions
    inside the range it has seen, does it explain demand better than "every week is average"?"""
    weeks = A.complete_weeks(d)
    wx = _weekly_weather(weather)
    panel = _panel(d).assign(region=lambda p: p["state"].map(d.drop_duplicates("state").set_index("state")["region"]))
    fold_of = pd.Series((np.arange(len(weeks)) // block) % folds, index=weeks)
    out = []
    for k in range(folds):
        held = fold_of.index[fold_of == k]
        m = fit(d, weather, exclude_weeks=held)
        pred = _predict_weeks(m, wx[wx["week"].isin(held)]).rename(columns={"units": "ml"})
        test = panel[panel["week"].isin(held)].merge(pred[["state", "category", "week", "ml"]], on=["state", "category", "week"])
        naive = panel[~panel["week"].isin(held)].groupby(["state", "category"])["units"].mean().rename("naive_mean")
        out.append(test.join(naive, on=["state", "category"]))
    test = pd.concat(out)
    test["month"] = test["week"].dt.to_period("M").astype(str)

    def wape(keys):
        g = test.groupby(keys)[["units", "ml", "naive_mean"]].sum()
        return {c: round(float((g[c] - g["units"]).abs().sum() / g["units"].sum() * 100), 1) for c in ("ml", "naive_mean")}

    t = pd.DataFrame({"total, by week": wape(["week"]), "region x week": wape(["region", "week"]), "region x month": wape(["region", "month"]),
                      "category x month": wape(["category", "month"]), "region x category x month": wape(["region", "category", "month"])}).T
    t.columns = ["ML model", "naive: flat average"]
    t["error removed by ML %"] = ((1 - t["ML model"] / t["naive: flat average"]) * 100).round(0)
    return t


def drivers(d: pd.DataFrame, weather: pd.DataFrame, normals: pd.DataFrame, model: Model | None = None) -> dict:
    """What the model has learned, as plain tables: the demand index of each category across the temperature range,
    by rain, and around Diwali (100 = the category's normal level, everything else held at typical values)."""
    m = model or fit(d, weather)
    cats = d.groupby("category")["quantity"].sum().sort_values(ascending=False).index.intersection(list(m.codes["category"]))
    big = d.groupby("state")["quantity"].sum().nlargest(8).index
    wx = _weekly_weather(weather)
    typical = wx[wx["state"].isin(big)].groupby("state")[["temp_max", "temp_min", "rain_mm", "humidity"]].median().reset_index()
    typical = typical.assign(week=pd.Timestamp("2026-05-04"), weeks_to_diwali=float(NO_FESTIVAL))

    def response(settings: dict) -> pd.DataFrame:
        t = pd.DataFrame({label: _predict_rows(m, rows).groupby("category")["units"].sum() for label, rows in settings.items()})
        return t.loc[cats]

    temp = response({f"{v}°C": typical.assign(temp_min=typical["temp_min"] + (v - typical["temp_max"]), temp_max=float(v)) for v in (20, 24, 28, 32, 36, 40)})
    rain = response({f"{v} mm/wk": typical.assign(rain_mm=float(v)) for v in (0, 10, 40, 100)})
    fest = response({("Diwali wk" if k == 0 else f"{k:+d} wk"): typical.assign(weeks_to_diwali=float(k)) for k in range(FESTIVE_WINDOW[0], FESTIVE_WINDOW[1] + 1)})
    normal = response({"n": typical})["n"]
    return {"temperature": (temp.div(temp.mean(axis=1), axis=0) * 100).round(0), "rain": (rain.div(rain["0 mm/wk"], axis=0) * 100).round(0),
            "diwali": (fest.div(normal, axis=0) * 100).round(0)}


# ------------------------------------------------------------------ plain-language outlook (works without the Claude key)
def outlook_text(pred: pd.DataFrame, d: pd.DataFrame, growth: float, weather_note: str) -> str:
    """Deterministic written summary of the prediction, region by region."""
    pred = pred.assign(year=pred["month"].str[:4])
    hist = d.assign(month=d["date"].dt.strftime("%Y-%m"))
    hist_monthly = hist.groupby("month")["quantity"].sum()
    hist_monthly = hist_monthly[hist_monthly.index < hist["month"].max()] if d["date"].max().day < 25 else hist_monthly
    tot = pred.groupby("month")["units"].sum()
    years = pred.groupby("year")["units"].sum()
    full_years = [y for y, n in pred.groupby("year")["month"].nunique().items() if n == 12]
    lines = [f"**Assumptions:** underlying growth {growth:+.0f}% a year; future weather = {weather_note} "
             f"Festive pattern as observed in 2025 (sale opened four weeks before Diwali week), moved to each year's Diwali date.",
             f"**All India:** history averaged {hist_monthly.mean():,.0f} units a month (best {hist_monthly.idxmax()} at {hist_monthly.max():,.0f}). "
             + " ".join(f"{y}: {years[y]:,.0f} units ({years[y] / 12:,.0f}/month)." for y in full_years)
             + f" Predicted peak month {tot.idxmax()} ({tot.max():,.0f}), weakest {tot.idxmin()} ({tot.min():,.0f})."]
    for region, g in pred.groupby("region"):
        if region == "Unknown":
            continue
        mt = g.groupby("month")["units"].sum()
        if not full_years:
            continue
        y = full_years[0]
        gy = g[g["year"] == y]
        cat = gy.groupby("category")["units"].sum().sort_values(ascending=False)
        my = gy.groupby("month")["units"].sum()
        swing = my.max() / max(my.min(), 1)
        share = gy["units"].sum() / pred[pred["year"] == y]["units"].sum() * 100
        lines.append(f"**{region}** ({share:.0f}% of units): {y} predicted {gy['units'].sum():,.0f} units. Strongest {pd.Period(my.idxmax()).strftime('%b')} "
                     f"({my.max():,.0f}), weakest {pd.Period(my.idxmin()).strftime('%b')} ({my.min():,.0f}), a {swing:.1f}x seasonal swing. "
                     f"Mix: {', '.join(f'{c} {v / cat.sum() * 100:.0f}%' for c, v in cat.head(3).items())}.")
    lines.append("**How much to trust it:** the first 3-6 months rest on the recent run-rate and are the most reliable. Beyond a year this is a "
                 "*theoretical* projection: the seasonal shape comes from weather and one Diwali, and growth is whatever you assume above. "
                 "It will sharpen materially once September-November 2026 actuals are ingested (first repeat of a month and of a festive season).")
    return "\n\n".join(lines)
