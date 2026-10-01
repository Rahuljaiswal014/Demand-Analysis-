"""Weekly demand forecast for any grouping (total, category, region, SKU ...).

Method, chosen for what a single year of history can and cannot support:
  * Baseline: damped-trend exponential smoothing (Holt) on log weekly units, spike weeks winsorised, parameters
    picked per series by one-step-ahead error. An 80% band comes from the residual spread, widening with horizon.
  * Thin series (few recent units) are not modelled on their own: they get their recent share of the total forecast.
  * Festive scenario: yearly seasonality cannot be learned from a single cycle, but the whole 2025 festive period
    (sale opening 22 Sep, Diwali 20 Oct) *is* in the data. Weeks are aligned by their distance from Diwali (which moves ~3 weeks between years), and each
    future week inherits last year's uplift at the same distance, measured against the quiet Dec-Feb level.
    It is a scenario built from a single observation, shown next to the baseline, not blended into it.
"""
import itertools

import numpy as np
import pandas as pd

from . import analytics as A
from . import config as cfg

Z80 = 1.2816
FESTIVE_OFFSETS = range(-4, 4)          # 2025: sale opened 4 weeks before Diwali week; the dip lasted ~3 weeks after it
_GRID = list(itertools.product([0.1, 0.2, 0.35, 0.5, 0.7], [0.02, 0.1, 0.25], [0.7, 0.85]))   # strong damping: one noisy slope should not run for 12 weeks


def _holt_damped(y: np.ndarray, alpha: float, beta: float, phi: float, horizon: int):
    level, trend, errs = y[0], 0.0, []
    for obs in y[1:]:
        pred = level + phi * trend
        errs.append(obs - pred)
        new_level = alpha * obs + (1 - alpha) * pred
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    steps = np.cumsum(phi ** np.arange(1, horizon + 1))
    return level + steps * trend, np.asarray(errs)


def _winsorise(y: np.ndarray, cap: float = 1.5) -> np.ndarray:
    med = pd.Series(y).rolling(9, center=True, min_periods=4).median().to_numpy()
    return np.minimum(y, cap * np.maximum(med, 1))


def _fit(y: np.ndarray, horizon: int):
    ly = np.log1p(_winsorise(y.astype(float)))
    best = min((_holt_damped(ly, a, b, p, horizon) + ((a, b, p),) for a, b, p in _GRID),
               key=lambda r: np.mean(r[1][len(r[1]) // 3:] ** 2))          # score on the later two-thirds (after burn-in)
    path, errs, _ = best
    sigma = np.std(errs[len(errs) // 3:]) * np.sqrt(np.arange(1, horizon + 1))
    return np.expm1(path).clip(min=0), np.expm1(path - Z80 * sigma).clip(min=0), np.expm1(path + Z80 * sigma)


def _diwali_weeks() -> list[pd.Timestamp]:
    if not cfg.EVENTS_CSV.exists():
        return []
    ev = pd.read_csv(cfg.EVENTS_CSV, parse_dates=["date"])
    dates = ev.loc[ev["event"].str.lower().eq("diwali"), "date"]
    return sorted(d - pd.Timedelta(days=d.dayofweek) for d in dates)


def festive_multipliers(history: pd.Series, future_weeks: pd.DatetimeIndex) -> pd.Series:
    """Uplift per future week from the same distance-to-Diwali week one year earlier (1.0 where unknown)."""
    mult = pd.Series(1.0, index=future_weeks)
    diwalis = _diwali_weeks()
    past = [w for w in diwalis if history.index.min() <= w + pd.Timedelta(weeks=6) and w <= history.index.max()]
    ahead = [w for w in diwalis if w > history.index.max()]
    if not past or not ahead:
        return mult
    last, nxt = past[-1], ahead[0]
    quiet = history[(history.index >= last + pd.Timedelta(weeks=6)) & (history.index <= last + pd.Timedelta(weeks=18))]
    if len(quiet) < 6 or quiet.median() <= 0:
        return mult
    for week in future_weeks:
        k = round((week - nxt).days / 7)
        src = last + pd.Timedelta(weeks=k)
        if k in FESTIVE_OFFSETS and src in history.index:
            mult[week] = float(np.clip(history[src] / quiet.median(), 0.6, 3.0))
    return mult


def forecast(d: pd.DataFrame, by: str | None = None, horizon: int = 12, min_recent_units: int = 80) -> pd.DataFrame:
    """Long table: group, week, baseline, low, high, festive_multiplier, festive_scenario, method."""
    dd = d.assign(_all="All") if by is None else d.dropna(subset=[by])
    w = A.weekly_matrix(dd, by or "_all")
    if w.shape[1] < 16:
        raise ValueError("Need at least 16 complete weeks of history to forecast.")
    future = pd.date_range(w.columns[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="7D")
    total = w.sum()
    total_fc = _fit(total.to_numpy(), horizon)
    total_mult = festive_multipliers(total, future)

    recent = w.iloc[:, -8:].sum(axis=1)
    out = []
    for grp, row in w.iterrows():
        if recent[grp] >= min_recent_units:
            base, lo, hi = _fit(row.to_numpy(), horizon)
            mult, method = festive_multipliers(row, future) if row.sum() >= 5000 else total_mult, "own history"
        elif recent[grp] > 0:
            share = recent[grp] / recent.sum()
            base, lo, hi = (x * share for x in total_fc)
            mult, method = total_mult, "share of total"
        else:
            continue
        out.append(pd.DataFrame({"group": grp, "week": future, "baseline": base.round(1), "low": lo.round(1), "high": hi.round(1),
                                 "festive_multiplier": mult.to_numpy().round(2), "festive_scenario": (base * mult.to_numpy()).round(1),
                                 "method": method}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def rolling_backtest(d: pd.DataFrame, by: str | None = None, holdout: int = 8, origins: int = 4, step: int = 4) -> pd.DataFrame:
    """Backtest from several cut-off dates (one holdout can be lucky or unlucky) and report each plus the pooled score."""
    weeks = A.complete_weeks(d)
    rows = []
    for i in range(origins):
        end = weeks[-1 - i * step] + pd.Timedelta(days=7)
        bt = backtest(d[d["date"] < end], by, holdout).loc["ALL GROUPS"]
        rows.append({"holdout_weeks": f"{(end - pd.Timedelta(weeks=holdout)).date()} to {(end - pd.Timedelta(days=1)).date()}", **bt.to_dict()})
    out = pd.DataFrame(rows)
    pooled = {"holdout_weeks": "POOLED", "actual": out["actual"].sum(), "forecast": out["forecast"].sum()}
    for c in ("wape_pct", "period_total_error_pct", "inside_band_pct"):
        pooled[c] = round(float(np.average(out[c], weights=out["actual"])), 1)
    pooled["bias_pct"] = round((pooled["forecast"] / pooled["actual"] - 1) * 100, 1)
    return pd.concat([out, pd.DataFrame([pooled])], ignore_index=True)


def backtest(d: pd.DataFrame, by: str | None = None, holdout: int = 8) -> pd.DataFrame:
    """Hide the last `holdout` complete weeks, forecast them, and score. WAPE = sum|error| / sum(actual)."""
    weeks = A.complete_weeks(d)
    cut = weeks[-holdout]
    fc = forecast(d[d["date"] < cut], by, horizon=holdout)
    actual = A.weekly_matrix(d.assign(_all="All") if by is None else d.dropna(subset=[by]), by or "_all").iloc[:, -holdout:]
    actual = actual.stack().rename("actual").rename_axis(["group", "week"]).reset_index()
    m = fc.merge(actual, on=["group", "week"], how="left").fillna({"actual": 0})
    m["abs_err"] = (m["baseline"] - m["actual"]).abs()
    m["inside_band"] = m["actual"].between(m["low"], m["high"])
    g = m.groupby("group").agg(actual=("actual", "sum"), forecast=("baseline", "sum"), abs_err=("abs_err", "sum"), inside_band=("inside_band", "mean"))
    g["period_err"] = (g["forecast"] - g["actual"]).abs()
    g.loc["ALL GROUPS"] = [g["actual"].sum(), g["forecast"].sum(), g["abs_err"].sum(), m["inside_band"].mean(), g["period_err"].sum()]
    g["wape_pct"] = (g["abs_err"] / g["actual"].clip(lower=1) * 100).round(1)                 # week-by-week error
    g["period_total_error_pct"] = (g["period_err"] / g["actual"].clip(lower=1) * 100).round(1)  # error on the whole holdout total
    g["bias_pct"] = ((g["forecast"] / g["actual"].clip(lower=1) - 1) * 100).round(1)
    g["inside_band_pct"] = (g["inside_band"] * 100).round(0)
    return g[["actual", "forecast", "wape_pct", "period_total_error_pct", "bias_pct", "inside_band_pct"]].round(1).sort_values("actual", ascending=False)
