"""Daily historical weather per state from the Open-Meteo archive API (no key needed).

One point per state (its largest demand centre, see geo.STATES). The archive lags real time
by ~5 days, so the newest few days of sales simply have no weather yet. Open-Meteo's free
tier is for non-commercial use; for production, buy their API plan or swap in IMD data here.
"""
import time

import pandas as pd
import requests

from . import config as cfg
from .geo import STATES

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
         "precipitation_sum", "relative_humidity_2m_mean"]
RENAME = {"temperature_2m_max": "temp_max", "temperature_2m_min": "temp_min", "temperature_2m_mean": "temp_mean",
          "precipitation_sum": "rain_mm", "relative_humidity_2m_mean": "humidity"}


class QuotaExceeded(RuntimeError):
    """The free tier's hourly / daily allowance is used up. Everything fetched so far is saved; run again later."""


def _fetch(state: str, start: str, end: str) -> pd.DataFrame:
    lat, lon = STATES[state]
    for attempt in range(4):
        r = requests.get(API, timeout=60, params={
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "daily": ",".join(DAILY), "timezone": cfg.TIMEZONE})
        if r.status_code == 429:
            if "hourly" in r.text.lower() or "daily" in r.text.lower():
                raise QuotaExceeded(r.json().get("reason", "Open-Meteo quota exceeded"))
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        d = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date", **RENAME})
        d["date"] = pd.to_datetime(d["date"])
        d["state"] = state
        return d.dropna(subset=["temp_max"])
    raise RuntimeError(f"Open-Meteo rate limit hit repeatedly for {state}")


def update_weather(log=print) -> pd.DataFrame:
    """Fetch whatever weather is missing for the date span of the order store."""
    orders = pd.read_parquet(cfg.ORDERS_PARQUET, columns=["date", "state", "is_demand"])
    orders = orders[orders["is_demand"]]
    start, end = orders["date"].min(), min(orders["date"].max(), pd.Timestamp.today().normalize() - pd.Timedelta(days=6))
    have = pd.read_parquet(cfg.WEATHER_PARQUET) if cfg.WEATHER_PARQUET.exists() else pd.DataFrame(columns=["date", "state"])

    frames = [have]
    for state in sorted(set(orders["state"]) & set(STATES)):
        got = have.loc[have["state"] == state, "date"]
        if len(got) and got.min() <= start and got.max() >= end:
            continue
        frm = start if not len(got) or got.min() > start else got.max() + pd.Timedelta(days=1)
        log(f"weather: {state} {frm.date()} to {end.date()}")
        try:
            frames.append(_fetch(state, str(frm.date()), str(end.date())))
        except QuotaExceeded as e:
            log(f"  stopped: {e}")
            break
        except (requests.RequestException, RuntimeError) as e:
            log(f"  failed ({e}); will retry on next run")
        time.sleep(0.4)

    weather = (pd.concat(frames, ignore_index=True).drop_duplicates(["state", "date"], keep="last")
                 .sort_values(["state", "date"]).reset_index(drop=True))
    weather.to_parquet(cfg.WEATHER_PARQUET, index=False)
    return weather


def update_normals(first_year: int = 2016, last_year: int = 2025, log=print) -> pd.DataFrame:
    """Climate normals per state and day-of-year (multi-year mean, smoothed). They stand in for the weather of
    future months, which nobody knows, when the long-range model predicts demand."""
    have = pd.read_parquet(cfg.NORMALS_PARQUET) if cfg.NORMALS_PARQUET.exists() else pd.DataFrame(columns=["state"])
    frames = [have]
    for state in sorted(set(STATES) - set(have["state"])):
        log(f"climate normals: {state} {first_year}-{last_year}")
        try:
            d = _fetch(state, f"{first_year}-01-01", f"{last_year}-09-30")
        except QuotaExceeded as e:
            log(f"  stopped: {e}. {len(frames) - 1} states saved this run; run again later to fetch the rest.")
            break
        except (requests.RequestException, RuntimeError) as e:
            log(f"  failed ({e}); will retry on next run")
            continue
        d["doy"] = d["date"].dt.dayofyear.clip(upper=365)
        n = d.groupby("doy")[["temp_max", "temp_min", "temp_mean", "rain_mm", "humidity"]].mean()
        # wrap-around 15-day smoothing: a single wet day ten years ago should not look like a yearly event
        wrapped = pd.concat([n.iloc[-15:], n, n.iloc[:15]])
        n = wrapped.rolling(15, center=True, min_periods=8).mean().iloc[15:-15]
        frames.append(n.reset_index().assign(state=state))
        pd.concat(frames, ignore_index=True).to_parquet(cfg.NORMALS_PARQUET, index=False)      # keep progress if interrupted
        time.sleep(15)      # a 10-year pull weighs ~130 calls against Open-Meteo's 600 / minute and 5,000 / hour free allowance
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    update_weather()
    update_normals()
