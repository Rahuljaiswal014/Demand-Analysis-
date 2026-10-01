"""Ingest Amazon Seller Central "All Orders" reports into a cleaned parquet store.

Usage:  python -m demand.ingest            (scan data/raw and the project root for new files)
        python -m demand.ingest --rebuild  (ignore the manifest and rebuild from every file)

Re-dropping a month is safe: rows are keyed on order id + order item id + sku and the
version with the latest `last-updated-date` wins, so status changes are picked up.
"""
import argparse
import json
import re
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .geo import STATE_TO_REGION, add_district, normalise_states

REQUIRED = ["amazon-order-id", "purchase-date", "last-updated-date", "order-status", "sales-channel",
            "fulfillment-channel", "product-name", "sku", "asin", "quantity", "item-price",
            "ship-city", "ship-state", "ship-postal-code", "order-item-id"]
TEXT_COLS = {"sku": str, "asin": str, "ship-postal-code": str, "order-item-id": str,
             "amazon-order-id": str, "merchant-order-id": str}

# First match wins, so order encodes the traps found when auditing titles against SKU codes:
#   fitted bedsheets say "elastic mattress cover" and "with 2 pillow covers"   -> bedsheet before protector-by-"cover" and pillow
#   bolsters are sold as "bolster round cushion covers" / "round pillow cover"  -> bolster before cushion and pillow
#   sofa covers say "quilted" and "sofa cover mats"                             -> sofa cover before comforter and rug
#   throws say "knitted blanket" and "for sofa, couch, bed, chair"              -> throw before comforter
#   duvet covers say "comforter cover | quilt cover | blanket cover"            -> duvet cover before comforter
CATEGORY_RULES = [
    ("Curtain", r"curtain|drape"),
    ("Mattress Protector", r"mattress (?:cover/)?protector|bed protector"),
    ("Bedsheet", r"bedsheet|bed sheet|fitted sheet|flat sheet"),
    ("Sofa/Chair Cover", r"sofa cover|chair cover"),
    ("Bolster Cover", r"bolster"),
    ("Cushion Cover", r"cushion"),
    ("Throw", r"\bthrows?\b"),
    ("Duvet Cover", r"duvet cover|quilt cover|rajai cover|comforter cover"),
    ("Comforter/Quilt", r"comforter|quilt|dohar|blanket"),
    ("Pillow Cover", r"pillow"),
    ("Table Linen", r"table|placemat|napkin"),
    ("Rug/Carpet/Mat", r"\brugs?\b|carpet|\bmats?\b"),
]
# The SKU stem is HomeMonde's own product coding. Where it is unambiguous it overrides the title, which Amazon
# truncates at ~150 characters and which is sometimes just "-".
PREFIX_CATEGORY = {"HMBL": "Bolster Cover", "HMTH": "Throw", "HMSC": "Sofa/Chair Cover", "HMFT": "Bedsheet", "HMBD": "Bedsheet",
                   "HMMP": "Mattress Protector", "HMDC": "Duvet Cover", "HMCC": "Cushion Cover", "HMTC": "Table Linen",
                   "HMRG": "Rug/Carpet/Mat", "HMCR": "Curtain", "RCR": "Curtain", "HMBK": "Comforter/Quilt", "HMCF": "Comforter/Quilt",
                   "HMDH": "Comforter/Quilt", "BDHR": "Comforter/Quilt", "BBKT": "Comforter/Quilt"}


def categorise(title: str, sku: str = "") -> str:
    m = re.match(r"[A-Za-z]+", sku or "")
    if m and m.group(0).upper() in PREFIX_CATEGORY:
        return PREFIX_CATEGORY[m.group(0).upper()]
    t = str(title).lower()
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, t):
            return cat
    return "Other"


def _copy_locked(src: Path, folder: Path) -> Path:
    """Copy a workbook that Excel holds open. Excel allows other readers only if they also agree to share the file for
    writing; Python's open() does not ask for that, so the file is opened through the Windows API with full sharing."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    create = ctypes.windll.kernel32.CreateFileW
    create.restype = wintypes.HANDLE
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    GENERIC_READ, SHARE_ALL, OPEN_EXISTING = 0x80000000, 0x1 | 0x2 | 0x4, 3
    handle = create(str(src), GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise PermissionError(f"{src.name} is open in another program and could not be read. Close it and run again.")
    dst = folder / src.name
    with os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY), "rb") as fin, open(dst, "wb") as fout:
        while chunk := fin.read(1 << 22):
            fout.write(chunk)
    return dst


def read_report(path: Path) -> pd.DataFrame:
    """Read one report file: .xlsx (every sheet), or Amazon's native tab-delimited .txt/.tsv, or .csv."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            frames = list(pd.read_excel(path, sheet_name=None, dtype=TEXT_COLS).values())
        except PermissionError:                       # the workbook is open in Excel: read a temporary copy of the saved file
            with tempfile.TemporaryDirectory() as tmp:
                frames = list(pd.read_excel(_copy_locked(path, Path(tmp)), sheet_name=None, dtype=TEXT_COLS).values())
    elif suffix in (".txt", ".tsv"):
        frames = [pd.read_csv(path, sep="\t", dtype=TEXT_COLS, encoding="utf-8", encoding_errors="replace")]
    elif suffix == ".csv":
        frames = [pd.read_csv(path, dtype=TEXT_COLS, encoding="utf-8", encoding_errors="replace")]
    else:
        raise ValueError(f"Unsupported file type: {path.name}")
    out = []
    for f in frames:
        if f.empty:
            continue
        f.columns = [str(c).strip() for c in f.columns]   # 'is-prime ' ships with a trailing space
        missing = [c for c in REQUIRED if c not in f.columns]
        if missing:
            raise ValueError(f"{path.name}: not an All Orders report, missing columns {missing}")
        out.append(f)
    df = pd.concat(out, ignore_index=True)
    df["source_file"] = path.name
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw report rows -> analysis-ready order lines."""
    df = pd.DataFrame({
        "order_id": raw["amazon-order-id"].astype("string"),
        "order_item_id": raw["order-item-id"].astype("string"),
        "sku": raw["sku"].astype("string").str.strip(),
        "asin": raw["asin"].astype("string").str.strip(),
        "product_name": raw["product-name"].astype("string"),
        "order_status": raw["order-status"].astype("string"),
        "sales_channel": raw["sales-channel"].astype("string"),
        "fulfillment": raw["fulfillment-channel"].astype("string"),
        "quantity": pd.to_numeric(raw["quantity"], errors="coerce").fillna(0).astype("int64"),
        "item_price": pd.to_numeric(raw["item-price"], errors="coerce"),
        "promo_discount": pd.to_numeric(raw.get("item-promotion-discount"), errors="coerce").fillna(0.0),
        "has_promo": raw.get("promotion-ids", pd.Series(index=raw.index, dtype="object")).notna(),
        "is_business": raw.get("is-business-order", pd.Series(False, index=raw.index)).astype(bool),
        "city": raw["ship-city"].astype("string").str.upper().str.strip(),
        "pin": raw["ship-postal-code"].astype("string").str.extract(r"(\d{6})")[0],
        "source_file": raw["source_file"].astype("string"),
    })
    # timestamps arrive in UTC; every business cut (day, week, month) is on IST
    ts = pd.to_datetime(raw["purchase-date"], utc=True, errors="coerce").dt.tz_convert(cfg.TIMEZONE)
    df["ts"] = ts.dt.tz_localize(None)
    df["date"] = df["ts"].dt.normalize()
    df["last_updated"] = pd.to_datetime(raw["last-updated-date"], utc=True, errors="coerce").dt.tz_localize(None)

    df["state"] = normalise_states(raw["ship-state"].astype("string"), df["pin"])
    df["region"] = df["state"].map(STATE_TO_REGION).fillna("Unknown")

    df["is_cancelled"] = df["order_status"].eq("Cancelled")
    # 'Non-Amazon' rows are Multi-Channel Fulfilment shipments (bulk transfers, website orders), not marketplace demand
    df["is_amazon"] = df["sales_channel"].eq("Amazon.in")
    df["is_demand"] = df["is_amazon"] & ~df["is_cancelled"] & df["quantity"].gt(0)
    df["revenue"] = df["item_price"].fillna(0.0).where(df["is_demand"], 0.0)
    # zero-priced shipped lines are replacements/free units: they count as units but not as a price point
    df["unit_price"] = np.where(df["is_demand"] & df["item_price"].gt(0), df["item_price"] / df["quantity"].clip(lower=1), np.nan)
    return df.dropna(subset=["ts", "asin"])


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    key = ["order_id", "order_item_id", "sku"]
    return (df.sort_values("last_updated").drop_duplicates(key, keep="last")
              .sort_values("ts").reset_index(drop=True))


def _family(sku: str) -> str:
    m = re.match(r"^([A-Za-z]+\d*[-_ ]?\d+)", sku or "")
    return (m.group(1) if m else (sku or "").split("-")[0]).upper().replace("_", "-").replace(" ", "")


def build_products(orders: pd.DataFrame) -> pd.DataFrame:
    """One row per ASIN. SKUs are aliases (…%-OOS, -NEW, -S2) of the same listing, so ASIN is the product key."""
    def mode(s):
        return s.value_counts().index[0]

    g = orders.groupby("asin")
    p = pd.DataFrame({
        "title": g["product_name"].agg(mode),
        "main_sku": g["sku"].agg(mode),
        "n_skus": g["sku"].nunique(),
        "skus": g["sku"].agg(lambda s: ", ".join(sorted(s.unique()))),
    })
    p["title"] = p["title"].str.replace(r"^HOMEMONDE\s+", "", regex=True)
    p["category"] = [categorise(t, k) for t, k in zip(p["title"], p["main_sku"])]
    p["family"] = p["main_sku"].map(_family)

    t = p["title"].str.lower()
    is_curtain = p["category"].eq("Curtain")
    # the lookbehind skips widths such as "4.5 Ft"; the SKU code ("-7FT", "7 FEET") fills in when the title has no length
    sku_l = p["main_sku"].str.lower()
    length = pd.to_numeric(t.str.extract(r"(?<![\d.])(\d{1,2})\s*(?:feet|ft|foot)\b")[0], errors="coerce")
    length_sku = pd.to_numeric(sku_l.str.extract(r"(\d{1,2})\s*(?:feet|ft)")[0], errors="coerce")
    p["curtain_length_ft"] = length.fillna(length_sku).where(is_curtain)
    p["length_title_vs_sku_conflict"] = is_curtain & length.notna() & length_sku.notna() & (length != length_sku)
    p["is_custom_size"] = is_curtain & sku_l.str.contains("cstm")
    p["curtain_type"] = np.select(
        [is_curtain & t.str.contains("blackout|room darkening|opaque"),
         is_curtain & t.str.contains("sheer|light filtering|transparent")],
        ["Blackout", "Sheer"], default=None)
    p.loc[is_curtain & p["curtain_type"].isna(), "curtain_type"] = "Other curtain"
    # titles are cut at ~150 characters, often right before "Pack of 2", so the SKU suffix (-S1, -S2, -S4) is the second source
    pack = (t.str.extract(r"(?:set|pack|pair)\s*of\s*(\d{1,2})")[0]
             .fillna(t.str.extract(r"\b(\d{1,2})[\s-]*(?:pieces?|pcs|pc|panels?)\b")[0])
             .fillna(t.str.extract(r"\b(single|one)\s*(?:pcs|piece|panel|curtain)")[0].map({"single": "1", "one": "1"}))
             .fillna(p["main_sku"].str.upper().str.extract(r"-S(\d{1,2})(?:-|$)")[0]))
    p["pack_size"] = pd.to_numeric(pack, errors="coerce")

    d = orders[orders["is_demand"]].groupby("asin")
    p["first_sale"] = d["date"].min()
    p["last_sale"] = d["date"].max()
    return p.reset_index()


def _find_files() -> list[Path]:
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    pats = ("*.xlsx", "*.xlsm", "*.txt", "*.tsv", "*.csv")
    files = [f for pat in pats for f in cfg.RAW_DIR.glob(pat)]
    files += [f for f in cfg.ROOT.glob("*.xlsx")]          # the original workbook lives in the project root
    return sorted(f for f in files if not f.name.startswith("~$"))


def ingest(rebuild: bool = False, log=print, refresh_products: bool = False) -> dict:
    cfg.STORE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {} if rebuild or not cfg.MANIFEST_JSON.exists() else json.loads(cfg.MANIFEST_JSON.read_text())
    existing = None if rebuild or not cfg.ORDERS_PARQUET.exists() else pd.read_parquet(cfg.ORDERS_PARQUET)

    new_frames, skipped = [], []
    for f in _find_files():
        sig = {"size": f.stat().st_size, "mtime": int(f.stat().st_mtime)}
        if manifest.get(f.name) == sig:
            skipped.append(f.name)
            continue
        log(f"reading {f.name} ...")
        try:
            new_frames.append(clean(read_report(f)))
        except ValueError as e:
            log(f"  skipped: {e}")
            continue
        manifest[f.name] = sig

    if not new_frames and existing is None:
        raise SystemExit(f"No report files found. Drop Amazon 'All Orders' reports into {cfg.RAW_DIR}")
    if new_frames or not cfg.PRODUCTS_PARQUET.exists() or refresh_products:
        parts = ([existing] if existing is not None else []) + new_frames
        orders = dedupe(pd.concat(parts, ignore_index=True))
        products = build_products(orders)
        orders = orders.drop(columns=[c for c in ("category", "family", "district", "district_source") if c in orders.columns])
        orders = orders.merge(products[["asin", "category", "family"]], on="asin", how="left")
        orders = pd.concat([orders, add_district(orders["pin"], orders["state"])], axis=1)
        orders.to_parquet(cfg.ORDERS_PARQUET, index=False)
        products.to_parquet(cfg.PRODUCTS_PARQUET, index=False)
        cfg.MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    else:
        orders = existing

    d = orders[orders["is_demand"]]
    summary = {
        "files_ingested": [f.name for f in _find_files() if f.name not in skipped],
        "files_unchanged": skipped,
        "rows": len(orders), "demand_rows": len(d), "units": int(d["quantity"].sum()),
        "asins": int(d["asin"].nunique()), "from": str(d["date"].min().date()), "to": str(d["date"].max().date()),
        "unknown_state_rows": int(orders["state"].eq("UNKNOWN").sum()),
        "districts": int(d["district"].nunique()), "rows_district_from_nearby_pin": int(d["district_source"].eq("nearby PIN").sum()),
        "rows_without_district": int(d["district_source"].eq("none").sum()),
    }
    log(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="re-read every file from scratch")
    ap.add_argument("--refresh-products", action="store_true", help="re-derive categories / attributes after changing the rules")
    args = ap.parse_args()
    ingest(rebuild=args.rebuild, refresh_products=args.refresh_products)
    from .weather import update_weather
    update_weather()
    from .fabric import update_fabric
    update_fabric()
    from .sourcing import update_outsource
    update_outsource()
