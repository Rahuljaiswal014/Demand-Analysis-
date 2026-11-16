"""Sales on the other marketplaces (Flipkart, Myntra, the HomeMonde website, Ajio, Blinkit), so fabric consumption can be
read per marketplace instead of for Amazon alone.

Source files go in data/raw/marketplaces/: each marketplace's own monthly sales export, recognised by its columns rather
than its name, plus the "Master SKU Sheet" workbook that maps every marketplace's seller SKU to HomeMonde's base SKU.
Amazon stays on its own pipeline (demand/ingest.py, the All Orders report); an Amazon report dropped here is pointed there.

What each report contributes (one cleaned row per order line, or per SKU x day x location for Flipkart's aggregated export):
    quantity          demand units = units ordered minus units cancelled; returned units stay in, as on Amazon, because the
                      goods were made and shipped (`returned_units` carries them separately)
    seller_sku        the SKU as the marketplace knows it; `base_sku` is the Master sheet's base SKU for it
    sku_match         how the base SKU was found: 'master sheet' / 'not in master' (the seller SKU is used as it is) /
                      'reference file' (Blinkit items, which the report names but does not give a SKU for; see
                      data/reference/marketplace_items.csv) / 'no SKU on report'
    state, district   where the marketplace says; Flipkart's and Ajio's exports carry no address, so theirs are UNKNOWN

Joining to fabric: a marketplace line is tied to the same product key the Amazon pipeline uses (the ASIN) whenever its
seller SKU, base SKU or any Master-sheet alias of the base is an Amazon SKU in the order history, so one product gets the
same fabric facts on every marketplace. Products that never sold on Amazon get a key of their own ('MP:<base SKU>') and a
product row built from the report (title, SKU, category, curtain length, pack size), and then go through exactly the same
fabric rules as Amazon listings (sheet by SKU, old list, size rule, outsourced, not in sheet).

Blinkit works differently: it knows HomeMonde's products by its own item id and UPC. Its "Order data" export carries the
HomeMonde SKU per item in a `Skus` column, and then Blinkit lines go the normal SKU route (Master sheet, Amazon listing, fabric
sheet). Its plain "Sales Summary" export has no SKU column and only names the item ("Homemonde Blackout Door Curtain
(Polyester, Beige)") without a size or pack; those lines are keyed by item id and read through an item table, data/reference/marketplace_items.csv, that the tool keeps up to date (one row per item id, with
UPC and name) and pre-fills from the name: category; for curtains the length from the trade words Door = 7 ft, Window = 5 ft,
Long Door = 9 ft (HomeMonde's own D / W / LD SKU codes say the same) and pieces = 2 (Blinkit's price per item matches the
Amazon price of the pack of 2, not the single piece); for other items the sheet product only, because the name gives no size.
Metres then come from the fabric sheet's size rule and are flagged `item: size rule` (fabric code unknown, like other
size-rule estimates). Every cell of that table can be corrected by hand, and a `base_sku` filled in ties the item to that
SKU's exact fabric facts instead.

The website export carries buyer names, phones and e-mail addresses. They are never read (the reader drops those columns)
and the copy kept in data/raw/marketplaces/ is written without them.
"""
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .geo import STATE_TO_REGION, add_district, normalise_states

AMAZON = "Amazon"
MARKETPLACES = [AMAZON, "Flipkart", "Myntra", "Website", "Ajio", "Blinkit"]
MATCH_ORDER = ["master sheet", "not in master", "reference file", "no SKU on report"]
# how a report is recognised: every one of these columns must be present (names as the marketplace exports them)
SIGNATURES = {
    "Flipkart": ["SKU ID", "Gross Units", "Cancellation Units", "Order Date"],
    "Myntra": ["seller sku code", "order release id", "created on", "order status"],
    "Ajio": ["Seller SKU", "Cust Order No", "Order Qty", "Cancelled Qty"],
    "Blinkit": ["Item Id", "UPC", "Order Date", "Quantity", "Product Name"],
    "Website": ["SKU", "Website Order ID", "QUANTITY", "ORDER STATUS", "PRODUCT NAME"],
}
AMAZON_SIGNATURE = ["amazon-order-id", "asin", "purchase-date"]
PII_COLS = ["NAME", "PHONE", "EMAIL", "GST Number"]                        # website export: dropped on read, never stored
COLS = ["marketplace", "order_id", "line_id", "date", "month", "seller_sku", "base_sku", "sku_match", "item_id", "upc", "product_name", "mp_category",
        "mp_size", "quantity", "gross_units", "cancelled_units", "returned_units", "revenue", "order_status", "is_cancelled", "is_demand",
        "state", "region", "city", "pin", "district", "district_source", "source_file"]
MASTER_TABS = {"Amazon": "Seller SKU on Amazon", "Flipkart": "Seller SKU on Flipkart", "Myntra": "Seller SKU on Myntra", "Shopify": "Seller SKU on Shopify"}


def _norm(s: pd.Series) -> pd.Series:
    """SKU text as the tool keys it: trimmed, upper case, no inner spaces; blanks and '-' placeholders become NA."""
    out = s.astype("string").str.strip().str.upper().str.replace(r"\s+", "", regex=True)
    return out.mask(out.isin(["", "-", "NAN", "NONE", "NA", "N/A"]))


def _read_any(path: Path, **kw) -> pd.DataFrame:
    """A workbook (first sheet) or CSV as text, tolerating a file that Excel holds open."""
    import warnings
    from .ingest import _copy_locked
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, encoding="utf-8", encoding_errors="replace", low_memory=False, **kw)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")      # marketplace exports often lack one; harmless
        try:
            return pd.read_excel(path, dtype=str, **kw)
        except PermissionError:
            with tempfile.TemporaryDirectory() as tmp:
                return pd.read_excel(_copy_locked(path, Path(tmp)), dtype=str, **kw)


# ------------------------------------------------------------------ the Master SKU sheet
def find_master() -> Path | None:
    for folder in (cfg.MARKETPLACE_DIR, cfg.RAW_DIR, cfg.ROOT):
        hits = sorted(folder.glob("Master SKU*.xls*"))
        if hits:
            return hits[-1]
    return None


def read_master(path: Path) -> pd.DataFrame:
    """Every alias the Master sheet knows -> base SKU: one row per (alias, base_sku, channel). Channels are the sheet's tabs
    (Amazon, Flipkart, Myntra, Shopify), 'base' for the base SKU itself and 'old base' for a renamed base SKU. Where the
    sheet gives one alias two different bases, the first tab in the workbook's order wins and the rest are reported."""
    import warnings
    from .ingest import _copy_locked
    warnings.filterwarnings("ignore", message="Workbook contains no default style")
    try:
        xl = pd.ExcelFile(path)
    except PermissionError:
        tmp = tempfile.mkdtemp()
        xl = pd.ExcelFile(_copy_locked(path, Path(tmp)))
    rows = []
    base_tab = next((s for s in xl.sheet_names if s.strip().lower() == "base"), None)
    if base_tab:
        b = xl.parse(base_tab, dtype=str)
        b.columns = [str(c).strip() for c in b.columns]
        for v in _norm(b["Base SKU"]).dropna().unique():
            rows.append((v, v, "base"))
    ready = next((s for s in xl.sheet_names if s.strip().lower() == "ready inventory"), None)
    if ready:
        r = xl.parse(ready, dtype=str, header=None)
        hdr = r.index[r.iloc[:, 0].astype(str).str.strip().str.lower() == "old base sku"]
        if len(hdr):
            body = r.iloc[hdr[0] + 1:, :2].dropna()
            old, new = _norm(body.iloc[:, 0]), _norm(body.iloc[:, 1])
            for o, n in zip(old, new):
                if pd.notna(o) and pd.notna(n):
                    rows.append((n, n, "base"))
                    if o != n:
                        rows.append((o, n, "old base"))
    for channel, prefix in MASTER_TABS.items():
        tab = next((s for s in xl.sheet_names if s.strip().lower() == channel.lower()), None)
        if tab is None:
            continue
        t = xl.parse(tab, dtype=str)
        t.columns = [str(c).strip() for c in t.columns]
        base = _norm(t["Base SKU"])
        for c in [c for c in t.columns if c.lower().startswith(prefix.lower())]:
            for b_, a in zip(base, _norm(t[c])):
                if pd.notna(a) and pd.notna(b_):
                    rows.append((a, b_, channel))
                    rows.append((b_, b_, "base"))
    m = pd.DataFrame(rows, columns=["alias", "base_sku", "channel"]).drop_duplicates()
    looks_like_sku = lambda s: s.str.contains("[A-Z]", regex=True) & (s.str.len() >= 4)           # stray cell values ('2', '-') are not SKUs
    m = m[looks_like_sku(m["alias"]) & looks_like_sku(m["base_sku"])]
    # one alias, one base: an alias listed under two bases keeps the first (base tabs come first, then the marketplaces in order)
    m["_rank"] = m["channel"].map({"base": 0, "old base": 1, "Amazon": 2, "Flipkart": 3, "Myntra": 4, "Shopify": 5})
    m = m.sort_values(["alias", "_rank"], kind="stable")
    keep = m.drop_duplicates("alias", keep="first").drop(columns="_rank").reset_index(drop=True)
    first = keep.set_index("alias")["base_sku"]
    conflicts = m[m["base_sku"] != m["alias"].map(first)].drop_duplicates(["alias", "base_sku"])
    conflicts = conflicts.assign(kept=conflicts["alias"].map(first))
    keep.attrs["conflicts"] = conflicts.drop(columns="_rank").reset_index(drop=True)
    keep.attrs["bases"] = int(keep["base_sku"].nunique())
    return keep


def load_master() -> pd.DataFrame:
    if not cfg.MASTER_PARQUET.exists():
        return pd.DataFrame(columns=["alias", "base_sku", "channel"])
    return pd.read_parquet(cfg.MASTER_PARQUET)


# ------------------------------------------------------------------ the reports
def detect(columns) -> str | None:
    cols = {str(c).strip() for c in columns}
    lower = {c.lower() for c in cols}
    if all(c in cols for c in AMAZON_SIGNATURE):
        return AMAZON
    for name, need in SIGNATURES.items():
        if all(c.lower() in lower for c in need):
            return name
    return None


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column by name, case- and space-insensitive (exports differ in capitalisation between months)."""
    want = re.sub(r"\s+", " ", name.strip().lower())
    for c in df.columns:
        if re.sub(r"\s+", " ", str(c).strip().lower()) == want:
            return df[c]
    return pd.Series(pd.NA, index=df.index, dtype="string")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False), errors="coerce")


def _flipkart(df: pd.DataFrame) -> pd.DataFrame:
    gross, cancel, ret = _num(_col(df, "Gross Units")).fillna(0), _num(_col(df, "Cancellation Units")).fillna(0), _num(_col(df, "Return Units")).fillna(0)
    gmv, camt = _num(_col(df, "GMV")).fillna(0), _num(_col(df, "Cancellation Amount")).fillna(0)
    sku = _norm(_col(df, "SKU ID"))
    date = pd.to_datetime(_col(df, "Order Date"), errors="coerce")
    out = pd.DataFrame({
        "order_id": pd.Series(pd.NA, index=df.index, dtype="string"),           # the export is aggregated: no order ids
        "line_id": sku.astype("string") + "|" + date.dt.strftime("%Y-%m-%d") + "|" + _col(df, "Location Id").astype("string").fillna("") + "|"
                   + _col(df, "Fulfillment Type").astype("string").fillna(""),
        "date": date, "seller_sku": sku, "item_id": _col(df, "Product Id").astype("string"), "upc": pd.Series(pd.NA, index=df.index, dtype="string"),
        "product_name": pd.Series(pd.NA, index=df.index, dtype="string"),
        "mp_category": _col(df, "Vertical").astype("string").str.replace(r"(?<=[a-z])(?=[A-Z])", " ", regex=True),   # 'LivingRoomCurtain' -> 'Living Room Curtain'
        "mp_size": pd.Series(pd.NA, index=df.index, dtype="string"),
        "quantity": (gross - cancel).clip(lower=0), "gross_units": gross, "cancelled_units": cancel, "returned_units": ret,
        "revenue": (gmv - camt).clip(lower=0), "order_status": pd.Series(pd.NA, index=df.index, dtype="string"),
        "state": pd.Series(pd.NA, index=df.index, dtype="string"), "city": pd.Series(pd.NA, index=df.index, dtype="string"),
        "pin": pd.Series(pd.NA, index=df.index, dtype="string"),
    })
    out["line_id"] = out["line_id"] + "|" + out.groupby("line_id").cumcount().astype(str)
    return out


def _myntra(df: pd.DataFrame) -> pd.DataFrame:
    status = _col(df, "order status").astype("string").str.strip().str.upper()
    cancelled = status.eq("F")                                     # F = failed / cancelled before dispatch; RTO shipped and came back
    returned = _col(df, "return creation date").notna() | status.eq("RTO")
    out = pd.DataFrame({
        "order_id": _col(df, "store order id").astype("string").str.lstrip("'").str.strip(),
        "line_id": _col(df, "order line id").astype("string").str.strip(),
        "date": pd.to_datetime(_col(df, "created on"), errors="coerce"),
        "seller_sku": _norm(_col(df, "seller sku code")).fillna(_norm(_col(df, "vendor article number"))),
        "item_id": _col(df, "sku id").astype("string"), "upc": pd.Series(pd.NA, index=df.index, dtype="string"), "product_name": _col(df, "style name").astype("string"),
        "mp_category": _col(df, "article type").astype("string"), "mp_size": _col(df, "size").astype("string"),
        "quantity": np.where(cancelled, 0, 1), "gross_units": 1, "cancelled_units": cancelled.astype(int), "returned_units": returned.astype(int),
        "revenue": np.where(cancelled, 0.0, _num(_col(df, "final amount")).fillna(0)),
        "order_status": status, "state": _col(df, "state").astype("string"), "city": _col(df, "city").astype("string"),
        "pin": _col(df, "zipcode").astype("string"),
    })
    return out


def _ajio(df: pd.DataFrame) -> pd.DataFrame:
    qty, cancel = _num(_col(df, "Order Qty")).fillna(0), _num(_col(df, "Cancelled Qty")).fillna(0)
    price = _num(_col(df, "Selling Price")).fillna(0)
    date = pd.to_datetime(_col(df, "Cust Order Date").astype("string").str.replace(r"\s+IST\s+", " ", regex=True), format="%a %b %d %H:%M:%S %Y", errors="coerce")
    if date.isna().any():                                           # a month exported in another date format
        date = date.fillna(pd.to_datetime(_col(df, "Cust Order Date"), errors="coerce"))
    out = pd.DataFrame({
        "order_id": _col(df, "Cust Order No").astype("string"),
        "line_id": _col(df, "Cust Order No").astype("string") + "|" + _col(df, "FWD Seller Order NO").astype("string").fillna("") + "|" + _norm(_col(df, "Seller SKU")).astype("string").fillna(""),
        "date": date, "seller_sku": _norm(_col(df, "Seller SKU")), "item_id": _col(df, "JioCode").astype("string"), "upc": _col(df, "EAN").astype("string").str.strip(),
        "product_name": _col(df, "Description").astype("string"), "mp_category": _col(df, "Description").astype("string"),
        "mp_size": pd.Series(pd.NA, index=df.index, dtype="string"),
        "quantity": (qty - cancel).clip(lower=0), "gross_units": qty, "cancelled_units": cancel, "returned_units": 0,
        "revenue": price * (qty - cancel).clip(lower=0), "order_status": _col(df, "Status").astype("string"),
        "state": pd.Series(pd.NA, index=df.index, dtype="string"), "city": pd.Series(pd.NA, index=df.index, dtype="string"),
        "pin": pd.Series(pd.NA, index=df.index, dtype="string"),
    })
    out["line_id"] = out["line_id"] + "|" + out.groupby("line_id").cumcount().astype(str)
    return out


def _blinkit(df: pd.DataFrame) -> pd.DataFrame:
    qty = _num(_col(df, "Quantity")).fillna(0)
    status = _col(df, "Order Status").astype("string").str.upper()
    cancelled = status.str.contains("CANCEL", na=False)
    out = pd.DataFrame({
        "order_id": _col(df, "Order Id").astype("string"),
        "line_id": _col(df, "Order Id").astype("string") + "|" + _col(df, "Item Id").astype("string"),
        "date": pd.to_datetime(_col(df, "Order Date"), errors="coerce"),
        # the "Order data" export carries HomeMonde's SKU per item ('Skus'); the plain "Sales Summary" export does not, and those items
        # go through the item table instead
        "seller_sku": _norm(_col(df, "Skus")) if any(str(c).strip().lower() == "skus" for c in df.columns) else pd.Series(pd.NA, index=df.index, dtype="string"),
        "item_id": _col(df, "Item Id").astype("string").str.strip(), "upc": _col(df, "UPC").astype("string").str.strip(),
        "product_name": _col(df, "Product Name").astype("string").str.strip(),
        "mp_category": _col(df, "Mapping on consumer app (L0, L1, L2)").astype("string").str.split("->").str[-1].str.strip(),
        "mp_size": pd.Series(pd.NA, index=df.index, dtype="string"),
        "quantity": np.where(cancelled, 0, qty), "gross_units": qty, "cancelled_units": np.where(cancelled, qty, 0), "returned_units": 0,
        "revenue": np.where(cancelled, 0.0, _num(_col(df, "Total Gross Bill Amount")).fillna(0)),
        "order_status": status, "state": _col(df, "Customer State").astype("string"), "city": _col(df, "Customer City").astype("string"),
        "pin": pd.Series(pd.NA, index=df.index, dtype="string"),
    })
    out["line_id"] = out["line_id"] + "|" + out.groupby("line_id").cumcount().astype(str)
    return out


def _website(df: pd.DataFrame, known_skus: set) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if str(c).strip().upper() in {p.upper() for p in PII_COLS} or str(c).lower().startswith("payer_")])
    name = _col(df, "PRODUCT NAME").astype("string").str.strip()
    price = _num(_col(df, "PRICE")).fillna(0)
    # "Discount for product: ..." lines are order-level discounts (negative prices), not items
    keep = ~(name.str.lower().str.startswith("discount", na=False) | (price < 0))
    df, name, price = df[keep], name[keep], price[keep]
    qty = _num(_col(df, "QUANTITY")).fillna(0)
    status = _col(df, "ORDER STATUS").astype("string").str.strip()
    cancelled = status.str.lower().isin(["cancelled", "canceled", "invalid shipment"])
    returned = status.str.lower().str.startswith("returning", na=False)
    sku = _norm(_col(df, "SKU"))
    # a line without a SKU often spells it in the product name ("... / HMCR_104-6FT", "SKU HM152-065-GRWH-7FT"); it is taken only
    # when it is a SKU the Master sheet or the fabric sheet knows
    todo = sku.isna()
    if todo.any() and known_skus:
        tokens = name[todo].str.upper().str.findall(r"[A-Z]{2,6}[-_][A-Z0-9][A-Z0-9&_.-]{2,}")
        found = tokens.map(lambda ts: next((t.rstrip(".-") for t in ts if t.rstrip(".-") in known_skus), pd.NA))
        sku = sku.astype("string")
        sku[todo] = found.astype("string")
    out = pd.DataFrame({
        "order_id": _col(df, "Website Order ID").astype("string"),
        "line_id": _col(df, "Item ID").astype("string"),
        "date": pd.to_datetime(_col(df, "Order DATE"), errors="coerce"),
        "seller_sku": sku, "item_id": _col(df, "PRODUCT ID").astype("string").mask(lambda s: s.isin(["0", "", "nan"])), "product_name": name,   # 0 = no id
        "upc": pd.Series(pd.NA, index=df.index, dtype="string"),
        "mp_category": _col(df, "Category").astype("string").str.replace("_", " ").mask(lambda s: s.str.lower().isin(["no data", "nan"])),
        "mp_size": pd.Series(pd.NA, index=df.index, dtype="string"),
        "quantity": np.where(cancelled, 0, qty), "gross_units": qty, "cancelled_units": np.where(cancelled, qty, 0),
        "returned_units": np.where(returned, qty, 0), "revenue": np.where(cancelled, 0.0, price * qty),        # PRICE is per unit, before order-level discounts
        "order_status": status, "state": _col(df, "delivery_state").astype("string"), "city": _col(df, "delivery_city").astype("string"),
        "pin": _col(df, "delivery_postal_code").astype("string"),
    })
    return out


READERS = {"Flipkart": _flipkart, "Myntra": _myntra, "Ajio": _ajio, "Blinkit": _blinkit}


def read_report(path: Path, known_skus: set = frozenset()) -> tuple[str, pd.DataFrame]:
    """(marketplace, standard rows) for one export; raises ValueError for a file no reader recognises."""
    raw = _read_any(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    name = detect(raw.columns)
    if name is None:
        raise ValueError(f"{path.name}: not a marketplace sales report the tool knows (Flipkart, Myntra, Ajio, Blinkit, website)")
    if name == AMAZON:
        raise ValueError(f"{path.name}: an Amazon All Orders report; put it in {cfg.RAW_DIR} for python -m demand.ingest")
    df = _website(raw, known_skus) if name == "Website" else READERS[name](raw)
    # a line with nothing on it is dropped; one that only carries a return of an earlier sale (Flipkart: gross 0, return 1) stays, so returns add up
    df = df[df["date"].notna() & ((df["gross_units"] > 0) | (df["cancelled_units"] > 0) | (df["returned_units"] > 0))].copy()
    df["marketplace"], df["source_file"] = name, path.name
    df["date"] = df["date"].dt.normalize()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    for c in ("quantity", "gross_units", "cancelled_units", "returned_units"):
        df[c] = df[c].astype("int64")
    df["is_cancelled"] = df["quantity"].eq(0) & df["cancelled_units"].gt(0)
    df["is_demand"] = df["quantity"].gt(0)
    df["pin"] = df["pin"].astype("string").str.extract(r"(\d{6})")[0]
    df["city"] = df["city"].astype("string").str.upper().str.strip()
    df["state"] = normalise_states(df["state"].astype("string"), df["pin"])
    df["region"] = df["state"].map(STATE_TO_REGION).fillna("Unknown")
    has_pin = df["pin"].notna()
    dist = pd.DataFrame({"district": "UNKNOWN", "district_source": "none"}, index=df.index)
    if has_pin.any():
        dd = add_district(df.loc[has_pin, "pin"], df.loc[has_pin, "state"])
        dist.loc[has_pin, ["district", "district_source"]] = dd[["district", "district_source"]].to_numpy()
    df["district"], df["district_source"] = dist["district"], dist["district_source"]
    return name, df


# ------------------------------------------------------------------ items known by item id only (Blinkit)
ITEM_COLS = ["marketplace", "item_id", "upc", "product_name", "category", "fabric_product", "size", "pieces", "basis", "base_sku"]
# our category -> the fabric sheet's product name whose (size -> metres per piece) rule applies
ITEM_PRODUCT = {"Curtain": "Curtains", "Cushion Cover": "Cushion Covers", "Pillow Cover": "Pillow Covers", "Bolster Cover": "Bloster Cover",
                "Bedsheet": "Bedsheet", "Table Linen": "Table Cover", "Duvet Cover": "Duvet Covers", "Comforter/Quilt": "Dohar"}
CURTAIN_WORDS = [("long door", 9), ("door", 7), ("window", 5)]        # the trade sizes; HomeMonde's own SKU codes LD / D / W carry the same lengths


def describe_item(marketplace: str, name: str) -> dict:
    """What a report's item name says about the product, in the terms the fabric sheet's size rule needs. Only what the name
    (or, for Blinkit curtains, the price evidence) supports is filled; the rest is left blank for the business to complete."""
    from .ingest import categorise
    n = str(name or "")
    low = n.lower()
    cat = categorise(n)
    out = {"category": cat, "fabric_product": ITEM_PRODUCT.get(cat), "size": None, "pieces": None, "basis": ""}
    if cat == "Bedsheet" and "fitted" in low:
        out["fabric_product"] = "Fitted Bedsheet"                       # its own size rule on the sheet
    m = re.search(r"(\d{2,3})\s*[xX×]\s*(\d{2,3})\s*(?:inch|in\b|\")?", n)
    if m:
        out["size"] = f"{m.group(1)} x {m.group(2)} Inch"
        out["basis"] = "size from the item name"
    if cat == "Curtain":
        ft = re.search(r"(?<![\d.])(\d{1,2})\s*(?:feet|ft|foot)\b", low)
        if ft:
            out["size"], out["basis"] = f"{int(ft.group(1))} Feet", "length from the item name"
        else:
            for word, feet in CURTAIN_WORDS:
                if word in low:
                    out["size"], out["basis"] = f"{feet} Feet", f"length from the word '{word}' in the name ({feet} ft, as HomeMonde's own {'LD' if feet == 9 else 'D' if feet == 7 else 'W'} codes)"
                    break
    pk = re.search(r"(?:set|pack|pair)\s*of\s*(\d{1,2})|\b(\d{1,2})\s*(?:pcs|pieces?|panels?)\b", low)
    if pk:
        out["pieces"] = int(pk.group(1) or pk.group(2))
        out["basis"] = (out["basis"] + "; " if out["basis"] else "") + "pieces from the item name"
    elif "single" in low or "1 pc" in low:
        out["pieces"] = 1
        out["basis"] = (out["basis"] + "; " if out["basis"] else "") + "single piece per the name"
    elif marketplace == "Blinkit" and cat == "Curtain" and out["size"]:
        out["pieces"] = 2
        out["basis"] += "; pieces = 2: Blinkit's price per item matches the Amazon price of the pack of 2 for this length (the single piece sells at about half)"
    elif out["fabric_product"] in ("Bedsheet", "Fitted Bedsheet", "Table Cover", "Dohar", "Duvet Covers"):
        out["pieces"] = 1                                                 # one piece per unit, as on the fabric sheet's own rows for these products
        out["basis"] = (out["basis"] + "; " if out["basis"] else "") + "one piece per unit (as on the fabric sheet)"
    if out["fabric_product"] and not out["size"]:
        out["basis"] = (out["basis"] + "; " if out["basis"] else "") + "size not on the report: fill in `size` (and `pieces`) to get metres"
    if not out["fabric_product"]:
        out["basis"] = (out["basis"] + "; " if out["basis"] else "") + "no size rule on the fabric sheet for this product: fill in `base_sku`"
    return out


def _items_file() -> pd.DataFrame:
    if cfg.MARKETPLACE_ITEMS_CSV.exists():
        f = pd.read_csv(cfg.MARKETPLACE_ITEMS_CSV, dtype=str, keep_default_na=False)
        for c in ITEM_COLS:
            if c not in f.columns:
                f[c] = ""
        f = f[ITEM_COLS].astype(str)
        return f.replace({"nan": "", "<NA>": "", "None": ""})
    return pd.DataFrame(columns=ITEM_COLS, dtype=str)


def _item_key(marketplace: pd.Series, item_id: pd.Series, name: pd.Series) -> pd.Series:
    """One key per item: the item id where the report gives one, else the item name."""
    iid = item_id.astype("string").str.strip().replace("", pd.NA)
    return marketplace.astype(str) + "|" + iid.fillna(name.astype("string").str.strip()).astype(str)


def update_items_file(mp: pd.DataFrame) -> pd.DataFrame:
    """Keep data/reference/marketplace_items.csv: one row per item a report names without a SKU (Blinkit's whole catalogue, a few
    website lines), with the item id, UPC and name as the report gives them and the tool's reading of the name (category, sheet
    product, size, pieces, basis). Rows already there are kept exactly as they are, so hand corrections survive; new items are
    appended. A `base_sku` filled in by hand ties the item to that SKU instead of the size rule."""
    f = _items_file()
    nosku = mp[mp["seller_sku"].isna() & (mp["item_id"].notna() | mp["product_name"].notna())].copy()
    if nosku.empty and cfg.MARKETPLACE_ITEMS_CSV.exists():
        return f
    nosku["_key"] = _item_key(nosku["marketplace"], nosku["item_id"], nosku["product_name"])
    first = lambda s_: s_.dropna().astype(str).str.strip().replace("", pd.NA).dropna().mode().iat[0] if s_.notna().any() and (s_.dropna().astype(str).str.strip() != "").any() else ""
    seen = nosku.groupby("_key").agg(marketplace=("marketplace", "first"), item_id=("item_id", first), upc=("upc", first), product_name=("product_name", first),
                                     units=("gross_units", "sum")).reset_index(drop=True)
    have = set(_item_key(f["marketplace"], f["item_id"], f["product_name"]))
    new = seen[~_item_key(seen["marketplace"], seen["item_id"], seen["product_name"]).isin(have)].sort_values("units", ascending=False)
    if len(new) or not cfg.MARKETPLACE_ITEMS_CSV.exists():
        desc = pd.DataFrame([describe_item(m, n) for m, n in zip(new["marketplace"], new["product_name"])], index=new.index)
        new = pd.concat([new.drop(columns="units"), desc], axis=1).assign(base_sku="")
        new["pieces"] = new["pieces"].map(lambda v: "" if v is None or pd.isna(v) else str(int(v)))
        f = pd.concat([f, new[ITEM_COLS].fillna("").astype(str)], ignore_index=True)
        cfg.MARKETPLACE_ITEMS_CSV.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(cfg.MARKETPLACE_ITEMS_CSV, index=False)
    return f


def items_table() -> pd.DataFrame:
    """The item table as it stands, with the product key each row maps to (`asin`) and whether it will give metres."""
    f = _items_file()
    if f.empty:
        return f.assign(asin=pd.Series(dtype=str), status=pd.Series(dtype=str))
    f["asin"] = NOSKU_KEY + _item_key(f["marketplace"], f["item_id"], f["product_name"])
    has_sku = f["base_sku"].str.strip() != ""
    complete = (f["fabric_product"].str.strip() != "") & (f["size"].str.strip() != "") & (f["pieces"].str.strip() != "")
    f["status"] = np.select([has_sku, complete], ["tied to base SKU", "metres by size rule"], "needs size / pieces or a base SKU")
    return f


def apply_item_rules(pf: pd.DataFrame, fab: pd.DataFrame) -> pd.DataFrame:
    """Fill the fabric facts of item-keyed products (Blinkit) from the item table and the fabric sheet's size rule: metres per
    piece for (sheet product, size) x pieces. Rows the table does not complete keep `not in sheet`. Items tied to a base SKU
    never reach here (they carry that SKU's own facts through the SKU route)."""
    from . import fabric as FB
    items = items_table()
    if pf.empty or items.empty:
        return pf
    rules = FB.size_rules(fab)
    use = items[items["status"] == "metres by size rule"].drop_duplicates("asin").set_index("asin")
    out = pf.copy()
    hit = out["asin"].isin(use.index) & (out["fabric_source"] == "not in sheet")
    for i in out.index[hit]:
        it = use.loc[out.at[i, "asin"]]
        size = FB._norm_size(it["size"]) if "Feet" not in str(it["size"]) else re.sub(r"\s+", " ", str(it["size"]).strip())
        try:
            pieces = float(str(it["pieces"]).strip())
        except ValueError:
            continue
        if (it["fabric_product"], size) not in rules:
            continue
        mpp = rules[(it["fabric_product"], size)]
        out.at[i, "fabric_product"], out.at[i, "size"], out.at[i, "pieces"] = it["fabric_product"], size, pieces
        out.at[i, "metres_per_piece"], out.at[i, "metres_per_unit"] = mpp, mpp * pieces
        out.at[i, "fabric_code"], out.at[i, "fabric_source"] = FB.UNKNOWN, FB.ITEM_RULE
        if it["category"]:
            out.at[i, "category"] = it["category"]
    return out


# ------------------------------------------------------------------ build the store
def _find_files() -> list[Path]:
    cfg.MARKETPLACE_DIR.mkdir(parents=True, exist_ok=True)
    files = [f for pat in ("*.xlsx", "*.xlsm", "*.xls", "*.csv") for f in cfg.MARKETPLACE_DIR.glob(pat)]
    return sorted(f for f in files if not f.name.startswith("~$") and not f.name.lower().startswith("master sku"))


def update_marketplaces(log=print) -> pd.DataFrame | None:
    """Rebuild marketplace.parquet (and sku_master.parquet) when any report or the Master sheet is new or changed.
    Returns the table, or None when there are no marketplace files at all."""
    files, master_path = _find_files(), find_master()
    if not files and master_path is None:
        return None
    sig = {f.name: {"size": f.stat().st_size, "mtime": int(f.stat().st_mtime)} for f in files}
    if master_path is not None:
        sig["__master__"] = {"size": master_path.stat().st_size, "mtime": int(master_path.stat().st_mtime), "name": master_path.name}
    items_sig = {"size": cfg.MARKETPLACE_ITEMS_CSV.stat().st_size, "mtime": int(cfg.MARKETPLACE_ITEMS_CSV.stat().st_mtime)} if cfg.MARKETPLACE_ITEMS_CSV.exists() else None
    sig["__items__"] = items_sig
    if cfg.MARKETPLACE_PARQUET.exists() and cfg.MARKETPLACE_MANIFEST.exists():
        have = json.loads(cfg.MARKETPLACE_MANIFEST.read_text())
        if have.get("files") == sig:
            return load()
    cfg.STORE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"files": sig, "reports": {}, "master": {}}
    master = pd.DataFrame(columns=["alias", "base_sku", "channel"])
    if master_path is not None:
        log(f"reading {master_path.name} ...")
        master = read_master(master_path)
        conf = master.attrs.pop("conflicts", pd.DataFrame())
        master.attrs = {}                                         # parquet metadata must be JSON; the summary goes to the manifest instead
        master.to_parquet(cfg.MASTER_PARQUET, index=False)
        manifest["master"] = {"name": master_path.name, "aliases": int(len(master)), "bases": int(master["base_sku"].nunique()),
                              "conflicts": conf.to_dict("records") if len(conf) else []}
        log(f"master SKU sheet: {len(master):,} seller SKUs over {master['base_sku'].nunique():,} base SKUs"
            + (f", {len(conf)} aliases listed under two bases (first kept)" if len(conf) else ""))
    alias = master.set_index("alias")["base_sku"] if len(master) else pd.Series(dtype="string")
    known = set(alias.index)
    try:
        from . import fabric as FB
        known |= set(FB.load_fabric()["sku"].dropna())
    except Exception:
        pass
    frames = []
    for f in files:
        log(f"reading {f.name} ...")
        try:
            name, df = read_report(f, known)
        except ValueError as e:
            log(f"  skipped: {e}")
            manifest["reports"][f.name] = {"skipped": str(e)}
            continue
        frames.append(df)
        manifest["reports"][f.name] = {"marketplace": name, "rows": int(len(df)), "units": int(df["quantity"].sum()),
                                       "from": str(df["date"].min().date()), "to": str(df["date"].max().date())}
        log(f"  {name}: {len(df):,} lines, {int(df['quantity'].sum()):,} demand units, {df['date'].min():%d %b %Y} - {df['date'].max():%d %b %Y}")
    if not frames:
        cfg.MARKETPLACE_MANIFEST.write_text(json.dumps(manifest, indent=2))
        return None
    mp = pd.concat(frames, ignore_index=True)
    # a month re-exported replaces the earlier copy of the same lines (the later file wins)
    mp["_file_order"] = mp["source_file"].map({f.name: i for i, f in enumerate(files)})
    mp = mp.sort_values("_file_order").drop_duplicates(["marketplace", "line_id"], keep="last").drop(columns="_file_order")
    # base SKU: the Master sheet's, else the seller SKU as it is; Blinkit items through the reference file
    mp["base_sku"] = mp["seller_sku"].map(alias)
    mp["sku_match"] = np.select([mp["seller_sku"].isna(), mp["base_sku"].notna()], ["no SKU on report", "master sheet"], "not in master")
    mp["base_sku"] = mp["base_sku"].fillna(mp["seller_sku"])
    items = update_items_file(mp)
    filled = items[items["base_sku"].str.strip() != ""]
    if len(filled):
        by_key = dict(zip(_item_key(filled["marketplace"], filled["item_id"], filled["product_name"]), _norm(filled["base_sku"])))
        ref = _item_key(mp["marketplace"], mp["item_id"], mp["product_name"]).map(by_key)
        hit = mp["seller_sku"].isna() & ref.notna()
        mp.loc[hit, "base_sku"] = ref[hit].map(lambda b: alias.get(b, b))
        mp.loc[hit, "sku_match"] = "reference file"
    mp = mp[COLS].sort_values(["date", "marketplace"]).reset_index(drop=True)
    mp.to_parquet(cfg.MARKETPLACE_PARQUET, index=False)
    manifest["items_without_sku"] = int(len(items)), int(len(filled))
    manifest["items_by_rule"] = int((items_table()["status"] == "metres by size rule").sum())
    manifest["files"]["__items__"] = ({"size": cfg.MARKETPLACE_ITEMS_CSV.stat().st_size, "mtime": int(cfg.MARKETPLACE_ITEMS_CSV.stat().st_mtime)}
                                      if cfg.MARKETPLACE_ITEMS_CSV.exists() else None)
    cfg.MARKETPLACE_MANIFEST.write_text(json.dumps(manifest, indent=2))
    d = mp[mp["is_demand"]]
    by = d.groupby("marketplace")["quantity"].sum().to_dict()
    log(f"marketplaces: {len(mp):,} lines, {int(d['quantity'].sum()):,} demand units ({', '.join(f'{k} {v:,}' for k, v in by.items())}), "
        f"{d['base_sku'].notna().mean() * 100:.1f}% of lines with a base SKU")
    return mp


def load() -> pd.DataFrame:
    if not cfg.MARKETPLACE_PARQUET.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_parquet(cfg.MARKETPLACE_PARQUET)


def info() -> dict:
    """What the manifest recorded (reports read, master sheet, items without a SKU); empty when there is none."""
    if not cfg.MARKETPLACE_MANIFEST.exists():
        return {}
    return json.loads(cfg.MARKETPLACE_MANIFEST.read_text())


# ------------------------------------------------------------------ join to the products the Amazon pipeline knows
KEY_PREFIX = "MP:"
NOSKU_KEY = KEY_PREFIX + "item "                        # products known by item id only: 'MP:item Blinkit|10236691'


def resolve(mp: pd.DataFrame, products: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Give every marketplace line the product key the fabric join uses: the ASIN of the Amazon listing that sells the same
    product (seller SKU, base SKU or any Master-sheet alias of the base found among the Amazon SKUs), else 'MP:<base SKU>'.
    Adds `asin`, `key_source` ('Amazon listing' / 'marketplace only') and `aliases` (every SKU name tied to the base)."""
    if mp.empty:
        return mp.assign(asin=pd.Series(dtype="string"), key_source=pd.Series(dtype="string"), aliases=pd.Series(dtype="object"))
    ex = products[["asin", "main_sku", "skus"]].copy()
    ex["sku"] = (ex["main_sku"].astype(str) + ", " + ex["skus"].astype(str)).str.upper().str.replace(" ", "").str.split(",")
    ex = ex.explode("sku")
    ex["_main"] = ex["sku"] == ex["main_sku"].str.upper().str.replace(" ", "")
    sku_asin = ex.sort_values("_main", ascending=False).drop_duplicates("sku").set_index("sku")["asin"]        # a SKU sold under two listings: the one it is the main SKU of
    by_base = master.groupby("base_sku")["alias"].agg(lambda s: sorted(set(s))) if len(master) else pd.Series(dtype="object")
    out = mp.copy()
    base = out["base_sku"].astype("string")
    cands = pd.DataFrame({"a": out["seller_sku"].astype("string"), "b": base})
    found = cands["a"].map(sku_asin).fillna(cands["b"].map(sku_asin))
    todo = found.isna() & base.notna()
    if todo.any():
        extra = base[todo].map(lambda b: next((sku_asin[a] for a in by_base.get(b, []) if a in sku_asin.index), pd.NA))
        found = found.astype("object")
        found[todo] = extra
    out["asin"] = found.where(found.notna(), (KEY_PREFIX + base).where(base.notna(), NOSKU_KEY + _item_key(out["marketplace"], out["item_id"], out["product_name"])))
    out["key_source"] = np.where(found.notna(), "Amazon listing", "marketplace only")
    out["aliases"] = base.map(lambda b: by_base.get(b, [b]) if pd.notna(b) else [])
    return out


def extra_products(mp: pd.DataFrame) -> pd.DataFrame:
    """A product row, in the shape of products.parquet, for every marketplace-only key, built from what the reports say about
    it (most common product name, base SKU, every alias) and read by the same attribute rules as Amazon listings."""
    from .ingest import categorise, product_attributes
    m = mp[mp["key_source"] == "marketplace only"]
    if m.empty:
        return pd.DataFrame(columns=["asin", "title", "main_sku", "n_skus", "skus", "category", "family", "curtain_length_ft",
                                     "length_title_vs_sku_conflict", "is_custom_size", "curtain_type", "pack_size", "first_sale", "last_sale"])
    first_name = lambda s: s.dropna().mode().iat[0] if s.notna().any() else ""
    g = m.groupby("asin")
    p = pd.DataFrame({
        "title": g["product_name"].agg(first_name),
        # an item the report names without a SKU keeps a label that says so ('(no SKU) Blinkit 10236691'), so it still shows up in the
        # SKUs-without-fabric-data list with its units instead of vanishing for want of a SKU
        "main_sku": g["base_sku"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
        "skus": g.apply(lambda x: ", ".join(sorted({a for al in x["aliases"] for a in al} | set(x["seller_sku"].dropna()))), include_groups=False),
        "mp_category": g["mp_category"].agg(first_name),
        "mp_size": g["mp_size"].agg(first_name),
    })
    p["n_skus"] = p["skus"].str.split(", ").map(len)
    nosku = p["main_sku"].astype(str).str.strip() == ""
    p.loc[nosku, "main_sku"] = p.index[nosku].str[len(NOSKU_KEY):].str.replace("|", " item ", regex=False)      # 'Blinkit item 10236691'
    p["title"] = p["title"].where(p["title"].astype(str).str.strip() != "", p["main_sku"])
    # where neither the SKU prefix nor the name says what the product is, the marketplace's own category word is put in front of the
    # name ('Living Room Curtain BCR152-093-NBLU-8FT-S1'), so the category and then the curtain length / pack size rules can read it
    other = pd.Series([categorise(t, k) == "Other" for t, k in zip(p["title"], p["main_sku"])], index=p.index) & p["mp_category"].notna() & (p["mp_category"] != "")
    p.loc[other, "title"] = p.loc[other, "mp_category"].astype(str) + " " + p.loc[other, "title"].astype(str)
    p = product_attributes(p.astype({"title": "string", "main_sku": "string"}))
    d = m[m["is_demand"]].groupby("asin")
    p["first_sale"], p["last_sale"] = d["date"].min(), d["date"].max()
    return p.drop(columns=["mp_category", "mp_size"]).reset_index()


def demand_frame(mp: pd.DataFrame, products: pd.DataFrame, extra: pd.DataFrame, sourcing: pd.Series) -> pd.DataFrame:
    """Marketplace demand lines in the shape the fabric views read (the same columns as the Amazon demand frame they are
    put next to): date, month, week, asin, sku, main_sku, quantity, revenue, order_id, category, place, sourcing, marketplace."""
    d = mp[mp["is_demand"]].copy()
    cat = pd.concat([products.set_index("asin")["category"], extra.set_index("asin")["category"]]) if len(extra) else products.set_index("asin")["category"]
    main = pd.concat([products.set_index("asin")["main_sku"], extra.set_index("asin")["main_sku"]]) if len(extra) else products.set_index("asin")["main_sku"]
    d["category"] = d["asin"].map(cat[~cat.index.duplicated()]).fillna("Other")
    d["main_sku"] = d["asin"].map(main[~main.index.duplicated()]).fillna(d["base_sku"]).fillna(d["seller_sku"]).fillna("(no SKU)")
    d["sku"] = d["seller_sku"].fillna(d["main_sku"])
    d["week"] = d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")
    d["sourcing"] = d["asin"].map(sourcing)
    cols = ["date", "month", "week", "asin", "sku", "main_sku", "quantity", "revenue", "order_id", "category", "state", "region", "district",
            "sourcing", "marketplace", "key_source", "sku_match"]
    return d[cols].reset_index(drop=True)


def mapping_summary(mp: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """Per marketplace: lines, units, how the SKUs were tied to products, and units that reached a fabric figure."""
    d = mp[mp["is_demand"]].merge(pf[["asin", "fabric_source"]], on="asin", how="left")
    from . import fabric as FB
    with_m = d["fabric_source"].isin(FB.WITH_METRES)
    g = d.assign(with_m=with_m).groupby("marketplace")
    t = pd.DataFrame({"lines": g.size(), "units": g["quantity"].sum(), "products": g["asin"].nunique(),
                      "units_master_sheet": g.apply(lambda x: x.loc[x["sku_match"] == "master sheet", "quantity"].sum(), include_groups=False),
                      "units_not_in_master": g.apply(lambda x: x.loc[x["sku_match"] == "not in master", "quantity"].sum(), include_groups=False),
                      "units_no_sku": g.apply(lambda x: x.loc[x["sku_match"].isin(["no SKU on report", "reference file"]), "quantity"].sum(), include_groups=False),
                      "units_amazon_listing": g.apply(lambda x: x.loc[x["key_source"] == "Amazon listing", "quantity"].sum(), include_groups=False),
                      "units_with_fabric_figure": g.apply(lambda x: x.loc[x["with_m"], "quantity"].sum(), include_groups=False)})
    t["fabric_figure_%"] = (t["units_with_fabric_figure"] / t["units"] * 100).round(1)
    return t.sort_values("units", ascending=False)
