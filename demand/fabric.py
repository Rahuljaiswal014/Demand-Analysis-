"""Fabric consumption: what the sales pull through in metres of each fabric, and what the forecast implies.

Source: "Fabric Consumption*.xlsx" in the project root or data/raw (HomeMonde's manufacturing sheet, one row per SKU:
fabric code, type, family, product type, colour, size, metres per piece, pieces per SKU). Cleaned into
data/store/fabric.parquet and joined to products by ASIN first, then by SKU.

A second, older list, "Old Fabric SKU*.xlsx" (SKU, product name, fabric code; no size, no metres), fills SKUs the main sheet
does not carry. The main sheet always wins where both list a SKU. Old-list products get their fabric code from the list and
their metres from the sheet's size rule, and are flagged `old list: code, size rule`; a custom-size product whose metres no
rule gives is `old list: code, no size`.

Products the sheet does not list yet (new ranges, alias SKUs, custom sizes) get metres by the sheet's own size rule when it
is unambiguous - curtain metres per piece depend on length only, cushion / pillow / bolster metres on the cover size - and are
flagged `inferred`; their fabric code stays unknown. Products on the outsource sheet (demand/sourcing.py) are bought in as
finished goods: they are marked `outsourced: no fabric` and never get metres by inference. Anything else stays `not in sheet`. Every figure the tool shows carries that source so sheet numbers are never mixed silently with
inferred ones.
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

BOUGHT_IN = "outsourced: no fabric"
OLD_RULE, OLD_NOSIZE = "old list: code, size rule", "old list: code, no size"
NOT_GIVEN = "(not given: old list)"                   # fabric type / family the old list's product name does not state
SOURCE_ORDER = ["sheet: ASIN", "sheet: SKU", OLD_RULE, "inferred: size rule", BOUGHT_IN, OLD_NOSIZE, "not in sheet"]
WITH_METRES = ["sheet: ASIN", "sheet: SKU", OLD_RULE, "inferred: size rule"]                      # sources that carry metres
UNKNOWN = "(unknown: new ranges, by size rule)"
COLS = {"SKU": "sku", "Fabric Type": "fabric_type", "Family": "fabric_family", "Product": "fabric_product", "Product Type": "product_type",
        "Colour": "colour", "Nested Category": "nested_category", "Fabric Code": "fabric_code", "Asins": "asin", "Size": "size",
        "Fabric Consumption in meters": "metres_per_piece", "Total Set In sku": "pieces", "Total Fabric used in sku": "metres_per_unit"}
FACTS = ["fabric_code", "fabric_type", "fabric_family", "fabric_product", "product_type", "colour", "size", "metres_per_piece", "pieces", "metres_per_unit"]
OLD_COLS = {"SKU": "sku", "Product Name": "fabric_name", "Product Code": "fabric_code"}
# the main sheet's own vocabulary; an old-list product name is only split into type / family where its first words are in it
OLD_TYPES = {"polyester": "Polyester", "satin": "Satin", "cotton": "Cotton", "sheer": "Sheer", "duck": "Duck", "velvet": "Velvet"}
OLD_FAMILIES = {"blackout": "Blackout", "digitalblackout": "Digital Blackout", "digital blackout": "Digital Blackout", "dimout": "Dimout",
                "printed": "Printed", "solid": "Solid", "stripe": "Stripe", "ombre": "Ombre"}
# our category -> the sheet's Product name(s) whose (size -> metres) rule may be borrowed for unlisted products
INFER_PRODUCT = {"Curtain": ["Curtains"], "Cushion Cover": ["Cushion Covers"], "Pillow Cover": ["Pillow Covers"], "Bolster Cover": ["Bloster Cover"]}


# ------------------------------------------------------------------ the sheet
def find_sheet() -> Path | None:
    for folder in (cfg.ROOT, cfg.RAW_DIR):
        hits = sorted(folder.glob("Fabric Consumption*.xls*"))
        if hits:
            return hits[-1]
    return None


def read_sheet(path: Path) -> pd.DataFrame:
    """The workbook as the tool uses it: one row per SKU, tidy names, numbers as numbers. Handles a workbook open in Excel."""
    from .ingest import _copy_locked
    try:
        raw = pd.read_excel(path, header=1, dtype=str)
    except PermissionError:
        with tempfile.TemporaryDirectory() as tmp:
            raw = pd.read_excel(_copy_locked(path, Path(tmp)), header=1, dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: columns not found: {missing}")
    f = raw[list(COLS)].rename(columns=COLS).dropna(how="all")
    for c in f.columns:
        f[c] = f[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).replace({"nan": None, "": None, "None": None})
    f = f.dropna(subset=["sku"])
    f["sku"] = f["sku"].str.upper()
    f["fabric_code"] = f["fabric_code"].str.upper().str.replace(r"\s+", "", regex=True)          # 'FAB -0057' -> 'FAB-0057'
    f["fabric_family"] = f["fabric_family"].str.title()                                          # 'DIMOUT' -> 'Dimout'
    f["product_type"] = f["product_type"].str.title()                                            # 'Big loop' == 'Big Loop'
    for c in ("metres_per_piece", "pieces", "metres_per_unit"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    dropped = int((f["metres_per_piece"].isna() | f["pieces"].isna()).sum())
    f = f[f["metres_per_piece"].notna() & f["pieces"].notna()].copy()
    # the sheet's own total column is trusted where present; otherwise metres x pieces
    f["metres_per_unit"] = f["metres_per_unit"].fillna(f["metres_per_piece"] * f["pieces"])
    f["size_ft"] = pd.to_numeric(f["size"].str.extract(r"(\d{1,2})\s*Feet", flags=re.I)[0], errors="coerce")
    # duplicate SKU rows are identical apart from a blank ASIN: keep the one with the ASIN
    f = f.sort_values("asin", na_position="last").drop_duplicates("sku", keep="first").reset_index(drop=True)
    f.attrs["dropped_rows"] = dropped
    return f


def _norm_size(s) -> str | None:
    """'72 x48 Inch' / '108X108' -> '72 x 48 Inch' / '108 x 108 Inch', so sheet sizes and SKU size tokens meet."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    m = re.fullmatch(r"\s*(\d{2,3})\s*[xX×]\s*(\d{2,3})\s*(?:inch(?:es)?)?\s*", str(s), flags=re.I)
    return f"{m.group(1)} x {m.group(2)} Inch" if m else re.sub(r"\s+", " ", str(s)).strip()


def _split_name(name: str) -> tuple[str | None, str | None, str]:
    """'Polyester Blackout Herringbone Beige' -> ('Polyester', 'Blackout', 'Herringbone Beige'). Words not in the sheet's
    vocabulary are left in the colour part rather than guessed."""
    words = str(name).split()
    ftype = fam = None
    if words and words[0].lower() in OLD_TYPES:
        ftype = OLD_TYPES[words.pop(0).lower()]
    if len(words) >= 2 and " ".join(words[:2]).lower() in OLD_FAMILIES:
        fam = OLD_FAMILIES[" ".join(words[:2]).lower()]
        words = words[2:]
    elif words and words[0].lower() in OLD_FAMILIES:
        fam = OLD_FAMILIES[words.pop(0).lower()]
    return ftype, fam, " ".join(words) or str(name)


def find_old_list() -> Path | None:
    for folder in (cfg.ROOT, cfg.RAW_DIR):
        hits = sorted(folder.glob("Old Fabric SKU*.xls*"))
        if hits:
            return hits[-1]
    return None


def read_old_list(path: Path) -> pd.DataFrame:
    """The old SKU list: one row per SKU with its fabric code, tidied the same way as the sheet. No sizes or metres."""
    from .ingest import _copy_locked
    try:
        raw = pd.read_excel(path, dtype=str)
    except PermissionError:
        with tempfile.TemporaryDirectory() as tmp:
            raw = pd.read_excel(_copy_locked(path, Path(tmp)), dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in OLD_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: columns not found: {missing}")
    o = raw[list(OLD_COLS)].rename(columns=OLD_COLS).dropna(how="all")
    for c in o.columns:
        o[c] = o[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).replace({"nan": None, "": None, "None": None})
    o = o.dropna(subset=["sku", "fabric_code"])
    o["sku"] = o["sku"].str.upper().str.replace(r"\s+", "", regex=True)
    code = o["fabric_code"].str.upper().str.replace(r"\s+", "", regex=True)
    padded = code.str.fullmatch(r"FAB-\d{1,3}").fillna(False)
    o["fabric_code"] = code.where(~padded, "FAB-" + code.str.extract(r"(\d+)$")[0].str.zfill(4))     # 'FAB-539' -> 'FAB-0539'
    o["padded_from"] = code.where(padded)
    o = o.drop_duplicates("sku").reset_index(drop=True)
    parts = o["fabric_name"].map(_split_name)
    o["fabric_type"], o["fabric_family"], o["colour"] = [x[0] for x in parts], [x[1] for x in parts], [x[2] for x in parts]
    o.attrs["padded_codes"] = sorted(set(zip(o.loc[o["padded_from"].notna(), "padded_from"], o.loc[o["padded_from"].notna(), "fabric_code"])))
    return o[["sku", "fabric_code", "fabric_name", "fabric_type", "fabric_family", "colour", "padded_from"]]


def update_fabric(log=print) -> pd.DataFrame | None:
    """Rebuild fabric.parquet when the sheet or the old list is new or changed. Returns the table, or None when there is no sheet.
    Sheet rows come first; old-list rows are added only for SKUs the sheet does not carry, with a `row_source` column saying
    which workbook each row came from. Conflicts (a SKU on both with different codes) go to the manifest; the sheet row is kept."""
    path = find_sheet()
    if path is None:
        return None
    old_path = find_old_list()
    sig = {"size": path.stat().st_size, "mtime": int(path.stat().st_mtime), "name": path.name}
    if old_path is not None:
        sig["old_list"] = {"size": old_path.stat().st_size, "mtime": int(old_path.stat().st_mtime), "name": old_path.name}
    if cfg.FABRIC_PARQUET.exists() and cfg.FABRIC_MANIFEST.exists():
        have = json.loads(cfg.FABRIC_MANIFEST.read_text())
        have_sig = {k: have[k] for k in ("size", "mtime", "name") if k in have}
        if "old_list" in have:
            have_sig["old_list"] = {k: have["old_list"].get(k) for k in ("size", "mtime", "name")}
        if have_sig == sig:
            return load_fabric()
    log(f"reading {path.name} ...")
    f = read_sheet(path)
    f["row_source"] = "sheet"
    f["fabric_name"], f["padded_from"] = None, None
    manifest = dict(sig)
    if old_path is not None:
        log(f"reading {old_path.name} ...")
        o = read_old_list(old_path)
        both = o.merge(f[["sku", "fabric_code"]], on="sku", suffixes=("_old", "_sheet"))
        conflicts = both[both["fabric_code_old"] != both["fabric_code_sheet"]]
        add = o[~o["sku"].isin(f["sku"])].copy()
        add["row_source"] = "old list"
        # an old code that the sheet consistently renumbered (every shared SKU says old X = sheet Y, and X never maps to a second
        # sheet code) is a renumbering: the old-list rows that still carry X get Y, so one fabric is not split across two codes
        pairs = conflicts.groupby("fabric_code_old")["fabric_code_sheet"].agg(["nunique", "first"])
        recode = {k: v for k, v in pairs.loc[pairs["nunique"] == 1, "first"].items()}
        if recode:
            add["fabric_code"] = add["fabric_code"].replace(recode)
            # a renumbered row takes the sheet's own description of its range (the sheet rows of the SKUs both workbooks
            # list), so one code does not show two spellings of the same colour
            desc = f[f["sku"].isin(both["sku"])].groupby("fabric_code")[["fabric_type", "fabric_family", "colour"]].agg(lambda s: s.mode().iloc[0])
            recoded = add["fabric_code"].isin(recode.values())
            for c in ("fabric_type", "fabric_family", "colour"):
                add.loc[recoded, c] = add.loc[recoded, "fabric_code"].map(desc[c])
        # type / family the product name did not give in the sheet's vocabulary stay visible in groupings rather than vanish
        add["fabric_type"] = add["fabric_type"].fillna(NOT_GIVEN)
        add["fabric_family"] = add["fabric_family"].fillna(NOT_GIVEN)
        f = pd.concat([f, add], ignore_index=True)
        manifest["old_list"] = dict(sig["old_list"], rows=int(len(o)), rows_added=int(len(add)), already_on_sheet=int(len(both)),
                                    conflicts=conflicts.rename(columns={"fabric_code_old": "old_list", "fabric_code_sheet": "sheet"})[["sku", "old_list", "sheet"]].to_dict("records"),
                                    recoded=[{"old_list": k, "sheet": v, "skus": int((o["fabric_code"] == k).sum())} for k, v in recode.items()],
                                    padded_codes=[list(x) for x in o.attrs.get("padded_codes", [])])
        log(f"old fabric list: {len(o):,} SKUs, {len(add):,} added ({add['fabric_code'].nunique()} fabric codes), {len(both):,} already on the sheet"
            + (f", {len(conflicts)} with a different code on the sheet (sheet kept)" if len(conflicts) else "")
            + (f", old code renumbered to the sheet's: {recode}" if recode else "")
            + (f", codes padded to 4 digits: {o.attrs['padded_codes']}" if o.attrs.get("padded_codes") else ""))
    f.to_parquet(cfg.FABRIC_PARQUET, index=False)
    cfg.FABRIC_MANIFEST.write_text(json.dumps(manifest))
    sheet = f[f["row_source"] == "sheet"]
    log(f"fabric sheet: {len(sheet):,} SKUs, {sheet['fabric_code'].nunique()} fabric codes, {sheet['asin'].nunique():,} ASINs"
        + (f", {f.attrs['dropped_rows']} rows without usable numbers skipped" if f.attrs.get("dropped_rows") else ""))
    return f


def load_fabric() -> pd.DataFrame:
    if not cfg.FABRIC_PARQUET.exists():
        return pd.DataFrame(columns=list(COLS.values()) + ["size_ft", "row_source", "fabric_name", "padded_from"])
    f = pd.read_parquet(cfg.FABRIC_PARQUET)
    if "row_source" not in f.columns:                      # parquet built before the old list existed
        f["row_source"], f["fabric_name"], f["padded_from"] = "sheet", None, None
    return f


def old_list_info() -> dict:
    """What the manifest recorded about the old list (rows added, conflicts, padded codes); empty when there is none."""
    if not cfg.FABRIC_MANIFEST.exists():
        return {}
    return json.loads(cfg.FABRIC_MANIFEST.read_text()).get("old_list") or {}


# ------------------------------------------------------------------ join to products
def _size_rules(fab: pd.DataFrame) -> dict:
    """(sheet product, size label) -> metres per piece, only where the sheet is unanimous. Sheet rows only: old-list rows
    carry no metres. Size labels are normalised so '72 x48 Inch' and a SKU's '72X48' meet."""
    s = fab[fab["row_source"] == "sheet"] if "row_source" in fab.columns else fab
    s = s.assign(size=s["size"].map(_norm_size))
    g = s.groupby(["fabric_product", "size"])["metres_per_piece"].agg(["nunique", "first"])
    return {k: v for k, v in g.loc[g["nunique"] == 1, "first"].items()}


def _sku_size(sku: str) -> str | None:
    """'HMBD-8016-BLU-108X108' -> '108 x 108 Inch'; None when the SKU carries no W x H token."""
    m = re.search(r"(\d{2,3})\s*[xX×]\s*(\d{2,3})(?:-|$)", str(sku))
    return f"{m.group(1)} x {m.group(2)} Inch" if m else None


def _title_size(title: str) -> str | None:
    m = re.search(r"(\d{2,3})\s*[xX×]\s*(\d{2,3})\s*(?:inch|in\b)", str(title), flags=re.I)
    return f"{m.group(1)} x {m.group(2)} Inch" if m else None


def product_fabric(products: pd.DataFrame, fab: pd.DataFrame, outsourced=()) -> pd.DataFrame:
    """One row per ASIN with its fabric facts and where they came from. `outsourced` = ASINs bought in as finished goods:
    unless the fabric sheet lists them they consume no in-house fabric, so the size rule is never applied to them."""
    outsourced = set(outsourced)
    base = products[["asin", "main_sku", "skus", "category", "title", "curtain_length_ft", "pack_size"]].reset_index(drop=True)
    if fab.empty:
        out = base.assign(**{c: np.nan for c in FACTS}, fabric_source="not in sheet")
        return out.drop(columns=["skus"])
    if "row_source" not in fab.columns:
        fab = fab.assign(row_source="sheet")
    by_asin = fab.dropna(subset=["asin"]).drop_duplicates("asin").set_index("asin")[FACTS]
    by_sku = fab.set_index("sku")[FACTS + ["row_source"]]
    facts = by_asin.reindex(base["asin"]).reset_index(drop=True)
    source = pd.Series(np.where(facts["fabric_code"].notna(), "sheet: ASIN", None), index=base.index, dtype="object")
    # SKU route for the rest: the main SKU, then any alias SKU of the listing, first hit wins
    todo = source.isna()
    cand = base.loc[todo, ["asin", "main_sku", "skus"]].copy()
    cand["sku"] = (cand["main_sku"] + ", " + cand["skus"].astype(str)).str.upper().str.split(", ")
    cand = cand.explode("sku")
    cand = cand[cand["sku"].isin(by_sku.index)].drop_duplicates("asin")
    if len(cand):
        hit = by_sku.reindex(cand["sku"]).reset_index(drop=True)
        hit.index = cand.index
        facts.loc[hit.index, FACTS] = hit[FACTS].to_numpy()
        source.loc[hit.index] = np.where(hit["row_source"] == "old list", OLD_NOSIZE, "sheet: SKU")
    out = pd.concat([base, facts], axis=1)
    out["fabric_source"] = source
    rules = _size_rules(fab)
    # old-list products: the code is known, the metres come from the sheet's size rule (curtains by length and pack size,
    # bedsheets by the W x H in the SKU or title, one piece each as on the sheet's own bedsheet rows)
    for i in out.index[out["fabric_source"] == OLD_NOSIZE]:
        cat, pack, sku = out.at[i, "category"], out.at[i, "pack_size"], str(out.at[i, "main_sku"])
        if cat == "Curtain":
            length = out.at[i, "curtain_length_ft"]
            if pd.isna(pack):
                m = re.search(r"-S(\d)$", sku)
                pack = float(m.group(1)) if m else np.nan
            prod, size = "Curtains", f"{int(length)} Feet" if pd.notna(length) else None
        elif cat == "Bedsheet":
            prod, size, pack = "Bedsheet", _sku_size(sku) or _norm_size(_title_size(out.at[i, "title"])), 1.0
        else:
            continue
        if size is not None and pd.notna(pack) and (prod, size) in rules:
            out.at[i, "metres_per_piece"], out.at[i, "pieces"] = rules[(prod, size)], float(pack)
            out.at[i, "metres_per_unit"] = rules[(prod, size)] * float(pack)
            out.at[i, "fabric_product"], out.at[i, "size"], out.at[i, "fabric_source"] = prod, size, OLD_RULE
    # inference by the sheet's own size rule, for listings neither workbook carries
    for i in out.index[out["fabric_source"].isna()]:
        cat, pack = out.at[i, "category"], out.at[i, "pack_size"]
        if out.at[i, "asin"] in outsourced:
            out.at[i, "fabric_source"] = BOUGHT_IN
            continue
        if cat not in INFER_PRODUCT or pd.isna(pack):
            continue
        length = out.at[i, "curtain_length_ft"]
        size = f"{int(length)} Feet" if cat == "Curtain" and pd.notna(length) else _title_size(out.at[i, "title"])
        for prod in INFER_PRODUCT[cat]:
            if (prod, size) in rules:
                out.at[i, "metres_per_piece"], out.at[i, "pieces"] = rules[(prod, size)], float(pack)
                out.at[i, "metres_per_unit"] = rules[(prod, size)] * float(pack)
                out.at[i, "fabric_product"], out.at[i, "size"], out.at[i, "fabric_source"] = prod, size, "inferred: size rule"
                break
    out["fabric_source"] = out["fabric_source"].fillna("not in sheet")
    out["fabric_code"] = out["fabric_code"].where(out["fabric_code"].notna(), np.where(out["fabric_source"] == "inferred: size rule", UNKNOWN, "(none)"))
    for c in ("metres_per_piece", "pieces", "metres_per_unit"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.drop(columns=["skus"])


def coverage(d: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """Units covered by each fabric source, overall and by category."""
    x = d.merge(pf[["asin", "fabric_source"]], on="asin", how="left")
    t = x.pivot_table(index="category", columns="fabric_source", values="quantity", aggfunc="sum", fill_value=0)
    t = t.reindex(columns=[c for c in SOURCE_ORDER if c in t.columns], fill_value=0)
    t["units"] = t.sum(axis=1)
    t = t.sort_values("units", ascending=False)
    t.loc["ALL"] = t.sum()
    unknown = sum(t[c] for c in ("not in sheet", OLD_NOSIZE) if c in t.columns)                  # bought-in items count as known: they need none
    t["known_%"] = ((t["units"] - unknown) / t["units"] * 100).round(1)
    return t.astype({c: int for c in t.columns if c != "known_%"})


MISSING_LABEL = {"inferred: size rule": "Estimated (no fabric code)", OLD_RULE: "Estimated metres (code from old list)",
                 OLD_NOSIZE: "No size (code from old list)", "not in sheet": "No data"}
PRIORITY = ["1 High", "2 Medium", "3 Low", "4 Not selling"]


def missing_skus(pf: pd.DataFrame, d: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Every SKU the fabric sheet should carry and does not: one row per SKU name (a listing's alias SKUs each get a row),
    with its sales so the gaps can be filled in order of importance. Outsourced products are not included."""
    m = pf[pf["fabric_source"].isin(list(MISSING_LABEL))][["asin", "main_sku", "category", "fabric_source", "size", "metres_per_unit", "title"]]
    m = m.merge(products[["asin", "skus"]], on="asin", how="left")
    m["sku"] = (m["main_sku"].astype(str) + ", " + m["skus"].astype(str)).str.split(", ")
    x = m.explode("sku").drop(columns="skus")
    x["sku"] = x["sku"].str.strip()
    x = x[~x["sku"].isin(["", "nan", "None"])].drop_duplicates(["sku", "asin"])
    end = d["date"].max()
    key = pd.Series(list(zip(x["sku"], x["asin"])), index=x.index)
    x["units_year"] = key.map(d.groupby(["sku", "asin"])["quantity"].sum()).fillna(0).astype(int)
    x["units_8w"] = key.map(d[d["date"] > end - pd.Timedelta(weeks=8)].groupby(["sku", "asin"])["quantity"].sum()).fillna(0).astype(int)
    x["priority"] = pd.cut(x["units_8w"], [-1, 0, 19, 99, 10 ** 9], labels=PRIORITY[::-1]).astype(str)
    x["fabric_data"] = x["fabric_source"].map(MISSING_LABEL)
    x["sku_range"] = x["main_sku"].str.extract(r"^([A-Za-z]+(?:\d+-\d+|[-_]?\d+)?)")[0].fillna(x["main_sku"])
    x = x.rename(columns={"size": "estimated_size", "metres_per_unit": "estimated_metres_per_unit"})
    cols = ["sku", "category", "sku_range", "fabric_data", "priority", "units_year", "units_8w", "estimated_size", "estimated_metres_per_unit", "asin", "main_sku", "title"]
    return x[cols].sort_values(["units_year", "sku"], ascending=[False, True]).reset_index(drop=True)


# ------------------------------------------------------------------ consumption
# Order-line detail the fabric views never read. Leaving it out of the merged table keeps it to about a third of the size,
# which matters on a hosted server where every open page holds its own copy.
_LINE_DETAIL = ["product_name", "temp_band", "rain_band", "source_file", "order_id", "order_item_id", "district_source", "sales_channel", "city",
                "order_status", "fulfillment", "pin", "ts", "last_updated", "item_price", "promo_discount", "unit_price", "has_promo", "is_business",
                "is_cancelled", "is_demand", "is_amazon"]


def with_metres(d: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """Demand rows + metres (units x metres per unit) and the fabric facts; metres is NaN where the fabric is unknown."""
    cols = ["asin", "fabric_code", "fabric_type", "fabric_family", "fabric_product", "product_type", "colour", "size", "metres_per_unit", "fabric_source"]
    x = d.drop(columns=[c for c in _LINE_DETAIL if c in d.columns]).merge(pf[cols], on="asin", how="left")
    x["metres"] = x["quantity"] * x["metres_per_unit"]
    x["fabric_key"] = x["fabric_code"].where(x["fabric_source"] != "inferred: size rule", UNKNOWN)
    return x


def _known(dm: pd.DataFrame, by: str) -> pd.DataFrame:
    k = dm.dropna(subset=["metres"])
    return k[~k["fabric_code"].isin([UNKNOWN, "(none)"])] if by == "fabric_code" else k


def by_fabric(dm: pd.DataFrame, by: str = "fabric_code") -> pd.DataFrame:
    """Metres, units, share and 4-week momentum per fabric code / type / family / product type."""
    k = _known(dm, by)
    g = k.groupby(by).agg(metres=("metres", "sum"), units=("quantity", "sum"), products=("asin", "nunique"))
    if by == "fabric_code":
        first = k.drop_duplicates(by).set_index(by)
        g["fabric_type"], g["fabric_family"] = first["fabric_type"], first["fabric_family"]
        g["colours"] = k.groupby(by)["colour"].nunique()
    g["share_%"] = (g["metres"] / g["metres"].sum() * 100).round(1)
    w = A.weekly_matrix(k.assign(quantity=k["metres"]), by)                 # weekly metres
    if w.shape[1] >= 8:
        g["last_4w_m"] = w.iloc[:, -4:].sum(axis=1).round(0)
        g["prev_4w_m"] = w.iloc[:, -8:-4].sum(axis=1).round(0)
        g["growth_4w_%"] = np.round((g["last_4w_m"] / g["prev_4w_m"].replace(0, np.nan) - 1) * 100, 1)
    g["metres"] = g["metres"].round(0)
    return g.sort_values("metres", ascending=False)


def monthly(dm: pd.DataFrame, by: str = "fabric_code", top: int = 12) -> pd.DataFrame:
    k = _known(dm, by)
    m = k.pivot_table(index=by, columns="month", values="metres", aggfunc="sum", fill_value=0).round(0)
    return m.loc[m.sum(axis=1).sort_values(ascending=False).index[:top]]


def requirement(d: pd.DataFrame, pf: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Metres of each fabric code needed for the next `horizon` weeks: the product-level weekly forecast (baseline and festive
    scenario) x metres per unit, summed per fabric code. Products known only by the size rule are reported on one row of their own."""
    fc = F.forecast(d, "asin", horizon)
    if fc.empty:
        return pd.DataFrame()
    m = pf.set_index("asin")
    fc = fc.join(m[["fabric_code", "fabric_type", "fabric_family", "metres_per_unit", "fabric_source"]], on="group")
    fc = fc[fc["metres_per_unit"].notna()].copy()
    fc["baseline_m"], fc["festive_m"] = fc["baseline"] * fc["metres_per_unit"], fc["festive_scenario"] * fc["metres_per_unit"]
    fc["key"] = fc["fabric_code"].where(fc["fabric_source"] != "inferred: size rule", UNKNOWN)
    g = fc.groupby("key").agg(fabric_type=("fabric_type", "first"), fabric_family=("fabric_family", "first"), products=("group", "nunique"),
                              baseline_m=("baseline_m", "sum"), festive_m=("festive_m", "sum"))
    recent = with_metres(d[d["date"] > d["date"].max() - pd.Timedelta(weeks=horizon)], pf)
    g[f"last_{horizon}w_actual_m"] = recent.groupby("fabric_key")["metres"].sum()
    return g.round(0).sort_values("festive_m", ascending=False)


def monthly_requirement(pred: pd.DataFrame, d: pd.DataFrame, pf: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Long-range metres by fabric code and month from the ML category prediction: each category's monthly units are split
    across its products by their last-12-week share (products sold in the last 28 days only), then multiplied by metres per unit."""
    end = d["date"].max()
    recent = d[d["date"] > end - pd.Timedelta(weeks=12)]
    alive = recent.loc[recent["date"] > end - pd.Timedelta(days=28), "asin"].unique()
    mix = recent[recent["asin"].isin(alive)].groupby(["category", "asin"])["quantity"].sum()
    mix = (mix / mix.groupby("category").transform("sum")).rename("mix").reset_index()
    mix = mix.join(pf.set_index("asin")[["fabric_code", "fabric_type", "metres_per_unit", "fabric_source"]], on="asin").dropna(subset=["metres_per_unit"])
    mix["key"] = mix["fabric_code"].where(mix["fabric_source"] != "inferred: size rule", UNKNOWN)
    p = pred.groupby(["category", "month"])["units"].sum().reset_index()
    p = p[p["month"].isin(sorted(p["month"].unique())[:months])]
    x = p.merge(mix, on="category")
    x["metres"] = x["units"] * x["mix"] * x["metres_per_unit"]
    t = x.pivot_table(index="key", columns="month", values="metres", aggfunc="sum", fill_value=0).round(0)
    t.insert(0, "fabric_type", x.drop_duplicates("key").set_index("key")["fabric_type"].reindex(t.index))
    return t.loc[t.iloc[:, 1:].sum(axis=1).sort_values(ascending=False).index]
