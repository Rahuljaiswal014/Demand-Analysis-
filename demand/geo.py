"""State name normalisation and representative coordinates for weather lookup.

`ship-state` in the Amazon report is free text typed by buyers (215 spellings in the
first year). Resolution order: exact canonical match -> known alias -> fuzzy match
-> majority state of the same 3-digit PIN prefix (learned from the clean rows).
"""
import difflib
import re
from pathlib import Path

import pandas as pd

PIN_DISTRICT_CSV = Path(__file__).resolve().parent.parent / "data" / "reference" / "pin_district.csv"

# canonical state -> (lat, lon) of its largest demand centre, used as the weather point
STATES = {
    "ANDAMAN AND NICOBAR ISLANDS": (11.62, 92.73), "ANDHRA PRADESH": (16.51, 80.65),
    "ARUNACHAL PRADESH": (27.08, 93.61), "ASSAM": (26.14, 91.74), "BIHAR": (25.59, 85.14),
    "CHANDIGARH": (30.73, 76.78), "CHHATTISGARH": (21.25, 81.63),
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": (20.40, 72.83), "DELHI": (28.61, 77.21),
    "GOA": (15.49, 73.83), "GUJARAT": (23.02, 72.57), "HARYANA": (28.46, 77.03),
    "HIMACHAL PRADESH": (31.10, 77.17), "JAMMU AND KASHMIR": (32.73, 74.86),
    "JHARKHAND": (23.34, 85.31), "KARNATAKA": (12.97, 77.59), "KERALA": (9.93, 76.27),
    "LADAKH": (34.15, 77.58), "LAKSHADWEEP": (10.57, 72.64), "MADHYA PRADESH": (22.72, 75.86),
    "MAHARASHTRA": (19.08, 72.88), "MANIPUR": (24.82, 93.94), "MEGHALAYA": (25.58, 91.89),
    "MIZORAM": (23.73, 92.72), "NAGALAND": (25.91, 93.73), "ODISHA": (20.30, 85.82),
    "PUDUCHERRY": (11.94, 79.81), "PUNJAB": (30.90, 75.85), "RAJASTHAN": (26.91, 75.79),
    "SIKKIM": (27.33, 88.61), "TAMIL NADU": (13.08, 80.27), "TELANGANA": (17.39, 78.49),
    "TRIPURA": (23.83, 91.29), "UTTAR PRADESH": (26.85, 80.95), "UTTARAKHAND": (30.32, 78.03),
    "WEST BENGAL": (22.57, 88.36),
}

ALIASES = {
    "NEW DELHI": "DELHI", "NCT OF DELHI": "DELHI", "ORISSA": "ODISHA", "PONDICHERRY": "PUDUCHERRY",
    "UTTARANCHAL": "UTTARAKHAND", "JAMMU KASHMIR": "JAMMU AND KASHMIR", "J AND K": "JAMMU AND KASHMIR",
    "ANDAMAN AND NICOBAR": "ANDAMAN AND NICOBAR ISLANDS", "DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DADRA AND NAGAR HAVELI": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "AP": "ANDHRA PRADESH", "AR": "ARUNACHAL PRADESH", "AS": "ASSAM", "BR": "BIHAR", "CG": "CHHATTISGARH",
    "CH": "CHANDIGARH", "DL": "DELHI", "GA": "GOA", "GJ": "GUJARAT", "HR": "HARYANA", "HP": "HIMACHAL PRADESH",
    "JK": "JAMMU AND KASHMIR", "JH": "JHARKHAND", "KA": "KARNATAKA", "KL": "KERALA", "MP": "MADHYA PRADESH",
    "MH": "MAHARASHTRA", "MN": "MANIPUR", "ML": "MEGHALAYA", "MZ": "MIZORAM", "NL": "NAGALAND", "OD": "ODISHA",
    "OR": "ODISHA", "PB": "PUNJAB", "PY": "PUDUCHERRY", "RJ": "RAJASTHAN", "SK": "SIKKIM", "TN": "TAMIL NADU",
    "TS": "TELANGANA", "TG": "TELANGANA", "TR": "TRIPURA", "UP": "UTTAR PRADESH", "UK": "UTTARAKHAND",
    "UA": "UTTARAKHAND", "WB": "WEST BENGAL",
}

# coarse region used in geography roll-ups
REGION = {
    "North": ["DELHI", "HARYANA", "PUNJAB", "CHANDIGARH", "HIMACHAL PRADESH", "JAMMU AND KASHMIR", "LADAKH",
              "UTTAR PRADESH", "UTTARAKHAND", "RAJASTHAN"],
    "South": ["KARNATAKA", "TAMIL NADU", "TELANGANA", "ANDHRA PRADESH", "KERALA", "PUDUCHERRY", "LAKSHADWEEP",
              "ANDAMAN AND NICOBAR ISLANDS"],
    "West": ["MAHARASHTRA", "GUJARAT", "GOA", "DADRA AND NAGAR HAVELI AND DAMAN AND DIU"],
    "East": ["WEST BENGAL", "ODISHA", "BIHAR", "JHARKHAND"],
    "Central": ["MADHYA PRADESH", "CHHATTISGARH"],
    "North-East": ["ASSAM", "ARUNACHAL PRADESH", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "SIKKIM", "TRIPURA"],
}
STATE_TO_REGION = {s: r for r, ss in REGION.items() for s in ss}


def _squash(s: str) -> str:
    s = re.sub(r"&", " AND ", str(s).upper())
    s = re.sub(r"[^A-Z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _resolve_text(raw: str) -> str | None:
    s = _squash(raw)
    if not s:
        return None
    if s in STATES:
        return s
    if s in ALIASES:
        return ALIASES[s]
    nospace = s.replace(" ", "")
    for canon in STATES:
        if canon.replace(" ", "") == nospace:
            return canon
    hit = difflib.get_close_matches(s, list(STATES), n=1, cutoff=0.82)
    return hit[0] if hit else None


def normalise_states(state: pd.Series, pin: pd.Series) -> pd.Series:
    """Return canonical state names; unresolved rows become 'UNKNOWN'."""
    lookup = {raw: _resolve_text(raw) for raw in state.dropna().unique()}
    out = state.map(lookup)

    pin3 = pin.astype("string").str.extract(r"(\d{3})")[0]
    known = out.notna() & pin3.notna()
    if known.any():
        majority = (pd.DataFrame({"pin3": pin3[known], "state": out[known]})
                    .groupby("pin3")["state"].agg(lambda s: s.value_counts().index[0]))
        out = out.fillna(pin3.map(majority))
    return out.fillna("UNKNOWN")


STATE_ABBR = {}
for _abbr, _state in ALIASES.items():
    if len(_abbr) == 2:
        STATE_ABBR.setdefault(_state, _abbr)
STATE_ABBR.update({"ANDAMAN AND NICOBAR ISLANDS": "AN", "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DN", "LADAKH": "LA", "LAKSHADWEEP": "LD"})


def add_district(pin: pd.Series, state: pd.Series) -> pd.DataFrame:
    """District for every order line, from its PIN code via the India Post directory (data/reference/pin_district.csv).

    Buyer-typed city names are unusable as a key (9,500 spellings: 'BANGALORW', 'BANGALORE- 78', ...), while the PIN is
    validated by Amazon at checkout. Checked against this data: 98.8% of rows have a PIN in the directory, and where they
    do the directory's state agrees with the order's state 99.93% of the time. PINs newer than the directory (1.2%) take
    the majority district of the PINs sharing their first four digits, then first three.

    Returns two columns: `district` (unique label such as 'PUNE, MH', since names like AURANGABAD exist in two states)
    and `district_source` ('directory' | 'nearby PIN' | 'none')."""
    ref = pd.read_csv(PIN_DISTRICT_CSV, dtype=str)
    exact = ref.set_index("pin")["district"]
    near4 = ref.groupby(ref["pin"].str[:4])["district"].agg(lambda x: x.value_counts().index[0])
    near3 = ref.groupby(ref["pin"].str[:3])["district"].agg(lambda x: x.value_counts().index[0])

    name = pin.map(exact)
    source = pd.Series("directory", index=pin.index).where(name.notna())
    nearby = pin.str[:4].map(near4).fillna(pin.str[:3].map(near3))
    source = source.fillna(pd.Series("nearby PIN", index=pin.index).where(nearby.notna())).fillna("none")
    name = name.fillna(nearby)
    name = name.str.replace(r"\s*\(.*\)$", "", regex=True)               # directory disambiguators such as 'RAIGARH(MH)'
    label = (name + ", " + state.map(STATE_ABBR).fillna("??")).where(name.notna(), "UNKNOWN")
    return pd.DataFrame({"district": label, "district_source": source})
