import ast
import csv
import gzip
import io
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV JSON Flattener (Wide)", layout="wide")

st.title("CSV JSON Flattener (Wide)")
st.write(
    "Upload a CSV where some columns contain embedded JSON (or Python-literal dict/list strings). "
    "The app flattens those fields into additional columns while keeping **one output row per input row**."
)


def parse_obj(x: Any) -> Optional[Any]:
    """Parse a cell that may contain JSON or a Python-literal dict/list.

    Returns a Python object (dict/list) or None if parsing fails.
    """
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (dict, list)):
        return x

    s = str(x).strip()
    if (not s) or s.lower() == "nan":
        return None

    # Strict JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Python-literal style (single quotes)
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def detect_delimiter(text_sample: str) -> str:
    """Detect a likely CSV delimiter.

    NOTE: We intentionally do NOT consider ':' as a delimiter because JSON/Python-literal dicts contain many colons.
    """
    candidates = [",", "	", ";", "|"]

    # Use a handful of lines; csv.reader will respect double-quoted fields.
    lines = [ln for ln in text_sample.splitlines() if ln.strip()][:25]
    if not lines:
        return ","

    best_delim = ","
    best_consistency = -1.0
    best_cols = -1

    for d in candidates:
        try:
            rdr = csv.reader(lines, delimiter=d, quotechar='"', doublequote=True)
            counts = [len(row) for row in rdr]
            if not counts:
                continue

            # Mode column count
            mode_cols = max(set(counts), key=counts.count)
            consistency = counts.count(mode_cols) / len(counts)

            # Ignore delimiters that don't actually split the header/rows
            if mode_cols <= 1:
                continue

            # Prefer higher consistency, then more columns
            if (consistency > best_consistency) or (consistency == best_consistency and mode_cols > best_cols):
                best_consistency = consistency
                best_cols = mode_cols
                best_delim = d
        except Exception:
            continue

    return best_delim


def maybe_decompress(file_bytes: bytes, filename: str) -> bytes:
    """Support .gz uploads."""
    if filename.lower().endswith(".gz"):
        return gzip.decompress(file_bytes)
    return file_bytes


@st.cache_data(show_spinner=False)
def read_csv_cached(
    file_bytes: bytes,
    filename: str,
    encoding: str,
    skiprows: int,
    delimiter_mode: str,
    manual_delimiter: str,
    read_all_as_text: bool,
) -> Tuple[pd.DataFrame, str]:
    raw = maybe_decompress(file_bytes, filename)

    if delimiter_mode == "Auto":
        try:
            sample_text = raw[:64000].decode(encoding, errors="replace")
            delim = detect_delimiter(sample_text)
        except Exception:
            delim = ","
    else:
        delim = manual_delimiter

    dtype = str if read_all_as_text else None

    df = pd.read_csv(
        io.BytesIO(raw),
        encoding=encoding,
        sep=delim,
        skiprows=int(skiprows),
        dtype=dtype,
        quotechar='"',
        engine="c",
    )
    return df, delim


@st.cache_data(show_spinner=False)
def flatten_wide_cached(
    df: pd.DataFrame,
    json_cols: Tuple[str, ...],
    list_col: Optional[str],
    max_items: int,
    name_style: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten selected columns into a wide table."""

    def normalize_dict_column(series: pd.Series, prefix: str) -> Tuple[pd.DataFrame, int]:
        parsed = series.map(parse_obj)
        ok_count = int(parsed.map(lambda v: isinstance(v, dict)).sum())
        parsed = parsed.map(lambda v: v if isinstance(v, dict) else {})
        flat = pd.json_normalize(parsed)
        if flat.shape[1] > 0:
            flat.columns = [f"{prefix}.{c}" for c in flat.columns]
        return flat, ok_count

    def widen_list_of_dicts(
        series: pd.Series,
        prefix: str,
        max_items: int,
        name_style: str,
    ) -> Tuple[pd.DataFrame, int, int, int]:
        parsed = series.map(parse_obj)
        ok_count = int(parsed.map(lambda v: isinstance(v, list)).sum())
        max_len_observed = int(parsed.map(lambda v: len(v) if isinstance(v, list) else 0).max())

        # Union of keys
        keys: List[str] = []
        seen = set()
        for v in parsed:
            if not isinstance(v, list):
                continue
            for item in v[:max_items]:
                if isinstance(item, dict):
                    for k in item.keys():
                        if k not in seen:
                            seen.add(k)
                            keys.append(k)

        def col_name(i0: int, key: str) -> str:
            if name_style == "Bracket":
                return f"{prefix}[{i0}].{key}"
            return f"{prefix}_{i0 + 1}_{key}"

        out: Dict[str, List[Any]] = {col_name(i, k): [] for i in range(max_items) for k in keys}

        for v in parsed:
            if not isinstance(v, list):
                for i in range(max_items):
                    for k in keys:
                        out[col_name(i, k)].append(None)
                continue

            for i in range(max_items):
                item = v[i] if i < len(v) else None
                for k in keys:
                    out[col_name(i, k)].append(item.get(k) if isinstance(item, dict) else None)

        wide_df = pd.DataFrame(out)
        return wide_df, ok_count, max_len_observed, len(keys)

    summary_rows: List[Dict[str, Any]] = []
    parts: List[pd.DataFrame] = []

    # Dict-like columns
    dict_cols = [c for c in json_cols if c != list_col]
    for c in dict_cols:
        flat, ok = normalize_dict_column(df[c], c)
        parts.append(flat)
        summary_rows.append(
            {"column": c, "type": "dict-like", "rows_parsed_as_dict": ok, "new_columns": int(flat.shape[1])}
        )

    # List-of-dicts column
    if list_col is not None:
        if max_items > 0:
            wide_df, ok, max_len, key_count = widen_list_of_dicts(df[list_col], list_col, int(max_items), name_style)
            parts.append(wide_df)
            summary_rows.append(
                {
                    "column": list_col,
                    "type": "list-of-dicts",
                    "rows_parsed_as_list": ok,
                    "max_len_observed": max_len,
                    "keys_union_count": key_count,
                    "new_columns": int(wide_df.shape[1]),
                }
            )
        else:
            parsed = df[list_col].map(parse_obj)
            summary_rows.append(
                {
                    "column": list_col,
                    "type": "list-of-dicts",
                    "rows_parsed_as_list": int(parsed.map(lambda v: isinstance(v, list)).sum()),
                    "max_len_observed": int(parsed.map(lambda v: len(v) if isinstance(v, list) else 0).max()),
                    "keys_union_count": 0,
                    "new_columns": 0,
                }
            )

    base = df.drop(columns=list(json_cols), errors="ignore")
    flat_df = pd.concat([base] + parts, axis=1) if parts else base
    summary_df = pd.DataFrame(summary_rows)
    return flat_df, summary_df


with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload CSV (optionally .gz)", type=["csv", "txt", "gz"], accept_multiple_files=False)

    st.subheader("CSV reading")
    encoding = st.text_input("Encoding", value="utf-8")
    skiprows = st.number_input("skiprows", min_value=0, value=0, step=1)

    delimiter_mode = st.selectbox("Delimiter", options=["Auto", "Manual"], index=0)
    manual_delimiter = st.text_input("Manual delimiter", value=",", disabled=(delimiter_mode != "Manual"))

    read_all_as_text = st.checkbox(
        "Read all columns as text (recommended)",
        value=True,
        help="Avoids pandas type inference and preserves IDs / mixed-type columns.",
    )

    st.subheader("Flattening")
    name_style = st.selectbox("Column naming", options=["Bracket", "Underscore"], index=0)


# Safe first launch
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

progress = st.progress(0)
status = st.empty()

status.write("Reading CSV…")
progress.progress(15)

try:
    df, detected_delim = read_csv_cached(
        file_bytes=uploaded.getvalue(),
        filename=uploaded.name,
        encoding=encoding,
        skiprows=int(skiprows),
        delimiter_mode=delimiter_mode,
        manual_delimiter=manual_delimiter,
        read_all_as_text=read_all_as_text,
    )
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

progress.progress(35)
st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
if delimiter_mode == "Auto":
    st.caption(f"Detected delimiter: `{detected_delim}`")

# Default JSON columns
default_cols = ["surfaced_articles", "quick_feedback", "relevance_breakdown", "dropoff_breakdown"]
existing_defaults = [c for c in default_cols if c in df.columns]
missing = [c for c in default_cols if c not in df.columns]
if missing:
    st.warning("Expected JSON columns not found: " + ", ".join(missing))

json_cols = st.multiselect(
    "Select columns to parse/flatten",
    options=list(df.columns),
    default=existing_defaults,
)

if not json_cols:
    st.info("Select at least one column to parse/flatten.")
    st.stop()

# Identify list-like candidates
list_like_candidates: List[str] = []
for c in json_cols:
    sample = df[c].dropna().head(20).map(parse_obj)
    if c == "surfaced_articles" or any(isinstance(v, list) for v in sample):
        list_like_candidates.append(c)

list_col: Optional[str] = None
if list_like_candidates:
    default_index = 1 if "surfaced_articles" in list_like_candidates else 0
    list_col = st.selectbox(
        "Treat ONE selected column as list-of-dicts (widened with indices)",
        options=[None] + list_like_candidates,
        index=default_index,
        help="Pick the column that contains a JSON array / Python list of objects.",
    )

max_items = 0
if list_col is not None:
    parsed = df[list_col].dropna().head(200).map(parse_obj)
    max_observed = int(parsed.map(lambda v: len(v) if isinstance(v, list) else 0).max())
    max_items = st.number_input(
        f"Max items to widen from '{list_col}'",
        min_value=0,
        value=min(max_observed, 10) if max_observed > 0 else 0,
        step=1,
        help="Higher values create more columns.",
    )

status.write("Flattening…")
progress.progress(60)

flat, summary_df = flatten_wide_cached(
    df=df,
    json_cols=tuple(json_cols),
    list_col=list_col,
    max_items=int(max_items),
    name_style=name_style,
)

progress.progress(100)
status.empty()

st.subheader("Flattening summary")
st.dataframe(summary_df, use_container_width=True)

st.subheader("Preview")
st.dataframe(flat.head(50), use_container_width=True)

st.subheader("Download")
filename = st.text_input("Output filename", value="flattened_wide.csv")
csv_text = flat.to_csv(index=False, encoding="utf-8")
csv_bytes = csv_text.encode("utf-8")

st.download_button(
    "Download flattened CSV",
    data=csv_bytes,
    file_name=filename,
    mime="text/csv",
)

with st.expander("Notes / Troubleshooting"):
    notes = [
        "- If a field uses single quotes (e.g., `{'positive': 1}`), strict JSON parsing fails; the app falls back to Python-literal parsing.",
        "- If parsing fails for a cell, it becomes empty in the flattened output.",
        "- For list columns (like `surfaced_articles`), widening can create many columns quickly—use **Max items** to cap width.",
        "- For very large files on Community Cloud, consider uploading `.csv.gz`.",
    ]
    st.markdown(chr(10).join(notes))
