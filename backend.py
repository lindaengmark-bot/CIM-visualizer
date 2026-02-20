"""
backend.py

Core parsing, normalization, aggregation, and export helpers for the LLM Visualizer Streamlit app.

Supports two export types:
- Basic "LLM tracker" exports (multiple LLMs, shallow data)
- Advanced exports (deeper data for ChatGPT and Gemini, with Results, Mentions, Citations, SearchQueries)

Design goals:
- Fast enough for thousands of rows
- Robust parsing of brands fields (string lists, comma-separated, bracketed, mixed formats)
- Provides tidy "long" tables for mentions (brand-level) and citations (domain-level)
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import ast
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


BASIC_SHEETS = {"answers", "brand_mentions"}
ADVANCED_SHEETS = {"Results", "Mentions", "Citations", "SearchQueries"}


@dataclass(frozen=True)
class ExportInfo:
    export_type: str  # "basic" or "advanced"
    sheet_names: List[str]


def _normalize_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_brand(brand: str) -> str:
    s = _normalize_text(brand)
    # remove wrapping quotes and brackets noise
    s = re.sub(r"^[\[\(\{\"\']+|[\]\)\}\"\']+$", "", s).strip()
    # collapse commas that come from "['A', 'B']" mishaps
    s = re.sub(r"\s*,\s*", ", ", s)
    return s


def normalize_domain(domain: str) -> str:
    s = _normalize_text(domain).lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/")[0]
    s = s.replace("www.", "")
    return s


def detect_export_type(sheet_names: Iterable[str]) -> ExportInfo:
    names = list(sheet_names)
    sset = set(names)
    if ADVANCED_SHEETS.intersection(sset) and ("Citations" in sset or "Results" in sset):
        return ExportInfo(export_type="advanced", sheet_names=names)
    if BASIC_SHEETS.intersection(sset):
        return ExportInfo(export_type="basic", sheet_names=names)
    # fallback, treat as basic-like and let downstream inference handle it
    return ExportInfo(export_type="basic", sheet_names=names)


def list_sheet_names(file_bytes: bytes) -> List[str]:
    xf = pd.ExcelFile(BytesIO(file_bytes))
    return list(xf.sheet_names)


def read_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)


def read_sheets(file_bytes: bytes, sheet_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    xf = pd.ExcelFile(BytesIO(file_bytes))
    names = sheet_names or list(xf.sheet_names)
    out: Dict[str, pd.DataFrame] = {}
    for s in names:
        out[s] = xf.parse(s)
    return out


def _parse_brands_cell(cell: object) -> List[str]:
    """
    Accepts a cell that may be:
    - an actual list (already parsed)
    - a stringified Python list like "['ECHO', 'STIHL']"
    - a comma-separated string like "ECHO, STIHL"
    - a bracketed/comma style like "[ECHO, STIHL]" or "ECHO (STIHL)"
    - empty / NaN
    Returns a list of brand strings.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        return [normalize_brand(x) for x in cell if _normalize_text(x)]
    s = str(cell).strip()
    if not s or s == "[]":
        return []

    # Try literal_eval for python-like lists
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [normalize_brand(x) for x in v if _normalize_text(x)]
        except Exception:
            pass

    # Remove surrounding brackets/quotes, then split
    s2 = re.sub(r"^[\[\(\{]+|[\]\)\}]+$", "", s).strip()
    # Some exports use single quotes around items inside a string
    s2 = s2.replace("'", "").replace('"', "")
    # Split on comma, semicolon, pipe, or newline
    parts = re.split(r"\s*[,;|\n]\s*", s2)
    parts = [normalize_brand(p) for p in parts if _normalize_text(p)]
    # If it was a single token that still contains multiple separated by " / "
    if len(parts) == 1 and " / " in parts[0]:
        parts = [normalize_brand(p) for p in parts[0].split(" / ") if _normalize_text(p)]
    return [p for p in parts if p]


def apply_manual_mappings(brands: pd.Series, mapping_lines: str) -> pd.Series:
    """
    mapping_lines format (one per line):
      alias=canonical
    Example:
      bosch=STIHL
      stihl=STIHL
    Mapping is applied case-insensitively on full-string matches after normalize_brand.
    """
    if not mapping_lines or not mapping_lines.strip():
        return brands

    mapping: Dict[str, str] = {}
    for raw in mapping_lines.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        left, right = raw.split("=", 1)
        left = normalize_brand(left).lower()
        right = normalize_brand(right)
        if left and right:
            mapping[left] = right

    if not mapping:
        return brands

    # vectorized map
    lower = brands.fillna("").astype(str).map(lambda x: normalize_brand(x).lower())
    mapped = lower.map(mapping)
    return np.where(mapped.notna(), mapped, brands).astype(str)


def build_mentions_long(
    df: pd.DataFrame,
    brand_col: str = "brands",
    inferred_brand_col: Optional[str] = None,
    keep_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Builds a tidy mentions table:
      one row per mention, with columns: brand, and selected metadata columns.
    """
    if brand_col not in df.columns:
        raise ValueError(f"Brand column '{brand_col}' not found in dataframe.")

    base = df.copy()
    keep = keep_cols or [c for c in df.columns if c != brand_col]
    keep = [c for c in keep if c in base.columns]

    # Parse and explode brands
    parsed = base[brand_col].map(_parse_brands_cell)
    out = base[keep].copy()
    out["brand"] = parsed
    out = out.explode("brand", ignore_index=True)
    out["brand"] = out["brand"].fillna("").map(normalize_brand)
    out = out[out["brand"] != ""]

    # Optional inferred brands (advanced Results often has inferred_brands)
    if inferred_brand_col and inferred_brand_col in df.columns:
        parsed_inf = base[inferred_brand_col].map(_parse_brands_cell)
        tmp = base[keep].copy()
        tmp["brand"] = parsed_inf
        tmp = tmp.explode("brand", ignore_index=True)
        tmp["brand"] = tmp["brand"].fillna("").map(normalize_brand)
        tmp = tmp[tmp["brand"] != ""]
        tmp["brand_source"] = "inferred"
        out["brand_source"] = "explicit"
        out = pd.concat([out, tmp], ignore_index=True)

    # Standardize common metadata columns if present
    for col in ["platform", "model", "country"]:
        if col in out.columns:
            out[col] = out[col].map(_normalize_text)
    if "run_date" in out.columns:
        out["run_date"] = pd.to_datetime(out["run_date"], errors="coerce")

    return out


def build_citations_long(df: pd.DataFrame, domain_col: str = "domain", keep_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if domain_col not in df.columns:
        raise ValueError(f"Domain column '{domain_col}' not found in dataframe.")

    base = df.copy()
    keep = keep_cols or [c for c in base.columns if c != domain_col]
    keep = [c for c in keep if c in base.columns]
    out = base[keep].copy()
    out["domain"] = base[domain_col].map(_normalize_text).map(normalize_domain)
    out = out[out["domain"] != ""]
    if "run_date" in out.columns:
        out["run_date"] = pd.to_datetime(out["run_date"], errors="coerce")
    for col in ["platform", "model", "country", "parent_domain", "domain_classification", "domain_subclassification"]:
        if col in out.columns:
            out[col] = out[col].map(_normalize_text)
    return out


def aggregate_counts(
    long_df: pd.DataFrame,
    item_col: str,
    group_cols: List[str],
    top_n: int = 25,
    include_other: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (counts_table, share_table).
    counts_table: rows are group_cols (multi-index) and columns are items
    share_table:  same shape, but row-wise shares
    """
    df = long_df.copy()
    for c in group_cols:
        if c not in df.columns:
            df[c] = "Unknown"
    if item_col not in df.columns:
        raise ValueError(f"item_col '{item_col}' not in long_df.")

    # Determine global top items
    top_items = df[item_col].value_counts().head(top_n).index.tolist()

    df["_item"] = np.where(df[item_col].isin(top_items), df[item_col], "Other" if include_other else df[item_col])
    grouped = df.groupby(group_cols + ["_item"], dropna=False).size().reset_index(name="count")

    pivot = grouped.pivot_table(index=group_cols, columns="_item", values="count", fill_value=0, aggfunc="sum")
    pivot = pivot.sort_index()

    # shares per row
    denom = pivot.sum(axis=1).replace(0, np.nan)
    share = pivot.div(denom, axis=0).fillna(0.0)

    # nice ordering: top items then Other
    ordered_cols = [c for c in top_items if c in pivot.columns]
    if include_other and "Other" in pivot.columns:
        ordered_cols.append("Other")
    remaining = [c for c in pivot.columns if c not in ordered_cols]
    pivot = pivot[ordered_cols + remaining]
    share = share[pivot.columns]
    return pivot, share


def sentiment_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if "sentiment" not in df.columns:
        raise ValueError("No 'sentiment' column present.")
    base = df.copy()
    for c in group_cols:
        if c not in base.columns:
            base[c] = "Unknown"
    base["sentiment"] = pd.to_numeric(base["sentiment"], errors="coerce")
    out = base.groupby(group_cols)["sentiment"].agg(["count", "mean", "median", "min", "max"]).reset_index()
    return out


def build_enhanced_excel(
    raw_sheets: Dict[str, pd.DataFrame],
    mentions_long: Optional[pd.DataFrame] = None,
    citations_long: Optional[pd.DataFrame] = None,
    mention_counts: Optional[pd.DataFrame] = None,
    mention_share: Optional[pd.DataFrame] = None,
    citation_counts: Optional[pd.DataFrame] = None,
    citation_share: Optional[pd.DataFrame] = None,
) -> bytes:
    """
    Builds an exportable xlsx in-memory.
    - Includes original sheets
    - Adds cleaned long tables
    - Adds summary tables (counts and share)
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Original sheets
        for name, df in raw_sheets.items():
            safe = re.sub(r"[:\\/?*\[\]]", "_", name)[:31]
            df.to_excel(writer, sheet_name=safe, index=False)

        # Derived
        if mentions_long is not None:
            mentions_long.to_excel(writer, sheet_name="Mentions_Long", index=False)
        if citations_long is not None:
            citations_long.to_excel(writer, sheet_name="Citations_Long", index=False)

        def _write_summary(df: Optional[pd.DataFrame], sheet: str):
            if df is None:
                return
            if isinstance(df.index, pd.MultiIndex):
                df.reset_index().to_excel(writer, sheet_name=sheet[:31], index=False)
            else:
                df.to_excel(writer, sheet_name=sheet[:31], index=False)

        _write_summary(mention_counts, "Mentions_Counts")
        _write_summary(mention_share, "Mentions_Share")
        _write_summary(citation_counts, "Citations_Counts")
        _write_summary(citation_share, "Citations_Share")

    return output.getvalue()
