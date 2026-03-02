"""
app.py

Streamlit frontend for the LLM Visualizer.
Upload a Basic or Advanced export, explore charts, and export an enhanced Excel file.

Run locally:
  streamlit run app.py
"""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import backend
import importlib


def _brand_merge_mapping_from_editor(df: pd.DataFrame) -> dict:
    """Convert edited suggestions table to alias->canonical mapping."""
    if df is None or getattr(df, "empty", True) or "combine" not in df.columns:
        return {}
    chosen = df[df["combine"] == True]
    mapping = {}
    for _, r in chosen.iterrows():
        alias = str(r.get("alias", "")).strip()
        canonical = str(r.get("canonical", "")).strip()
        if alias and canonical and alias != canonical:
            mapping[alias] = canonical
    return mapping


try:
    import plotly.express as px
except Exception:
    px = None


st.set_page_config(page_title="LLM Visualizer", layout="wide")


@st.cache_data(show_spinner=False)
def _sheet_names(file_bytes: bytes) -> List[str]:
    return backend.list_sheet_names(file_bytes)


@st.cache_data(show_spinner=True)
def _read_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return backend.read_sheet(file_bytes, sheet_name)


@st.cache_data(show_spinner=False)
def _read_sheets(file_bytes: bytes, sheet_names: List[str]) -> Dict[str, pd.DataFrame]:
    return backend.read_sheets(file_bytes, sheet_names)



def _pivot_to_long(df_pivot: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Convert a pivot table (index=group(s), columns=item) to long form with a readable group label.
    """
    if df_pivot is None or df_pivot.empty:
        return pd.DataFrame(columns=["group", "item", value_name])

    idx_cols = [c for c in (df_pivot.index.names or []) if c is not None]
    long_df = df_pivot.reset_index().melt(id_vars=idx_cols, var_name="item", value_name=value_name)
    long_df = long_df[long_df[value_name].notna()]

    if idx_cols:
        long_df["group"] = long_df[idx_cols].astype(str).agg(" | ".join, axis=1)
    else:
        long_df["group"] = "All"

    return long_df


def _plot_clustered_bars(df_pivot: pd.DataFrame, value_name: str, title: str, as_percent: bool = False):
    """
    Clustered (grouped) columns:
    - x axis: item (brand or domain)
    - color: group (platform, model, etc.)
    - y axis: value (count or share)
    Also renders a copy-friendly table directly under the chart.
    """
    st.subheader(title)

    if df_pivot is None or df_pivot.empty:
        st.info("No data to plot.")
        return

    long_df = _pivot_to_long(df_pivot, value_name=value_name)

    # Optionally convert shares to %
    plot_df = long_df.copy()
    if as_percent:
        plot_df[value_name] = plot_df[value_name] * 100.0

    if px is None:
        st.dataframe(df_pivot.reset_index(), use_container_width=True)
        st.caption("Install plotly to enable interactive charts.")
        return

    fig = px.bar(
        plot_df,
        x="item",
        y=value_name,
        color="group",
        barmode="group",
        title=title,
    )

    y_title = "Percent" if as_percent else value_name.capitalize()
    fig.update_layout(xaxis_title="Item", yaxis_title=y_title, legend_title="Group")

    if as_percent:
        fig.update_traces(hovertemplate="%{y:.1f}%<extra></extra>")

    st.plotly_chart(fig, use_container_width=True)

    # Copy-friendly table under the chart
    st.caption("Copy-friendly table")
    if as_percent:
        tbl = df_pivot.copy() * 100.0
        st.dataframe(tbl.reset_index().round(2), use_container_width=True)
    else:
        st.dataframe(df_pivot.reset_index(), use_container_width=True)


def _maybe_date_filter(df: pd.DataFrame, date_col: str = "run_date", key_prefix: str = "date") -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    d = df.copy()
    if d[date_col].isna().all():
        return d
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    min_d = d[date_col].min()
    max_d = d[date_col].max()
    if pd.isna(min_d) or pd.isna(max_d):
        return d
    with st.sidebar.expander("Date filter", expanded=False):
        start, end = st.date_input(
            "Run date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
            key=f"{key_prefix}_{date_col}",
        )
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return d[(d[date_col] >= start_ts) & (d[date_col] <= end_ts)]


def main():
    st.title("LLM Visualizer")
    st.caption("Upload a Basic or Advanced LLM tracking export, explore charts, and export an enhanced Excel file.")

    uploaded = st.file_uploader("Upload an Excel export (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.stop()

    file_bytes = uploaded.getvalue()
    sheet_names = _sheet_names(file_bytes)
    info = backend.detect_export_type(sheet_names)

    colA, colB = st.columns([2, 3])
    with colA:
        st.write("**Detected export type:**", info.export_type)
    with colB:
        st.write("**Sheets found:**", ", ".join(sheet_names))

    # Load only relevant sheets by default
    if info.export_type == "advanced":
        default_sheets = [s for s in ["Results", "Mentions", "Citations", "SearchQueries"] if s in sheet_names]
    else:
        default_sheets = [s for s in ["brand_mentions", "answers"] if s in sheet_names]
    with st.sidebar:
        st.header("Data selection")
        selected_sheets = st.multiselect(
            "Sheets to load",
            sheet_names,
            default=default_sheets or sheet_names[:1],
            key="sheets_select",
        )

    if not selected_sheets:
        st.stop()

    # Loading sheets is cached and usually fast, heavy calculations are gated behind a Run button.
    raw_sheets = _read_sheets(file_bytes, selected_sheets)

    # Decide mentions source and citations source
    if info.export_type == "advanced":
        mentions_sheet = "Mentions" if "Mentions" in raw_sheets else ("Results" if "Results" in raw_sheets else selected_sheets[0])
        citations_sheet = "Citations" if "Citations" in raw_sheets else None
    else:
        # basic: prefer brand_mentions for mentions, answers for sentiment context if needed
        mentions_sheet = "brand_mentions" if "brand_mentions" in raw_sheets else selected_sheets[0]
        citations_sheet = None

    df_mentions_src = raw_sheets.get(mentions_sheet)
    df_citations_src = raw_sheets.get(citations_sheet) if citations_sheet else None

    # Sidebar options and explicit Run button.
    with st.sidebar:
        st.header("Options")

        # Brand column selection (depends on loaded mentions sheet)
        brand_col_options = [c for c in df_mentions_src.columns if c.lower() in {"brands", "brand"}] + [
            c for c in df_mentions_src.columns if "brand" in c.lower()
        ]
        brand_col = st.selectbox(
            "Brand column",
            options=brand_col_options or list(df_mentions_src.columns),
            index=0,
            key="brand_col_select",
        )

        inferred_brand_col = None
        if "inferred_brands" in df_mentions_src.columns:
            use_inferred = st.checkbox("Include inferred_brands (advanced Results)", value=False, key="use_inferred")
            inferred_brand_col = "inferred_brands" if use_inferred else None

        group_candidates = [c for c in ["platform", "model", "country", "run_date"] if c in df_mentions_src.columns]
        group_cols = st.multiselect(
            "Group by",
            options=group_candidates,
            default=[c for c in ["platform"] if c in group_candidates],
            key="mentions_groupby",
        )

        top_n = st.slider("Top N items", 5, 50, int(st.session_state.get("top_n", 25)), 1, key="top_n_slider")
        include_other = st.checkbox("Include 'Other' bucket", value=bool(st.session_state.get("include_other", True)), key="include_other_chk")

        mapping_lines = st.text_area(
            "Manual brand mappings (one per line, alias=canonical)",
            value=st.session_state.get("mapping_lines", ""),
            help="Example: bosch=STIHL",
            height=120,
            key="mapping_lines_area",
        )

        # Placeholder so Brand merge suggestions appears directly under Options
        merge_box = st.container()


        st.divider()
        run_now = st.button("Run calculations", type="primary", key="run_now_btn")

    # Persist options
    st.session_state["top_n"] = top_n
    st.session_state["include_other"] = include_other
    st.session_state["mapping_lines"] = mapping_lines

    # Compute-heavy transforms are gated behind the Run button.
    if run_now or "mentions_long" not in st.session_state:
        mentions_long = backend.build_mentions_long(
            df_mentions_src,
            brand_col=brand_col,
            inferred_brand_col=inferred_brand_col,
            keep_cols=[c for c in df_mentions_src.columns if c != brand_col],
        )
        if mapping_lines.strip():
            mentions_long["brand"] = backend.apply_manual_mappings(mentions_long["brand"], mapping_lines)

        mentions_long = _maybe_date_filter(mentions_long, "run_date", key_prefix="mentions")
        # Auto-merge exact-case brand variants (no user approval needed)
        if hasattr(backend, "auto_merge_exact_case"):
            mentions_long["brand"] = backend.auto_merge_exact_case(mentions_long["brand"])

        st.session_state["mentions_long"] = mentions_long
        st.session_state["last_run_ready"] = True
    else:
        mentions_long = st.session_state["mentions_long"]


    # ================================
    # Brand merge suggestions (user approved)
    # ================================
    backend_mod = importlib.reload(backend)

    if "brand_merge_mapping" not in st.session_state:
        st.session_state["brand_merge_mapping"] = {}
    if "brand_merge_suggestions" not in st.session_state:
        st.session_state["brand_merge_suggestions"] = None

    if run_now:
        try:
            if not hasattr(backend_mod, "suggest_brand_merges"):
                raise AttributeError("suggest_brand_merges is missing from backend.py")
            brand_counts = mentions_long["brand"].value_counts()
            sugg = backend_mod.suggest_brand_merges(
                brands=brand_counts.index.tolist(),
                counts=brand_counts,
                min_containment_len=3,
                similarity_threshold=0.80,
                max_suggestions=500,
            )
            if sugg is not None and (hasattr(sugg, "empty") and (not sugg.empty)):
                sugg = sugg.copy()
                sugg.insert(0, "combine", False)
            st.session_state["brand_merge_suggestions"] = sugg
        except Exception as e:
            st.session_state["brand_merge_suggestions"] = None
            with merge_box.expander("Brand merge suggestions", expanded=False):
                st.error("Brand merge suggestions could not be generated.")
                st.caption(f"Reason: {type(e).__name__}: {e}")
                st.caption(f"Backend file loaded: {getattr(backend_mod, '__file__', 'unknown')}")
            pass

    with merge_box.expander("Brand merge suggestions", expanded=False):
        sugg_df = st.session_state.get("brand_merge_suggestions")
        if sugg_df is None:
            st.caption("Run calculations to generate suggestions.")
        elif getattr(sugg_df, "empty", True):
            st.caption("No merge suggestions found for the current selection.")
        else:
            st.write("Select pairs to combine, then click Apply selected merges.")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("Select all", key="select_all_merges_btn"):
                    tmp = sugg_df.copy()
                    tmp["combine"] = True
                    st.session_state["brand_merge_suggestions"] = tmp
                    st.rerun()
            with col_s2:
                if st.button("Clear selection", key="clear_all_merges_btn"):
                    tmp = sugg_df.copy()
                    tmp["combine"] = False
                    st.session_state["brand_merge_suggestions"] = tmp
                    st.rerun()

            display_df = sugg_df[["combine", "alias", "canonical"]].copy()
            edited = st.data_editor(
                display_df,
                use_container_width=True,
                hide_index=True,
                key="brand_merge_editor",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Apply selected merges", type="primary", key="apply_brand_merges_btn"):
                    merge_map = _brand_merge_mapping_from_editor(edited)
                    st.session_state["brand_merge_mapping"] = merge_map
                    mentions_long2 = mentions_long.copy()
                    if hasattr(backend_mod, "apply_merge_mapping"):
                        mentions_long2["brand"] = backend_mod.apply_merge_mapping(mentions_long2["brand"], merge_map)
                    st.session_state["mentions_long"] = mentions_long2
                    st.rerun()
            with col_b:
                if st.button("Clear merges", key="clear_brand_merges_btn"):
                    st.session_state["brand_merge_mapping"] = {}
                    st.rerun()


    if "last_run_ready" not in st.session_state:
        st.info("Choose your options in the sidebar, then click Run calculations.")
        st.stop()

    # Build citations long if present
    citations_long = None
    if df_citations_src is not None:
        dom_col_opts = [c for c in df_citations_src.columns if c.lower() in {"domain", "url"}] + [
            c for c in df_citations_src.columns if "domain" in c.lower()
        ]
        with st.sidebar:
            domain_col = st.selectbox(
                "Citations domain column",
                options=dom_col_opts or list(df_citations_src.columns),
                index=0,
                key="cit_domain_col",
            )

        if run_now or "citations_long" not in st.session_state:
            citations_long = backend.build_citations_long(df_citations_src, domain_col=domain_col)
            citations_long = _maybe_date_filter(citations_long, "run_date", key_prefix="citations")
            st.session_state["citations_long"] = citations_long
        else:
            citations_long = st.session_state.get("citations_long")

    # Tabs
    tab_overview, tab_mentions, tab_sentiment, tab_citations, tab_export = st.tabs(
        ["Overview", "Mentions", "Sentiment", "Citations", "Export"]
    )

    with tab_overview:
        st.subheader("Quick stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows in mentions source", len(df_mentions_src))
        c2.metric("Mentions (exploded)", len(mentions_long))
        c3.metric("Unique brands", mentions_long["brand"].nunique())
        if citations_long is not None:
            c4.metric("Citations", len(citations_long))
        else:
            c4.metric("Citations", 0)

        st.markdown("### Data preview")
        st.write("Mentions (long format)")
        st.dataframe(mentions_long.head(200), use_container_width=True)
        if citations_long is not None:
            st.write("Citations (long format)")
            st.dataframe(citations_long.head(200), use_container_width=True)

    with tab_mentions:
        st.header("Brand mentions")
        if not group_cols:
            st.info("Pick at least one 'Group by' field in the sidebar to build grouped charts.")
        counts, share = backend.aggregate_counts(
            mentions_long,
            item_col="brand",
            group_cols=group_cols or ["platform"] if "platform" in mentions_long.columns else [],
            top_n=top_n,
            include_other=include_other,
        )
        _plot_clustered_bars(counts, value_name="count", title="Mentions count (clustered)")
        _plot_clustered_bars(share, value_name="share", title="Mentions share (clustered)", as_percent=True)

        # Trend chart if run_date selected
        if "run_date" in group_cols and px is not None:
            st.markdown("### Trend over time")
            tmp = mentions_long.copy()
            tmp["run_date"] = pd.to_datetime(tmp["run_date"], errors="coerce").dt.date
            g = [c for c in group_cols if c != "run_date"]
            tmp2 = tmp.groupby(["run_date"] + g + ["brand"]).size().reset_index(name="count")
            fig = px.line(tmp2, x="run_date", y="count", color="brand", line_group="brand", title="Brand mentions over time")
            st.plotly_chart(fig, use_container_width=True)

    
        # Mentioned URLs by platform (normalized, ignore UTM parameters)
        url_cols_m = [c for c in mentions_long.columns if "url" in c.lower()]
        if url_cols_m:
            st.markdown("### Mentioned URLs by platform")
            url_col_m = url_cols_m[0]
            if "platform" in mentions_long.columns:
                mention_url_counts = backend.build_url_counts(mentions_long, url_col=url_col_m, group_col="platform")
            else:
                mention_url_counts = backend.build_url_counts(mentions_long, url_col=url_col_m, group_col=None)
                if not mention_url_counts.empty:
                    mention_url_counts.insert(0, "platform", "All")

            st.session_state["mention_url_counts_by_platform"] = mention_url_counts

            q_mu = st.text_input("Search URLs", value="", placeholder="Filter, for example bygghemma", key="mentions_url_search")
            min_mu = st.number_input("Minimum count", min_value=1, value=1, step=1, key="mentions_url_min")

            shown_mu = mention_url_counts[mention_url_counts["count"] >= int(min_mu)]
            if q_mu.strip():
                shown_mu = shown_mu[shown_mu["url"].str.contains(q_mu.strip(), case=False, na=False)]

            st.dataframe(shown_mu, use_container_width=True, hide_index=True)

            st.download_button(
                "Download mentioned URL counts (CSV)",
                data=mention_url_counts.to_csv(index=False).encode("utf-8"),
                file_name="mentioned_url_counts_by_platform.csv",
                mime="text/csv",
                use_container_width=True,
            )


with tab_sentiment:
        st.header("Sentiment")
        # Try to find sentiment from source sheet if not in mentions_long
        sentiment_df = df_mentions_src.copy()
        if "sentiment" not in sentiment_df.columns:
            # attempt to locate sentiment from other likely sheet
            if info.export_type == "basic" and "answers" in raw_sheets and "sentiment" in raw_sheets["answers"].columns:
                sentiment_df = raw_sheets["answers"]
            elif info.export_type == "advanced" and "Results" in raw_sheets and "sentiment" in raw_sheets["Results"].columns:
                sentiment_df = raw_sheets["Results"]

        if "sentiment" not in sentiment_df.columns:
            st.info("No sentiment column found in the loaded sheets.")
        else:
            cols = [c for c in ["platform", "model", "country", "run_date"] if c in sentiment_df.columns]
            group_s = st.multiselect("Group sentiment by", options=cols, default=[c for c in ["platform"] if c in cols], key="sentiment_group")
            sentiment_df2 = sentiment_df.copy()
            if "run_date" in sentiment_df2.columns:
                sentiment_df2["run_date"] = pd.to_datetime(sentiment_df2["run_date"], errors="coerce")
            sentiment_df2 = _maybe_date_filter(sentiment_df2, "run_date", key_prefix="sentiment")

            summary = backend.sentiment_summary(sentiment_df2, group_s or (["platform"] if "platform" in sentiment_df2.columns else []))
            st.dataframe(summary, use_container_width=True)

            if px is not None:
                sentiment_df2["sentiment"] = pd.to_numeric(sentiment_df2["sentiment"], errors="coerce")
                if group_s:
                    gcol = group_s[0]
                    fig = px.box(sentiment_df2.dropna(subset=["sentiment"]), x=gcol, y="sentiment", title=f"Sentiment distribution by {gcol}")
                    st.plotly_chart(fig, use_container_width=True)
                fig2 = px.histogram(sentiment_df2.dropna(subset=["sentiment"]), x="sentiment", nbins=20, title="Sentiment histogram")
                st.plotly_chart(fig2, use_container_width=True)

    with tab_citations:
        st.header("Citations")
        if citations_long is None:
            st.info("No citations sheet loaded (Advanced exports have a 'Citations' sheet).")
        else:
            group_cands = [
                c
                for c in ["platform", "model", "country", "parent_domain", "domain_classification"]
                if c in citations_long.columns
            ]
            group_cols_c = st.multiselect(
                "Group citations by",
                options=group_cands,
                default=[c for c in ["platform"] if c in group_cands],
                key="cit_group",
            )
            top_n_c = st.slider("Top N domains", 5, 50, 20, 1, key="topn_dom")

            counts_c, share_c = backend.aggregate_counts(
                citations_long,
                item_col="domain",
                group_cols=group_cols_c or (["platform"] if "platform" in citations_long.columns else []),
                top_n=top_n_c,
                include_other=True,
            )

            _plot_clustered_bars(counts_c, value_name="count", title="Citations count (clustered)")
            _plot_clustered_bars(share_c, value_name="share", title="Citations share (clustered)", as_percent=True)

            st.markdown("### Top domains overall")
            top_domains = citations_long["domain"].value_counts().head(top_n_c).reset_index()
            top_domains.columns = ["domain", "count"]
            st.dataframe(top_domains, use_container_width=True)

            # URLs overall (normalized, ignore UTM parameters)
            url_series = None
            url_cols = [c for c in citations_long.columns if "url" in c.lower()] if citations_long is not None else []
            if url_cols:
                url_series = citations_long[url_cols[0]]
            elif df_citations_src is not None:
                src_url_cols = [c for c in df_citations_src.columns if "url" in c.lower()]
                if src_url_cols:
                    url_series = df_citations_src[src_url_cols[0]]

            if url_series is not None:
                st.markdown("### All URLs overall")

                urls_norm = url_series.dropna().astype(str).map(backend.normalize_url_for_count)
                urls_norm = urls_norm[urls_norm.str.strip() != ""]

                url_counts_all = (
                    urls_norm.value_counts()
                    .rename_axis("url")
                    .reset_index(name="count")
                )
                st.session_state["citation_url_counts_all"] = url_counts_all

                q = st.text_input(
                    "Search URLs",
                    value="",
                    placeholder="Type to filter, for example bygghemma",
                    key="url_search",
                )
                min_count = st.number_input(
                    "Minimum count",
                    min_value=1,
                    value=1,
                    step=1,
                    key="url_min_count",
                )

                shown = url_counts_all[url_counts_all["count"] >= int(min_count)]
                if q.strip():
                    shown = shown[shown["url"].str.contains(q.strip(), case=False, na=False)]

                st.dataframe(shown, use_container_width=True)

                st.download_button(
                    "Download URL counts (CSV)",
                    data=url_counts_all.to_csv(index=False).encode("utf-8"),
                    file_name="url_counts_all.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


            if px is not None and "domain_classification" in citations_long.columns:
                cls = (
                    citations_long["domain_classification"]
                    .replace("", "Unknown")
                    .fillna("Unknown")
                    .value_counts()
                    .head(15)
                    .reset_index()
                )
                cls.columns = ["classification", "count"]
                fig = px.bar(cls, x="classification", y="count", title="Top domain classifications")
                st.plotly_chart(fig, use_container_width=True)


    with tab_export:
        st.header("Export enhanced Excel")
        st.write("The export includes your original sheets plus cleaned long tables and summary tables.")

        # Build summaries for export based on current selections
        exp_group_cols = group_cols or (["platform"] if "platform" in mentions_long.columns else [])
        mention_counts, mention_share = backend.aggregate_counts(
            mentions_long, item_col="brand", group_cols=exp_group_cols, top_n=top_n, include_other=include_other
        )

        citation_counts = citation_share = None
        if citations_long is not None:
            exp_group_cols_c = ["platform"] if "platform" in citations_long.columns else []
            citation_counts, citation_share = backend.aggregate_counts(
                citations_long, item_col="domain", group_cols=exp_group_cols_c, top_n=20, include_other=True
            )

                # Ensure URL count tables are available for export, even if the user did not open the Mentions or Citations tabs

        # Mentions URL counts by platform
        if st.session_state.get("mention_url_counts_by_platform") is None:
            url_cols_m_exp = [c for c in mentions_long.columns if "url" in str(c).lower()]
            if url_cols_m_exp:
                if "platform" in mentions_long.columns:
                    st.session_state["mention_url_counts_by_platform"] = backend.build_url_counts(
                        mentions_long, url_col=url_cols_m_exp[0], group_col="platform"
                    )
                else:
                    tmp_counts = backend.build_url_counts(mentions_long, url_col=url_cols_m_exp[0], group_col=None)
                    if not tmp_counts.empty:
                        tmp_counts.insert(0, "platform", "All")
                    st.session_state["mention_url_counts_by_platform"] = tmp_counts

        # Citations URL counts overall
        if citations_long is not None and st.session_state.get("citation_url_counts_all") is None:
            url_cols_c_exp = [c for c in citations_long.columns if "url" in str(c).lower()]
            if url_cols_c_exp:
                st.session_state["citation_url_counts_all"] = backend.build_url_counts(
                    citations_long, url_col=url_cols_c_exp[0], group_col=None
                )

        xlsx_bytes = backend.build_enhanced_excel(
            raw_sheets=raw_sheets,
            mentions_long=mentions_long,
            citations_long=citations_long,
            mention_counts=mention_counts,
            mention_share=mention_share,
            citation_counts=citation_counts,
            citation_share=citation_share,
                   mention_url_counts=st.session_state.get("mention_url_counts_by_platform"),
            citation_url_counts=st.session_state.get("citation_url_counts_all"),
        )
        st.download_button(
            "Download enhanced Excel",
            data=xlsx_bytes,
            file_name="llm_visualizer_enhanced.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("### What gets exported")
        st.write(
            [
                "Original loaded sheets (as-is)",
                "Mentions_Long: exploded and normalized mentions, one row per brand mention",
                "Citations_Long: cleaned citations with normalized domains (Advanced exports)",
                "Mentions_Counts and Mentions_Share: summary pivot tables",
                "Citations_Counts and Citations_Share: summary pivot tables (Advanced exports)",
            ]
        )


if __name__ == "__main__":
    main()