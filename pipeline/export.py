# pipeline/export.py
from __future__ import annotations
import pathlib
from typing import Iterable, Mapping, Optional, Sequence, Union
import pandas as pd
import numpy as np


# ---------------------------
# Helpers
# ---------------------------
def _normalize_header_path(val) -> str:
    """Return a sensible section title string from header_path in any shape."""
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) else "untitled"
    if pd.isna(val):
        return "untitled"
    return str(val)


def _ensure_dataframe(df_or_list) -> pd.DataFrame:
    """Allow passing a list of dicts or a DataFrame; normalize to DataFrame."""
    if isinstance(df_or_list, pd.DataFrame):
        return df_or_list
    return pd.DataFrame(df_or_list)


# ---------------------------
# Public API
# ---------------------------
def export_markdown_from_df(
    nodes_df: Union[pd.DataFrame, Iterable[dict]],
    output_path: Union[str, pathlib.Path],
    *,
    labels: Optional[Sequence[int]] = None,
    cluster_col: str = "cluster",
    include_metadata: bool = True,
    title_map: Optional[Mapping[int, str]] = None,
    order: Optional[Sequence[int]] = None,
    sort_within: Optional[str] = None,       # e.g., "words", "ts_ms"
    ascending: bool = False,                 # descending by default
    section_format: str = "### {header}\n{body}",  # formatting of each note
) -> pathlib.Path:
    """
    Export a Markdown document from a nodes DataFrame.

    Parameters
    ----------
    nodes_df
        DataFrame with at least: ["id", "text", "header_path", "fname", "ts_ms", "words"].
    output_path
        Destination .md path.
    labels
        Optional array-like cluster labels aligned to *current* nodes_df order.
        If provided, a cluster section is emitted.
    cluster_col
        Column name to place labels into when `labels` is provided.
    include_metadata
        If True, include a per-note heading with header_path.
    title_map
        Optional dict {cluster_id: human_title} for nicer section names.
    order
        Optional index order (e.g., dendrogram `leaf_order`). Applied before grouping.
    sort_within
        Optional column to sort items *inside* each cluster/flat export.
    ascending
        Sort direction for `sort_within`.
    section_format
        Template used for each note; receives header and body.

    Returns
    -------
    The pathlib.Path where the file was written.
    """
    df = _ensure_dataframe(nodes_df).copy()

    # Apply external order (e.g. dendrogram leaf order)
    if order is not None:
        # order is an array of row positions (not ids); use iloc
        df = df.iloc[list(order)].reset_index(drop=True)

    # Attach labels as a column, if provided
    if labels is not None:
        if len(labels) != len(df):
            raise ValueError("`labels` length must match `nodes_df` length.")
        df[cluster_col] = np.asarray(labels)

    # Optional secondary ordering
    if sort_within and sort_within in df.columns and labels is None:
        df = df.sort_values(sort_within, ascending=ascending)

    lines = []

    if labels is None and cluster_col not in df.columns:
        # Flat export
        for row in df.itertuples(index=False):
            body = getattr(row, "text", "") or ""
            if include_metadata:
                header = _normalize_header_path(getattr(row, "header_path", "untitled"))
                lines.append(section_format.format(header=header, body=body))
            else:
                lines.append(body)

    else:
        # Clustered export
        # If order is set, groups will still preserve input order within each cluster
        grouped = df.groupby(cluster_col, sort=False)

        for cid, g in sorted(grouped, key=lambda kv: len(kv[1]), reverse=True):
            title = title_map.get(cid) if title_map else None
            heading = f"# Cluster {cid} ({len(g)})"
            if title:
                heading += f" — {title}"
            lines.append(heading)

            # Sort within cluster if requested
            if sort_within and sort_within in g.columns:
                g = g.sort_values(sort_within, ascending=ascending)

            for row in g.itertuples(index=False):
                body = getattr(row, "text", "") or ""
                if include_metadata:
                    header = _normalize_header_path(getattr(row, "header_path", "untitled"))
                    lines.append(section_format.format(header=header, body=body))
                else:
                    lines.append(body)

    combined = "\n\n".join(lines)
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(combined, encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size/1024:.1f} KB)")
    return path


def export_markdown_flat(
    nodes_df: Union[pd.DataFrame, Iterable[dict]],
    output_path: Union[str, pathlib.Path],
    *,
    include_metadata: bool = True,
    order: Optional[Sequence[int]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    section_format: str = "### {header}\n{body}",
) -> pathlib.Path:
    """Convenience wrapper for simple, non-clustered exports."""
    return export_markdown_from_df(
        nodes_df,
        output_path,
        labels=None,
        include_metadata=include_metadata,
        order=order,
        sort_within=sort_by,
        ascending=ascending,
        section_format=section_format,
    )



# --- Optional: Placeholder for Labeling ---
def extract_keywords_for_clusters(cluster_dict, method="top-tfidf"):
    """
    Placeholder. Can implement RAKE, TF-IDF, LLM summaries.
    """
    labels = {}
    for cid, nodes in cluster_dict.items():
        text = "\n".join(n.text for n in nodes)
        labels[cid] = f"[Cluster {cid}] {text[:50]}..."  # stub
    return labels


def export_markdown_clusters(cluster_dict, output_path, include_metadata=True, titles=None):
    lines = []
    for cid, ns in sorted(cluster_dict.items(), key=lambda x: len(x[1]), reverse=True):
        title = titles.get(cid) if titles else None
        header_line = f"# Cluster {cid} ({len(ns)} nodes)"
        if title:
            header_line += f" — {title}"
        lines.append(header_line)
        for n in ns:
            if include_metadata:
                raw_header = n.metadata.get("header_path", "untitled")
                if isinstance(raw_header, (list, tuple)):
                    raw_header = raw_header[0]
                lines.append(f"### {raw_header}\n{n.text}")
            else:
                lines.append(n.text)
    combined = "\n\n".join(lines)
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(combined, encoding="utf-8")
    print(f"Wrote {output_path} ({path.stat().st_size / 1024:.1f} KB)")
