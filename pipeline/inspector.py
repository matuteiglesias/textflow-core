import statistics
import datetime
import textwrap
import pandas as pd

def summarize_nodes(docstore, index_store):
    n_docs = len(docstore)
    lengths, timestamps, titles = [], [], []

    for entry in docstore.values():
        txt = entry.get("text", "")
        lengths.append(len(txt.split()))
        meta = entry.get("metadata", {})
        ts = meta.get("timestamp")
        if ts: timestamps.append(int(ts))
        ttl = meta.get("title")
        if ttl: titles.append(ttl)

    stats = {
        "Documents": n_docs,
        "Min words": min(lengths) if lengths else 0,
        "Median words": int(statistics.median(lengths)) if lengths else 0,
        "Max words": max(lengths) if lengths else 0,
        "Titles w/ metadata": len(titles),
    }

    if timestamps:
        stats["Time span"] = f"{datetime.datetime.fromtimestamp(min(timestamps)/1000):%Y-%m-%d} → {datetime.datetime.fromtimestamp(max(timestamps)/1000):%Y-%m-%d}"

    roots = index_store.get("root_nodes", [])
    children_map = index_store.get("children", {})

    def depth(node_id):
        kids = children_map.get(node_id, [])
        return 1 if not kids else 1 + max(depth(k) for k in kids)

    stats["Roots"] = len(roots)
    stats["Max depth"] = max((depth(r) for r in roots), default=0)
    return stats

def preview_nodes(docstore, limit=10):
    rows = []
    for node_id, entry in list(docstore.items())[:limit]:
        txt = entry.get("text", "")
        meta = entry.get("metadata", {})
        rows.append({
            "node_id": node_id[:8],
            "words": len(txt.split()),
            "title": meta.get("title", "")[:40],
            "timestamp": meta.get("timestamp"),
            "preview": textwrap.shorten(txt.replace("\n", " "), width=100, placeholder=" …"),
        })
    return pd.DataFrame(rows)


def preview_nodes_filtered(docstore, keyword=None, after=None, limit=50):
    rows = []
    for node_id, entry in docstore.items():
        txt = entry.get("text", "")
        meta = entry.get("metadata", {})
        if keyword and keyword.lower() not in txt.lower():
            continue
        if after:
            ts = meta.get("timestamp", 0)
            if int(ts) < after.timestamp() * 1000:
                continue
        rows.append({
            "node_id": node_id[:8],
            "words": len(txt.split()),
            "title": meta.get("title", "")[:40],
            "timestamp": meta.get("timestamp"),
            "preview": textwrap.shorten(txt.replace("\n", " "), width=100, placeholder=" …"),
        })
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)



import pandas as pd

def extract_node_edges(index_store):
    edges = []
    children_map = index_store.get("children", {})
    for parent_id, children in children_map.items():
        for child_id in children:
            edges.append((parent_id, child_id))
    return pd.DataFrame(edges, columns=["parent", "child"])

import networkx as nx
import matplotlib.pyplot as plt

def plot_node_tree(edges_df, limit=100):
    G = nx.DiGraph()
    for _, row in edges_df.head(limit).iterrows():
        G.add_edge(row["parent"], row["child"])

    pos = nx.spring_layout(G, k=0.3)  # or use nx.nx_agraph.graphviz_layout(G, prog='dot') for hierarchy
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, arrows=True)
    plt.title("Node Tree (First N Edges)")
    plt.show()
