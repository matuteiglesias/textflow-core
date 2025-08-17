import numpy as np
from snippetflow.storage import coll  # assumes already initialized Chroma collection



import sqlite3
import pickle

def load_kv_table(conn, table, key_col="id", value_col="blob"):
    cursor = conn.cursor()
    query = f"SELECT {key_col}, {value_col} FROM {table}"
    rows = cursor.execute(query).fetchall()
    return {k: pickle.loads(v) for k, v in rows}

def load_docstore_and_index(
    path,
    docstore_table=None,
    index_table="vecs",
    docstore_key_col=None,
    docstore_value_col=None,
    index_key_col="id",
    index_value_col="blob",
):
    conn = sqlite3.connect(path)
    docstore = {}
    if docstore_table:
        try:
            docstore = load_kv_table(conn, docstore_table, docstore_key_col, docstore_value_col)
        except Exception as e:
            print(f"Could not load docstore: {e}")
    index_store = load_kv_table(conn, index_table, index_key_col, index_value_col)
    return docstore, index_store



# snippetflow/loader.py
import sqlite3

def load_docstore_and_index_from_sqlite(path: str):
    conn = sqlite3.connect(path)
    cur  = conn.cursor()

    # docstore = { id: {"text":..., "metadata": {...}} }
    docstore = {}
    for id_, text, header_path, fname, ts_ms in cur.execute(
        "SELECT id, text, header_path, fname, ts_ms FROM nodes"
    ):
        md = {"header_path": header_path, "file": fname}
        if ts_ms is not None:
            md["timestamp"] = ts_ms
        docstore[id_] = {"text": text, "metadata": md}

    # very simple index_store: flat tree (you can enhance later with edges)
    index_store = {
        "root_nodes": [],
        "children": {id_: [] for id_ in docstore.keys()}
    }

    return docstore, index_store



def load_vectors_and_nodes():
    data = coll.get(include=["embeddings", "metadatas", "documents"])
    vecs = np.array(data["embeddings"], dtype=np.float32)
    return vecs, data["ids"], data["metadatas"]

# upsert_node_chroma

# def ingest_paths(paths):
#     nodes, vecs = [], []
#     parser = MarkdownNodeParser(include_metadata=True)

#     for p in paths:
#         doc = jsonl_to_md(p)
#         for n in parser.get_nodes_from_documents([doc]):
#             if "\n" not in n.text.strip():
#                 continue
#             upsert_node(n)
#             nodes.append(n)
#             vecs.append(cached_embed(node_id(n), n.get_content(metadata_mode="EMBED")))
#     return np.vstack(vecs), nodes





# ---- loaders: SQLite nodes + Chroma embeddings ----
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import chromadb

# If you already have snippetflow.storage.get_chroma_collection, use that.
# Otherwise, a tiny local helper:
def _get_chroma_collection(name: str, path: Path | str = "store/chroma"):
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(name)

def load_nodes_df_from_sqlite(sqlite_path: str = "embeds.sqlite") -> pd.DataFrame:
    conn = sqlite3.connect(sqlite_path)
    # matches the schema you’re writing in ingest_paths()
    q = """
        SELECT id, fname, header_path, title, ts_ms, text
        FROM nodes
        ORDER BY rowid
    """
    df = pd.read_sql_query(q, conn)
    conn.close()
    if df.empty:
        return df
    df["words"] = df["text"].fillna("").str.split().map(len)
    return df



# def load_vectors_from_chroma():
#     data = coll.get(include=["embeddings", "metadatas"])
#     vecs = np.array(data["embeddings"], dtype=np.float32)
#     # re‑create node metadata if you need it
#     return vecs, data["ids"], data["metadatas"]




def load_vectors_from_chroma(collection_name: str, chroma_dir: str = "store/chroma"):
    coll = _get_chroma_collection(collection_name, chroma_dir)

    # Ask Chroma for everything; older versions may default-limit otherwise
    data = coll.get(include=["embeddings", "metadatas", "documents"], limit=None)

    ids = data.get("ids")
    if ids is None:
        ids = []

    embs = data.get("embeddings", None)

    if embs is None:
        vecs = np.empty((0, 0), dtype=np.float32)
    elif isinstance(embs, np.ndarray):
        # Already an array
        vecs = embs.astype(np.float32, copy=False)
        if vecs.ndim == 1:  # single vector edge case
            vecs = vecs.reshape(1, -1)
    else:
        # List-of-lists → array
        if len(embs) == 0:
            vecs = np.empty((0, 0), dtype=np.float32)
        else:
            vecs = np.asarray(embs, dtype=np.float32)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)

    return vecs, ids, data

def align_vecs_with_nodes(vecs: np.ndarray, ids: list[str], nodes_df: pd.DataFrame):
    if nodes_df.empty or vecs.size == 0 or not ids:
        return vecs, nodes_df

    id_to_row = {rid: i for i, rid in enumerate(ids)}
    # keep only nodes that exist in Chroma
    nodes_df = nodes_df[nodes_df["id"].isin(id_to_row)].copy()
    # order nodes to match vec rows
    order = nodes_df["id"].map(id_to_row).to_numpy()
    nodes_df = nodes_df.assign(_order=order).sort_values("_order").drop(columns="_order")
    vecs = vecs[order]
    # Optional sanity logs:
    missing_in_chroma = set(nodes_df["id"]) - set(ids)
    if missing_in_chroma:
        print(f"⚠️ {len(missing_in_chroma)} nodes missing vectors in Chroma (will ignore).")
    return vecs, nodes_df


import pandas as pd
import numpy as np

def nodes_df_from_chroma(collection_name: str, chroma_dir: str = "store/chroma"):
    coll = _get_chroma_collection(collection_name, chroma_dir)
    data = coll.get(include=["metadatas", "documents", "embeddings"], limit=None)

    ids   = data.get("ids") or []
    metas = data.get("metadatas") or []
    docs  = data.get("documents") or []
    embs  = data.get("embeddings")

    rows = []
    for i, nid in enumerate(ids):
        md = metas[i] if i < len(metas) and metas else {}
        rows.append({
            "id": nid,
            "fname": md.get("fname") or md.get("file") or "",
            "header_path": md.get("header_path") or md.get("date") or "/",
            "title": md.get("title", ""),
            "ts_ms": md.get("timestamp"),
            "text": docs[i] if i < len(docs) and docs else "",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["words"] = df["text"].fillna("").str.split().map(len)

    # vectors
    if embs is None:
        vecs = np.empty((0, 0), dtype=np.float32)
    else:
        vecs = np.asarray(embs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

    return vecs, df
