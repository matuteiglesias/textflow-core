# pip install llama-index-embeddings-jinaai

# snippetflow/ingest.py
from __future__ import annotations

import json, hashlib, datetime as dt, sqlite3, logging
from pathlib import Path
from typing import List, Tuple, Callable
import numpy as np

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode


# %%

# %%
# import pathlib, json, datetime as dt, textwrap, numpy as np
# from llama_index.core import Document
# from llama_index.embeddings.jinaai import JinaEmbedding
# from llama_index.embeddings.openai import OpenAIEmbedding
# import os
# import json, glob, os, pathlib, logging

# from scipy.spatial.distance import pdist
# from scipy.cluster.hierarchy import linkage, leaves_list

##############################################################################
# 1. JSONL  →  Markdown  →  Document
##############################################################################

import json, datetime as dt, pathlib
from llama_index.core import Document

def jsonl_to_md(path: pathlib.Path) -> Document:
    date = path.stem
    lines = [f"# {date}"]
    for row in path.open(encoding="utf-8"):
        j = json.loads(row)
        if j.get("role") != "assistant":
            continue
        ts = j.get("timestamp")
        title = j.get("title", "untitled")
        if ts:
            iso = dt.datetime.fromtimestamp(ts/1000).isoformat()[:19]
            lines.append(f"\n## {title}\n### {iso}\n{j['content']}")
        else:
            lines.append(f"\n## {title}\n{j['content']}")
    return Document(text="\n".join(lines), metadata={"file": path.name})



###############################################################################
# 1.  Streaming JSONL iterator (no newline required)
###############################################################################
_DECODER = json.JSONDecoder()


# import json, glob
# from llama_index.core.schema import Document
from typing import Iterator, List


def iter_json_objects(fp) -> Iterator[dict]:
    buf = ""
    for chunk in iter(lambda: fp.read(4096), ""):
        buf += chunk
        while buf:
            try:
                obj, idx = _DECODER.raw_decode(buf)
                yield obj
                buf = buf[idx:].lstrip()
            except json.JSONDecodeError:
                break





###############################################################################
# 2.  Parse -> List[Document]
###############################################################################
def load_documents(input_dir: pathlib.Path) -> List[Document]:
    docs: List[Document] = []
    for fname in input_dir.glob("*.jsonl"):
        with open(fname, encoding="utf-8") as f:
            for j in iter_json_objects(f):
                if j.get("role") != "assistant":
                    continue
                docs.append(
                    Document(
                        text=j.get("content", ""),
                        metadata={
                            k: j.get(k)
                            for k in ("id", "conversation_id", "timestamp", "title")
                            if k in j
                        },
                    )
                )
    logging.info("Loaded %d assistant chunks", len(docs))
    return docs



from .ingest import jsonl_to_md
# from .storage import upsert_node_chroma, node_id, get_chroma_collection
# from .cache import cached_embed
import numpy as np

# snippetflow/ingest.py (utilities)
import hashlib, re
from llama_index.core.node_parser import MarkdownNodeParser

def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)  # collapse whitespace
    return s

def stable_uid(node: TextNode) -> str:
    """Hash text + header_path + file + timestamp to get a stable, collision-resistant ID."""
    h = hashlib.blake2s(digest_size=16)
    h.update(node.text.encode("utf-8"))
    h.update(node.metadata.get("header_path", "").encode("utf-8"))
    h.update(node.metadata.get("file", "").encode("utf-8"))
    ts = str(node.metadata.get("timestamp", ""))
    h.update(ts.encode("utf-8"))
    return h.hexdigest()


# snippetflow/ingest.py (more utilities)
import sqlite3, time, json

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS files (
      fname       TEXT PRIMARY KEY,
      mtime       INTEGER,
      sha1        TEXT,
      n_nodes     INTEGER,
      ingested_at INTEGER,
      status      TEXT
    );

    -- Mirror of accepted nodes so inspector/loader can work without LlamaIndex internals
    CREATE TABLE IF NOT EXISTS nodes (
      id          TEXT PRIMARY KEY,
      fname       TEXT,
      header_path TEXT,
      title       TEXT,
      ts_ms       INTEGER,
      text        TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(fname);
    CREATE INDEX IF NOT EXISTS idx_nodes_ts   ON nodes(ts_ms);

    CREATE TABLE IF NOT EXISTS vecs (
      id   TEXT PRIMARY KEY,
      dim  INT,
      blob BLOB
    );

    CREATE TABLE IF NOT EXISTS edges (
      parent TEXT,
      child  TEXT,
      ord    INT,
      PRIMARY KEY(parent, child)
    );

    CREATE TABLE IF NOT EXISTS processed_files (
      fname TEXT PRIMARY KEY
    );
    """)
    conn.commit()

def write_node(conn: sqlite3.Connection, *, node_id: str, fname: str,
               header_path: str, title: str | None, ts_ms: int | None, text: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO nodes(id,fname,header_path,title,ts_ms,text) VALUES (?,?,?,?,?,?)",
        (node_id, fname, header_path, title, ts_ms, text),
    )

def mark_file_done(conn: sqlite3.Connection, fname: str, n_nodes: int) -> None:
    conn.execute("INSERT OR REPLACE INTO files(fname, n_nodes, ingested_at, status) VALUES (?,?,?,?)",
                 (fname, n_nodes, int(time.time()*1000), "ok"))
    conn.execute("INSERT OR IGNORE INTO processed_files(fname) VALUES (?)", (fname,))
    conn.commit()





def ingest_paths(
    paths: List[Path],
    *,
    parser: MarkdownNodeParser,
    embed_fn: Callable[[str, str], np.ndarray],       # (uid, text) -> vector
    upsert_fn: Callable[..., None],                   # (node, vec, uid=...) -> None
    db_conn: sqlite3.Connection,
    embed_dim: int = 1024,                            # Jina v3 default
    show_logs: bool = True,
    skip_single_line: bool = True,
    max_chars: int = 20_000,                          # coarse safety guard
    commit_every: int = 200,                          # batch sqlite commits
) -> Tuple[np.ndarray, List[TextNode]]:
    """Ingest JSONL files → Markdown → Nodes, embed & upsert to Chroma, cache vectors in SQLite.

    Returns
    -------
    (vecs, nodes): (np.ndarray [N, D], List[TextNode])
      If no nodes were added, vecs is shape (0, embed_dim).
    """
    ensure_schema(db_conn)

    all_vecs: List[np.ndarray] = []
    all_nodes: List[TextNode] = []
    pending = 0

    for path in paths:
        # idempotent at file level
        already = db_conn.execute(
            "SELECT 1 FROM processed_files WHERE fname = ?", (path.name,)
        ).fetchone()
        if already:
            if show_logs:
                print(f"Skipping already-ingested file: {path.name}")
            continue

        doc = jsonl_to_md(path)

        try:
            nodes = parser.get_nodes_from_documents([doc])
        except Exception as e:
            print(f"Failed parsing {path.name}: {e}")
            continue

        if show_logs:
            print(f"File: {path.name} | {len(nodes)} nodes parsed.")

        for node in nodes:
            # attach filename for consistent UID creation and metadata
            node.metadata.setdefault("file", path.name)

            text = node.get_content(metadata_mode="EMBED").strip()
            if skip_single_line and "\n" not in text:
                continue
            if max_chars and len(text) > max_chars:
                if show_logs:
                    print(f"[skip] too long ({len(text)} chars): {path.name}")
                continue

            uid = stable_uid(node)

            try:
                vec = embed_fn(uid, text)           # cached on subsequent runs
                upsert_fn(node, vec, uid=uid)       # idempotent upsert to Chroma
                all_nodes.append(node)
                all_vecs.append(vec)

                # mirror node for inspector/loader
                db_conn.execute(
                    "INSERT OR REPLACE INTO nodes (id, fname, header_path, title, ts_ms, text) VALUES (?,?,?,?,?,?)",
                    (
                        uid,
                        path.name,
                        node.metadata.get("header_path", "/"),
                        node.metadata.get("title", ""),
                        int(node.metadata.get("timestamp")) if node.metadata.get("timestamp") else None,
                        node.text,
                    ),
                )
                pending += 1
                if pending >= commit_every:
                    db_conn.commit()
                    pending = 0

            except RuntimeError as e:
                # Typical provider error text check; keep generic to avoid coupling
                if "cannot exceed" in str(e).lower():
                    if show_logs:
                        print(f"[skip] provider length limit: {uid}")
                    continue
                else:
                    raise
            except Exception as e:
                print(f"Error embedding/upserting node from {path.name}: {e}")
                continue

        # mark file done only after its nodes were processed
        db_conn.execute("INSERT OR REPLACE INTO processed_files VALUES (?)", (path.name,))
        db_conn.commit()  # flush anything pending for this file

    vecs = np.empty((0, embed_dim), dtype=np.float32) if not all_vecs else np.vstack(all_vecs)
    return vecs, all_nodes


