import sqlite3
import numpy as np

import json, glob, os, pathlib, logging

from .embedder import get_embedder

CACHE_DB = "embedding_cache.sqlite"
con = sqlite3.connect(CACHE_DB)
con.execute("""create table if not exists vecs
               (id text primary key, dim int, blob blob)""")
con.commit()

def cached_embed(text_id: str, text: str) -> np.ndarray:
    row = con.execute("select blob, dim from vecs where id = ?", (text_id,)).fetchone()
    if row:
        return np.frombuffer(row[0], dtype=np.float32).reshape(row[1])
    vec = np.array(get_embedder().get_text_embedding(text), dtype=np.float32)
    con.execute("insert into vecs values (?,?,?)", (text_id, vec.size, vec.tobytes()))
    con.commit()
    return vec

from llama_index.core import Document
from typing import Iterator, List
DOCS_JSON          = pathlib.Path("store/docs.json")  # raw dump for inspection

###############################################################################
# 6.  Dump raw docs for sanity checks
###############################################################################
def dump_docs_json(docs: List[Document]) -> None:
    DOCS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump([{"text": d.text, **d.metadata} for d in docs], f, ensure_ascii=False)
    logging.info("docs.json written (%s)", DOCS_JSON)
