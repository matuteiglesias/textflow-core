import os
import numpy as np
import sqlite3
from typing import Callable

from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


# -------- Configuration --------
DEFAULT_MODEL = "jina"  # switch to "openai" if needed

JINA_MODEL = "jina-embeddings-v3"
OPENAI_MODEL = "text-embedding-3-small"

# Load API keys from environment or fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "openai_...")
JINAAI_API_KEY = os.getenv("JINAAI_API_KEY", "jina_...")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["JINAAI_API_KEY"] = JINAAI_API_KEY


# -------- Embedder Instances --------
JINA_EMBEDDER = JinaEmbedding(
    api_key=JINAAI_API_KEY,
    model=JINA_MODEL,
    task="retrieval.passage"
)

OPENAI_EMBEDDER = OpenAIEmbedding(model_name=OPENAI_MODEL)


# -------- Embedder Factory --------
def get_embedder(model: str = DEFAULT_MODEL):
    if model == "openai":
        return OPENAI_EMBEDDER
    elif model == "jina":
        return JINA_EMBEDDER
    else:
        raise ValueError(f"Unsupported embedder: {model}")


# -------- Cached Embedder --------
def get_cached_embed_fn(con: sqlite3.Connection, model: str = DEFAULT_MODEL) -> Callable[[str, str], np.ndarray]:
    embedder = get_embedder(model)
    con.execute(
        "CREATE TABLE IF NOT EXISTS vecs (id TEXT PRIMARY KEY, dim INT, blob BLOB)"
    )
    con.commit()

    def embed_fn(uid: str, text: str) -> np.ndarray:
        row = con.execute("SELECT blob, dim FROM vecs WHERE id=?", (uid,)).fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.float32).reshape(row[1])

        vec = np.asarray(embedder.get_text_embedding(text), dtype=np.float32)
        con.execute("INSERT INTO vecs VALUES (?, ?, ?)", (uid, vec.size, vec.tobytes()))
        con.commit()
        return vec

    return embed_fn
