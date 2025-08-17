
import numpy as np
import json, glob, os, pathlib, logging
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.core.schema import TextNode
from typing import List, Optional

from llama_index.core import SimpleDirectoryReader, TreeIndex


import chromadb

CHROMA_DIR = "store/chroma"
COLL_NAME = "gpt_logs_jina_v3"
client = chromadb.PersistentClient(path=CHROMA_DIR)
coll = client.get_or_create_collection(COLL_NAME)

TREE_DIR           = pathlib.Path("store/tree")       # docstore + index_store
CHROMA_DIR         = pathlib.Path("store/chroma")     # vector store parquet/sqlite

from llama_index.embeddings.openai import OpenAIEmbedding


# Set your Jina key in env OR externally
small_model = os.getenv("SMALL_EMBEDDING_MODEL", "jina_...")  # keep secure
os.environ["SMALL_EMBEDDING_MODEL"] = small_model

EMBED_MODEL = OpenAIEmbedding(model_name=small_model)

def _sanitize_metadata(d: dict) -> dict:
    clean = {}
    for k, v in d.items():
        if v is None:
            continue                     # drop None entirely
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)            # last-resort stringify
    return clean

##############################################################################
# 5.  Upsert logic – idempotent: cache hit ‑> no API call, ID clash ‑> skip
##############################################################################

def upsert_node_chroma(node: TextNode, vec: np.ndarray, *, coll, uid: Optional[str] = None):
    """Idempotent add; tolerate repeated inserts."""
    

    uid = uid or getattr(node, "node_id", None)
    if not uid:
        raise ValueError("upsert_node_chroma: missing uid")

    # Build minimal metadata – only scalars
    meta_raw = {
        "file":        node.metadata.get("file", ""),                        # always a str
        "header_path": node.metadata.get("header_path", "/"),                # str
        "title":       node.metadata.get("title") or "",                     # avoid None
        "ts_ms":       int(node.metadata.get("timestamp"))                   # int
                       if node.metadata.get("timestamp") else None,
    }
    meta = _sanitize_metadata(meta_raw)    
    try:
        coll.add(
            ids=[uid],
            embeddings=[vec.tolist()],
            documents=[node.get_content(metadata_mode="EMBED")],
            metadatas=[meta],
        )
    except chromadb.errors.IDAlreadyExistsError:
        pass  # idempotent skip




###############################################################################
# 3.  Build + persist TreeIndex
###############################################################################
def build_tree_index(docs: List[Document]) -> TreeIndex:
    TREE_DIR.mkdir(parents=True, exist_ok=True)
    index = TreeIndex.from_documents(
        docs,
        chunk_size=1024,
        embed_model=EMBED_MODEL,
        show_progress=True,
    )
    index.storage_context.persist(TREE_DIR)
    logging.info("TreeIndex persisted to %s", TREE_DIR)
    return index


###############################################################################
# 4.  Attach Chroma vectors
###############################################################################
def add_chroma_vectors(docs: List[Document]) -> None:
    from llama_index.core import StorageContext, VectorStoreIndex

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll   = client.get_or_create_collection("gpt_logs")
    vstore = ChromaVectorStore(chroma_collection=coll)

    store  = StorageContext.from_defaults(
                persist_dir=TREE_DIR,
                vector_store=vstore)

    VectorStoreIndex.from_documents(
        docs, storage_context=store,
        embed_model=EMBED_MODEL, show_progress=True)

    # client.persist()                            # optional explicit flush
    logging.info("Chroma vectors stored → %s", CHROMA_DIR)


def get_chroma_collection(name: str, path: str = "store/chroma"):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name)

# ##############################################################################
# # 4.  Chroma collection – created once, reused forever
# ##############################################################################
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# def get_chroma_collection(name):
#     try:
#         coll = client.get_collection(name)
#         # dim  = coll._embedding_dimension          # works for Chroma ≤0.4
#         # if dim != expected_dim:
#         #     print(f"⚠️  Resetting collection {name}: {dim}‑D → {expected_dim}‑D")
#         #     client.delete_collection(name)
#         #     raise chromadb.errors.NotFoundError
#         return coll
#     except chromadb.errors.NotFoundError:
#         return client.create_collection(name)

# coll = get_or_reset_collection(COLL_NAME, EMBED_DIM)

