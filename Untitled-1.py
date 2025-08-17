# %%
# pip install llama-index-embeddings-jinaai

# %%
import pathlib, json, datetime as dt, textwrap, numpy as np
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import os

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list


import sqlite3, pickle, pathlib, numpy as np


# %%
##############################################################################
# 2. Markdown split   (one node per ### block)
##############################################################################
parser = MarkdownNodeParser(include_metadata=True)


# %%
##############################################################################
# 0.  Configuration – change once, reuse everywhere
##############################################################################
TEST_DIR   = pathlib.Path("test_data")         # where the JSONL files live
CACHE_DB   = pathlib.Path("embedding_cache.sqlite")
CHROMA_DIR = pathlib.Path("store/chroma")
COLL_NAME  = "gpt_logs_jina_v3"
EMBED_DIM  = 1024                              # Jina‑V3

##############################################################################
# 1.  Imports & helpers
##############################################################################
import os, json, datetime as dt, hashlib, sqlite3, pathlib, numpy as np
# from llama_index.core import Document, MarkdownNodeParser
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document

from llama_index.embeddings.jinaai import JinaEmbedding
import chromadb


##############################################################################
# 2.  Embedding model (one object, reused)
##############################################################################
os.environ["JINAAI_API_KEY"] = "jina_5b0aa4cf7715402492a7e71123831915WsqPh5_0qOiwI_C4YoIUrttL0Aat"
embedder = JinaEmbedding(model="jina-embeddings-v3", task="retrieval.passage")



# %%
# con.execute("""
#     create table if not exists processed_files
#     (fname text primary key)
# """)
# con.commit()

# embedder = JinaEmbedding(
#     api_key=jinaai_api_key,
#     model="jina-embeddings-v3",
#     # choose `retrieval.passage` to get passage embeddings
#     task="retrieval.passage",
# )


# ##############################################################################
# # 7.  Do the ingest for files matching 2025‑*.jsonl
# ##############################################################################
# files = sorted(TEST_DIR.glob("2025*.jsonl"))
# vecs, nodes = ingest_paths(files)     # takes ~0 s if everything is cached

# print(f"Ingested {len(nodes)} nodes from {len(files)} files")

# %%
xx

# %%
##############################################################################
# 4. Cosine linkage → dendrogram leaf order
##############################################################################
Z     = linkage(pdist(vecs, metric="cosine"), method="average")
order = leaves_list(Z)          # permutation that gives dendrogram left→right

vecs, nodes = ingest_paths(files)     # takes ~0 s if everything is cached

leaf_nodes = [n for n in nodes if "\n" in n.text.strip()]


for idx in order:
    n   = leaf_nodes[idx]
    hdr = n.metadata["header_path"]
    preview = textwrap.shorten(n.text, 100).replace("\n", " ")
    print(f"{hdr:40s} | {preview}")

# %%

vecs.shape


# %%

vecs, ids, metas = load_vectors_from_chroma()



# %%
vecs.shape

# %%

# # ──────────────────────────────────────────────────────────
# # 1.  If an old collection exists with wrong dimension, drop it
# # ──────────────────────────────────────────────────────────
# try:
#     old = client.get_collection(COLL_NAME)
#     logging.info("Deleting stale collection %s", COLL_NAME)
#     client.delete_collection(COLL_NAME)
# except chromadb.errors.NotFoundError:
#     pass  # nothing to delete


# %%
xx

# %%
from embedder import *

# %%
# (A) First try to ingest new material
files = sorted(TEST_DIR.glob("2025*.jsonl"))
new_vecs, new_nodes = ingest_paths(files)

# (B) Then load the *entire* corpus for analysis
vecs, nodes = load_vectors_and_nodes()     # always returns everything

# --- same clustering block as before ---
leaf_nodes = [n for n in nodes if "\n" in n.text.strip()]
Z     = linkage(pdist(vecs, metric="cosine"), method="average")
order = leaves_list(Z)

for idx in order:
    n   = leaf_nodes[idx]
    hdr = n.metadata["header_path"]
    preview = textwrap.shorten(n.text, 100).replace("\n", " ")
    print(f"{hdr:40s} | {preview}")


# %%
# ------------------------------------------------------------------
# 5. Concatenate & write                                              
# ------------------------------------------------------------------
# combined_md = "\n\n".join(doc.text for doc in docs)


combined_md = "\n\n".join(leaf_nodes[idx].text for idx in order)



out_path = pathlib.Path("combined_notes.md")
out_path.write_text(combined_md, encoding="utf-8")

print(f"Wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")


# %%
from scipy.cluster.hierarchy import fcluster
labels = fcluster(Z, t=0.25, criterion="distance")   # tweak t ∈ [0,1]

# group indices by label and print inside each cluster
from collections import defaultdict
bucket = defaultdict(list)
for idx, lab in enumerate(labels):
    bucket[lab].append(idx)

for lab, idxs in sorted(bucket.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"\n## Cluster {lab}  (n={len(idxs)})")
    for i in idxs:
        n = leaf_nodes[i]
        print(f"{n.metadata['header_path']:40s} | "
              f"{textwrap.shorten(n.text, 90)}")


# %%



