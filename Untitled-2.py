# %%
from raptor.raptor import RetrievalAugmentation

# %%
# pipeline.py
import json, glob, os, pathlib, logging
# from typing import Iterator, List
from llama_index.core import SimpleDirectoryReader, TreeIndex
import chromadb
from llama_index.core import (
    Document,
    TreeIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
# from raptor import RetrievalAugmentation                          # optional



###############################################################################
# 0.  CONFIGURATION
###############################################################################
INPUT_DIR          = pathlib.Path("test_data")        # *.jsonl files
CHROMA_COLLECTION  = "gpt_logs"
DOCS_JSON          = pathlib.Path("store/docs.json")  # raw dump for inspection
BUILD_RAPTOR       = True
RAPTOR_PATH        = pathlib.Path("store/raptor_tree.json")

from pathlib import Path
RAPTOR_TREE = Path("store/raptor_tree.pkl")     # single file on disk

 

# %%
# print("Embed model:", EMBED_MODEL.__class__.__name__)
# import openai, os
# print("OpenAI key set:", bool(os.getenv("OPENAI_API_KEY")))


# %%
RAPTOR_TREE = Path("store/raptor_tree.pkl")     # single file on disk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from llama_index.core import StorageContext, VectorStoreIndex

RAPTOR_JSON = Path("store/raptor_leafs.jsonl")   # texts + sha1
RAPTOR_CFG  = Path("store/raptor_cfg.json")      # config dict

###############################################################################
# 7.  Orchestrator
###############################################################################
def main() -> None:
    docs = load_documents(INPUT_DIR)
    dump_docs_json(docs)
    build_tree_index(docs)
    add_chroma_vectors(docs)
    build_raptor(docs)
    
    logging.info("Pipeline finished 🎉")


if __name__ == "__main__":
    main()


# %%
RA.save("store/raptor_tree.pkl")

# %%
print(RA.answer_question("discuss linkedin profile of candidate matias iglesias"))


# %%
xx

# %%
RA = load_raptor()
print(RA.answer_question("What were my Streamlit sessions about?"))


# %%


# %%

# docs = load_documents()          # parses your JSONL
build_raptor(docs)               # writes store/raptor_tree.pkl


# %%
import pickle, pathlib
from raptor.raptor import RetrievalAugmentation

RAPTOR_PATH = pathlib.Path("store/raptor_tree.pkl")

with RAPTOR_PATH.open("rb") as f:
    RA: RetrievalAugmentation = pickle.load(f)

print(RA.answer_question("give me a summary of my PromptFlow debugging sessions"))


# %%
xx

# %%



