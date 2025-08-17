


###############################################################################
# 5.  Build / load RAPTOR (incremental, non‑interactive)
###############################################################################
from llama_index.core import Document
import pathlib
RAPTOR_TREE       = pathlib.Path("store/raptor_tree.json")

from pathlib import Path
from raptor.raptor import RetrievalAugmentation
import logging

def build_raptor(docs: list[Document], path: Path = RAPTOR_TREE) -> None:
    """
    Rebuild the RAPTOR tree from *scratch* each run.
    • No CoreBPE problems (we let RA.save handle pickling).
    • Idempotent: running twice with the same texts produces the same file.
    """

    # ❶ Create a fresh RA (ignores any previous pickle)
    ra = RetrievalAugmentation()             # default config

    # ❷ Add every assistant chunk as one document
    #    (RA internally chunks at ~100 tokens, so this is safe)
    for d in docs:
        ra.add_documents(d.text)

    # ❸ Persist – this writes a clean pickle without CoreBPE headaches
    path.parent.mkdir(parents=True, exist_ok=True)
    ra.save(path)                            # → 'store/raptor_tree.pkl'
    logging.info("RAPTOR tree saved → %s", path)


def load_raptor(path: Path = RAPTOR_TREE) -> RetrievalAugmentation:
    """
    Reload the previously saved tree.
    Raises FileNotFoundError if build_raptor hasn't been run yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found – run build_raptor() first."
        )
    return RetrievalAugmentation(tree=str(path))


import numpy as np
from .storage import coll




from types import SimpleNamespace          # light wrapper

def load_vectors_and_nodes(coll):
    data = coll.get(include=["embeddings", "documents", "metadatas"])
    vecs = np.array(data["embeddings"], dtype=np.float32)

    # Build a minimal “node” object that has .text and .metadata["header_path"]
    nodes = [
        SimpleNamespace(
            text=doc,
            metadata={"header_path": meta["header_path"]}
        )
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]
    return vecs, nodes

