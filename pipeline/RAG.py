from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, List

import logging, time

# llama-index bits
from llama_index.core import (
    Document, Settings, StorageContext, VectorStoreIndex,
)
from llama_index.core.retrievers import BaseRetriever



# -------------------------------
# Config objects (swap pieces here)
# -------------------------------

@dataclass
class StoreConfig:
    kind: Literal["memory","chroma"]
    chroma_path: Path = Path("store/chroma")
    collection: str = "gpt_logs"
    # behavior toggles
    load_only: bool = True          # default: never write, only attach
    refresh: bool = False           # wipe collection first
    auto_fork_on_mismatch: bool = False
    fail_if_empty: bool = True
    load_if_exists: bool = True     # NEW: default to reuse


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    multilingual: bool = False
    cache_dir: Optional[str] = None
    batch_size: int = 16
    device: Optional[str] = None  # e.g., "cpu" or "cuda"

@dataclass
class SplitterConfig:
    chunk_size: int = 1024
    overlap: int = 120

@dataclass
class RerankerConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    top_n: int = 8

@dataclass
class RetrievalConfig:
    top_k: int = 8
    similarity_cutoff: float = 0.32
    # token_cap intentionally not enforced at build-time;
    # apply during answer phase if doing context-only responses.

@dataclass
class BuildResult:
    qe: Any
    retriever: BaseRetriever
    num_nodes: int
    build_seconds: float
    effective_embed_model: str
    store_kind: str




# # -------------------------------
# # Factories
# # -------------------------------

# def _select_model_id(embed: EmbeddingConfig) -> str:
#     # simple policy: auto-swap to multilingual small if requested and the base is the EN model
#     if embed.multilingual and embed.model == "BAAI/bge-small-en-v1.5":
#         return "intfloat/multilingual-e5-small"
#     return embed.model



# from snippetflow.storage import upsert_node_chroma, get_chroma_collection

# def _make_index_from_nodes(
#     nodes: List[Any],
#     store: StoreConfig,
#     *,
#     effective_model: str,   # what your CLI/embed resolver picked (for metadata only)
#     split: SplitterConfig,  # used for mismatch warnings / metadata
# ) -> Tuple[VectorStoreIndex, int, int, bool]:
#     """
#     Returns: (index, count_before, count_after, loaded_only)

#     Design:
#       - MEMORY: build an in-RAM view from your Markdown nodes (no vector DB).
#       - CHROMA: attach-only; never re-embed. Uses from_vector_store().
#       - No ingest in here. If you *want* to ingest when empty, do it outside
#         via your existing ingest.py entrypoint, then call this again.
#     """
#     import logging
#     from pathlib import Path
#     from llama_index.core import VectorStoreIndex, StorageContext

#     # ----------------
#     # In-memory store
#     # ----------------
#     if store.kind == "memory":
#         # This *will* compute embeddings into RAM once, but doesn't touch Chroma.
#         idx = VectorStoreIndex(nodes=nodes, show_progress=True)
#         n = len(nodes)
#         return idx, n, n, False

#     # -------------
#     # Chroma store
#     # -------------
#     if store.kind != "chroma":
#         raise ValueError(f"Unknown store.kind={store.kind!r}")

#     try:
#         import chromadb
#         from llama_index.vector_stores.chroma import ChromaVectorStore
#     except Exception as e:
#         raise RuntimeError(
#             "Chroma requested but not installed. "
#             "pip install chromadb llama-index-vector-stores-chroma"
#         ) from e

#     # Connect
#     store.chroma_path.mkdir(parents=True, exist_ok=True)
#     client = chromadb.PersistentClient(path=str(store.chroma_path))

#     # Desired metadata for sanity
#     desired_meta = {
#         "embed_model": effective_model or "",
#         "chunk_size": str(split.chunk_size),
#         "overlap":    str(split.overlap),
#         # Optional: note LlamaIndex major used to index (purely informational)
#         "li_ver":     "0.10",
#     }

#     # Note: get_or_create_collection won't rewrite metadata for an existing coll.
#     coll = client.get_or_create_collection(name=store.collection, metadata=desired_meta)
#     meta = coll.metadata or {}

#     # Optional: wipe first
#     if getattr(store, "refresh", False):
#         coll.delete(where={})
#         logging.info("Chroma collection %s cleared.", store.collection)

#     # Warn (or fork) on metadata mismatch
#     mismatches = {
#         k: (meta.get(k), v) for k, v in desired_meta.items()
#         if meta.get(k) is not None and meta.get(k) != v
#     }
#     if mismatches:
#         if getattr(store, "auto_fork_on_mismatch", False):
#             suffix = f"{desired_meta['embed_model'].split('/')[-1]}__cs{split.chunk_size}_ov{split.overlap}"
#             fork = f"{store.collection}__{suffix}"
#             logging.warning("Chroma metadata differs %s — forking to %s", mismatches, fork)
#             coll = client.get_or_create_collection(name=fork, metadata=desired_meta)
#             meta = coll.metadata or {}
#         else:
#             logging.warning("Chroma metadata differs: %s (continuing with current collection)", mismatches)

#     count_before = coll.count()

#     # Load-only semantics (default): attach to existing vectors or fail if empty
#     load_only = getattr(store, "load_only", True)
#     fail_if_empty = getattr(store, "fail_if_empty", True)

#     if load_only:
#         if count_before == 0 and fail_if_empty:
#             raise RuntimeError(
#                 f"Chroma collection '{store.collection}' at {store.chroma_path} is empty. "
#                 f"Run your ingest first (ingest.py) to populate vectors/documents."
#             )
#         vstore = ChromaVectorStore(chroma_collection=coll)
#         index  = VectorStoreIndex.from_vector_store(vstore, show_progress=False)
#         return index, count_before, count_before, True  # loaded_only=True

#     # Non load-only (advanced): we STILL do not embed here.
#     # This function keeps retrieval pure. If you want to populate on-the-fly,
#     # do it OUTSIDE (call ingest_paths) and then re-call this function.
#     vstore = ChromaVectorStore(chroma_collection=coll)
#     index  = VectorStoreIndex.from_vector_store(vstore, show_progress=False)
#     count_after = coll.count()
#     return index, count_before, count_after, True


# -------------------------------
# Public builder
# -------------------------------

def build_retrieval_pipeline(
    docs: List[Document],
    *,
    store: StoreConfig,        # add: .load_only=True, .fail_if_empty=True, .sqlite_cache=Path(...)
    embed: EmbeddingConfig,    # add: .provider in {"hf","jina","openai",None}, .model (optional override)
    split: SplitterConfig,     # used only for memory mode & for mismatch warnings
    reranker: RerankerConfig,  # .enabled, .model, .top_n
    retrieval: RetrievalConfig # .top_k, .similarity_cutoff
) -> BuildResult:
    """
    Attach-only retrieval pipeline for RAG.
    - Chroma path: never re-embeds; attaches to existing vectors & docs.
    - Memory path: parses via MarkdownNodeParser (not SentenceSplitter).
    """
    import time, logging
    t0 = time.time()

    from llama_index.core import Settings, VectorStoreIndex, StorageContext
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import BaseRetriever

    # ----------------------------
    # Helper: resolve query embedder to MATCH stored vectors
    # ----------------------------
    def _resolve_query_embedder(provider: str | None, model_id: str | None):
        """
        Returns (embed_model_impl, effective_provider, effective_model_id).
        Uses your embedder stub for Jina/OpenAI when requested; HF otherwise.
        """
        eff_provider = (provider or getattr(embed, "provider", None) or "hf").lower()
        eff_model    = model_id or getattr(embed, "model", None)

        try:
            if eff_provider == "jina":
                from llama_index.embeddings.jinaai import JinaEmbedding
                if not eff_model:
                    eff_model = "jina-embeddings-v3"
                em = JinaEmbedding(model=eff_model, task="retrieval.passage")
                return em, "jina", eff_model

            if eff_provider == "openai":
                from llama_index.embeddings.openai import OpenAIEmbedding
                if not eff_model:
                    eff_model = "text-embedding-3-small"
                em = OpenAIEmbedding(model_name=eff_model)
                return em, "openai", eff_model

            # default: HF
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            if not eff_model:
                eff_model = "BAAI/bge-small-en-v1.5"
            kwargs = dict(model_name=eff_model)
            if getattr(embed, "cache_dir", None):
                kwargs["cache_folder"] = embed.cache_dir
            if getattr(embed, "batch_size", None):
                kwargs["embed_batch_size"] = embed.batch_size
            if getattr(embed, "device", None):
                kwargs["device"] = embed.device
            em = HuggingFaceEmbedding(**kwargs)
            return em, "hf", eff_model

        except Exception as e:
            raise RuntimeError(f"Failed to initialize query embedder (provider={eff_provider}, model={eff_model}): {e}") from e

    # ----------------------------
    # CHROMA (attach-only) branch
    # ----------------------------
    if store.kind == "chroma":
        try:
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except Exception as e:
            raise RuntimeError("Chroma requested but not installed. `pip install chromadb llama-index-vector-stores-chroma`") from e

        store.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(store.chroma_path))
        coll   = client.get_or_create_collection(store.collection)
        meta   = coll.metadata or {}

        # diagnostics / metadata mismatch warnings
        desired = {
            "chunk_size": str(split.chunk_size),
            "overlap":    str(split.overlap),
        }
        diffs = {k: (meta.get(k), v) for k, v in desired.items() if meta.get(k) and meta.get(k) != v}
        if diffs:
            logging.warning("Chroma collection metadata differs from CLI splitter: %s", diffs)

        n = coll.count()
        if n == 0 and getattr(store, "fail_if_empty", True):
            raise RuntimeError(
                f"Chroma collection '{store.collection}' at {store.chroma_path} is empty.\n"
                "→ Run your ingest first (ingest.py) to populate vectors/documents."
            )

        # pick query embedder to match the space used at ingest
        coll_provider = meta.get("embed_provider")
        coll_model    = meta.get("embed_model")
        if embed.model and coll_model and embed.model != coll_model:
            logging.warning("Embed model override (%s) differs from collection (%s) — retrieval quality may degrade.",
                            embed.model, coll_model)

        em, eff_provider, eff_model = _resolve_query_embedder(coll_provider, coll_model or embed.model)
        Settings.embed_model = em

        # attach index view (no re-embedding)
        vstore = ChromaVectorStore(chroma_collection=coll)
        index  = VectorStoreIndex.from_vector_store(vstore, show_progress=False)

        # retriever + post
        retriever: BaseRetriever = index.as_retriever(similarity_top_k=retrieval.top_k)
        post = [SimilarityPostprocessor(similarity_cutoff=retrieval.similarity_cutoff)]

        # reranker (optional)
        if reranker.enabled:
            try:
                from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
                post.insert(1, FlagEmbeddingReranker(model=reranker.model, top_n=reranker.top_n))
                logging.info("Reranker enabled (%s, top_n=%d).", reranker.model, reranker.top_n)
            except Exception as e:
                logging.warning("Failed to enable reranker (%s); continuing without.", e.__class__.__name__)
        else:
            logging.info("Reranker disabled.")

        # query engine (version-safe)
        try:
            qe = index.as_query_engine(retriever=retriever, node_postprocessors=post, response_mode="compact")
        except TypeError as e:
            if "multiple values for argument 'retriever'" in str(e):
                logging.debug("Falling back to RetrieverQueryEngine due to retriever arg collision.")
                qe = RetrieverQueryEngine(retriever=retriever, node_postprocessors=post)
            else:
                raise

        build_seconds = time.time() - t0
        return BuildResult(
            qe=qe,
            retriever=retriever,
            num_nodes=n,
            build_seconds=build_seconds,
            effective_embed_model=eff_model,
            store_kind="chroma",
        )

    # ----------------------------
    # MEMORY branch (no Chroma, no re-embed in vector DB)
    # Uses your MarkdownNodeParser to match ingest semantics.
    # ----------------------------
    else:
        from llama_index.core.node_parser import MarkdownNodeParser

        # query embedder: use CLI embed.provider/model if set
        em, eff_provider, eff_model = _resolve_query_embedder(getattr(embed, "provider", None), getattr(embed, "model", None))
        Settings.embed_model = em

        parser = MarkdownNodeParser(include_metadata=True)
        nodes  = parser.get_nodes_from_documents(docs)
        logging.info("Parsed %d markdown nodes (memory mode).", len(nodes))

        # Build a transient index in-RAM (this *will* compute embeddings once for RAM,
        # but doesn't touch Chroma). If you don't want even this, use a FAISS/NP backend.
        index = VectorStoreIndex(nodes=nodes, show_progress=True)

        retriever: BaseRetriever = index.as_retriever(similarity_top_k=retrieval.top_k)
        post = [SimilarityPostprocessor(similarity_cutoff=retrieval.similarity_cutoff)]
        if reranker.enabled:
            try:
                from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
                post.insert(1, FlagEmbeddingReranker(model=reranker.model, top_n=reranker.top_n))
                logging.info("Reranker enabled (%s, top_n=%d).", reranker.model, reranker.top_n)
            except Exception as e:
                logging.warning("Failed to enable reranker (%s); continuing without.", e.__class__.__name__)
        else:
            logging.info("Reranker disabled.")

        try:
            qe = index.as_query_engine(retriever=retriever, node_postprocessors=post, response_mode="compact")
        except TypeError as e:
            if "multiple values for argument 'retriever'" in str(e):
                logging.debug("Falling back to RetrieverQueryEngine due to retriever arg collision.")
                qe = RetrieverQueryEngine(retriever=retriever, node_postprocessors=post)
            else:
                raise

        build_seconds = time.time() - t0
        return BuildResult(
            qe=qe,
            retriever=retriever,
            num_nodes=len(nodes),
            build_seconds=build_seconds,
            effective_embed_model=eff_model,
            store_kind="memory",
        )
 



from dataclasses import dataclass, asdict
import json, time

@dataclass
class RunReport:
    date: str
    input_file: str
    output_file: str
    docs_loaded: int
    nodes_embedded: int
    top_k: int
    similarity_cutoff: float
    token_cap: int
    llm: str
    embed_model: str
    cache_dir: str | None
    reranker: bool
    started_at: float
    finished_at: float
    build_seconds: float
    total_seconds: float
    warnings: list[str]





# store_cfg = StoreConfig(
#     kind="chroma" if args.use_chroma else "memory",
#     chroma_path=Path(args.chroma_path),
#     collection=args.chroma_collection,
# )
# embed_cfg = EmbeddingConfig(
#     model=args.embed,
#     multilingual=args.multilingual,
#     cache_dir=args.cache_dir,
#     batch_size=args.embed_batch_size or 16,
#     device=getattr(args, "embed_device", None),
# )
# split_cfg = SplitterConfig(chunk_size=1024, overlap=120)
# rerank_cfg = RerankerConfig(enabled=not args.no_reranker, top_n=args.top_k)
# retrieval_cfg = RetrievalConfig(top_k=args.top_k, similarity_cutoff=args.similarity_cutoff)

# result = build_retrieval_pipeline(
#     docs,
#     store=store_cfg,
#     embed=embed_cfg,
#     split=split_cfg,
#     reranker=rerank_cfg,
#     retrieval=retrieval_cfg,
# )

# # feed into your run report
# num_nodes = result.num_nodes
# build_seconds = result.build_seconds
# effective_model = result.effective_embed_model
# store_kind = result.store_kind
# qe = result.qe
# retriever = result.retriever




# # … capture timings …
# t0 = time.time()
# # after building QE:
# t_build_done = time.time()
# # after writing MD:
# t_end = time.time()

# warnings = []
# # if you detect that faulthandler fired, append a note, e.g. by setting a flag when it triggers

# report = RunReport(
#     date=in_path.stem,
#     input_file=str(in_path),
#     output_file=str(args.out),
#     docs_loaded=len(docs),
#     nodes_embedded=len(index.docstore.docs) if 'index' in locals() else None,  # or track during chunking
#     top_k=args.top_k,
#     similarity_cutoff=0.32,
#     token_cap=args.token_cap,
#     llm=args.llm,
#     embed_model=model_id,
#     cache_dir=args.cache_dir,
#     reranker=not args.no_reranker,
#     started_at=t0,
#     finished_at=t_end,
#     build_seconds=(t_build_done - t0),
#     total_seconds=(t_end - t0),
#     warnings=warnings,
# )

# # write JSON + a tiny .run.md
# report_path_json = Path(args.out).with_suffix(".run.json")
# report_path_md   = Path(args.out).with_suffix(".run.md")
# report_path_json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
# report_path_md.write_text(
#     f"""# RAG run report — {report.date}

# - Input: `{report.input_file}`
# - Output: `{report.output_file}` ({Path(args.out).stat().st_size} bytes)
# - Docs loaded: **{report.docs_loaded}**
# - Nodes embedded: **{report.nodes_embedded}**
# - Embedding model: `{report.embed_model}` (cache: `{report.cache_dir}`)
# - Retriever: top_k={report.top_k}, cutoff={report.similarity_cutoff}
# - Token cap: {report.token_cap}
# - Reranker: {"on" if report.reranker else "off"}
# - LLM: {report.llm}

# **Timings**
# - Build time: {report.build_seconds:.2f}s
# - Total time: {report.total_seconds:.2f}s

# {"**Warnings**\n- " + "\n- ".join(report.warnings) if report.warnings else ""}
# """,
#     encoding="utf-8",
# )
# print(f"Report: {report_path_json} & {report_path_md}")





## UTILS

# at module top
try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    _HAS_RERANKER = True
except Exception:
    _HAS_RERANKER = False





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




import pathlib, json, argparse
from typing import Dict


def load_jsonl_as_documents(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # text = obj.get("aiResponse") or obj.get("response") or obj.get("text") or ""
            text = obj.get("content") or ""
            text = (text or "").strip()
            if not text:
                continue
            meta = {
                "exercise_id": obj.get("exerciseId"),
                "user_id": obj.get("userId"),
                "timestamp": obj.get("timestamp"),
                "source": path.name,
            }
            docs.append(Document(text=text, metadata=meta))
    return docs


# from snippetflow.ingest import load_documents, load_jsonl_as_documents

FAQS_EN = [
    "Snippets that belong to developer journal.",
    "Snippets that belong to teaching material"
]
FAQS_ES = [
    "Segmentos que pertenecen al developer journal.",
    "Segmentos que pertenecen a materiales de estudio"
]




def answer_with_llm(qe, questions: List[str]) -> List[Dict[str, Any]]:
    out = []
    guard = ("Answer tersely; bullets where natural. Cite sources inline as [^1],[^2] "
             "and include a Sources section mapping footnotes to titles/ids.")
    for q in questions:
        resp = qe.query(f"{q}\n\n{guard}")
        cites = []
        for i, s in enumerate(resp.source_nodes, start=1):
            label = (s.node.metadata.get("exercise_id") or
                     s.node.metadata.get("timestamp") or
                     s.node.metadata.get("source") or
                     f"node-{i}")
            cites.append((i, label))
        out.append({"question": q, "answer": str(resp), "citations": cites})
    return out

def answer_context_only(retriever, questions: List[str], per_q: int = 5) -> List[Dict[str, Any]]:
    out = []
    for q in questions:
        nodes = retriever.retrieve(q)[:per_q]
        # simple stitched context
        parts = []
        cites = []
        for i, n in enumerate(nodes, start=1):
            label = (n.node.metadata.get("exercise_id") or
                     n.node.metadata.get("timestamp") or
                     n.node.metadata.get("source") or f"node-{i}")
            cites.append((i, label))
            parts.append(f"**Passage {i}** — {label}\n\n{n.node.get_content(metadata_mode='none')}\n")
        stitched = ("_LLM disabled: showing top passages._\n\n" + "\n".join(parts)) if parts else "_No passages found._"
        out.append({"question": q, "answer": stitched, "citations": cites})
    return out

def render_markdown(day_label: str, results: List[Dict[str, Any]], theme="redux", layout="fixed") -> str:
    front = f"---\nconfig:\n  theme: {theme}\n  layout: {layout}\n---\n"
    title = f"# Daily FAQ — {day_label}\n\n"
    body = []
    foot = ["\n---\n", "## Sources\n"]
    seen = set()
    for idx, r in enumerate(results, start=1):
        body.append(f"## Q{idx}. {r['question']}\n\n{r['answer']}\n")
        for j, label in r["citations"]:
            key = f"[^{j}]"
            if key in seen: 
                continue
            seen.add(key)
            foot.append(f"{key}: {label}\n")
    return front + title + "".join(body) + "".join(foot)



def main():
    # ---------------------------
    # CLI
    # ---------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a JSONL file OR a directory containing day files")
    ap.add_argument("--date", help="YYYY-MM-DD (used when --input is a directory)")
    ap.add_argument("--out", required=True, help="Output Markdown file")
    ap.add_argument("--lang", choices=["es","en"], default="es")
    ap.add_argument("--multilingual", action="store_true")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--token_cap", type=int, default=1800)
    ap.add_argument("--llm", choices=["openai","none"], default="none")
    ap.add_argument("--openai-model", default="gpt-4o-mini")

    # observability / runtime
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (repeat for DEBUG)")
    ap.add_argument("--dry-run", action="store_true", help="Build but do not write output")

    # storage
    ap.add_argument("--use-chroma", action="store_true", help="Persist vectors with Chroma")
    ap.add_argument("--chroma-path", default="store/chroma")
    ap.add_argument("--chroma-collection", default="gpt_logs")

    # embeddings
    ap.add_argument("--embed", default="BAAI/bge-small-en-v1.5", help="HF embedding model id")
    ap.add_argument("--embed-batch-size", type=int, default=16)
    ap.add_argument("--embed-device", choices=["cpu", "cuda"], help="Force device (cpu/cuda)")
    ap.add_argument("--cache-dir", default=None, help="HF cache dir (e.g., /tmp/hf_cache)")

    # retrieval behavior
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--similarity-cutoff", type=float, default=0.32)
    ap.add_argument("--refresh", action="store_true",
                    help="Drop & rebuild the Chroma collection.")

    args = ap.parse_args()

    # ---------------------------
    # Logging
    # ---------------------------
    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[{asctime}] {levelname} {name}: {message}",
        style="{",
    )
    print("RAG: entering main()", flush=True)
    logging.info("Args: %s", vars(args))
    print("RAG: logging configured", flush=True)

    # ---------------------------
    # Resolve input
    # ---------------------------
    in_path = Path(args.input)
    if in_path.is_dir():
        if args.date:
            in_path = in_path / f"{args.date}.jsonl"
        else:
            files = sorted(in_path.glob("*.jsonl"))
            if not files:
                raise SystemExit(f"No JSONL files in {in_path}")
            in_path = files[-1]
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    logging.info("Input resolved to: %s", in_path)

    # ---------------------------
    # Load docs
    # ---------------------------
    t0 = time.time()
    docs = load_jsonl_as_documents(in_path)  # assumes your existing helper
    logging.info("Loaded %d docs from %s in %.2fs", len(docs), in_path, time.time() - t0)
    if not docs:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            f"---\nconfig:\n  theme: redux\n  layout: fixed\n---\n# Daily FAQ — {in_path.stem}\n\n_No notes found._\n",
            encoding="utf-8",
        )
        print("RAG: no docs, wrote empty report", flush=True)

        # minimal run report on empty input
        try:
            rr = RunReport(
                date=in_path.stem,
                input_file=str(in_path),
                output_file=str(out_p),
                docs_loaded=0,
                nodes_embedded=0,
                top_k=args.top_k,
                similarity_cutoff=args.similarity_cutoff,
                token_cap=args.token_cap,
                llm=args.llm,
                embed_model=args.embed,
                cache_dir=args.cache_dir,
                reranker=not args.no_reranker,
                started_at=t0,
                finished_at=time.time(),
                build_seconds=0.0,
                total_seconds=time.time() - t0,
                warnings=[],
            )
            report_path_json = out_p.with_suffix(".run.json")
            report_path_md = out_p.with_suffix(".run.md")
            report_path_json.write_text(json.dumps(asdict(rr), indent=2), encoding="utf-8")
            report_path_md.write_text(
                f"# RAG run report — {rr.date}\n\n"
                f"- Input: `{rr.input_file}`\n"
                f"- Output: `{rr.output_file}` ({out_p.stat().st_size} bytes)\n"
                f"- Docs loaded: **0**\n"
                f"- Nodes embedded: **0**\n"
                f"- Embedding model: `{rr.embed_model}` (cache: `{rr.cache_dir}`)\n"
                f"- Retriever: top_k={rr.top_k}, cutoff={rr.similarity_cutoff}\n"
                f"- Token cap: {rr.token_cap}\n"
                f"- Reranker: {'on' if rr.reranker else 'off'}\n"
                f"- LLM: {rr.llm}\n\n"
                f"**Timings**\n- Build time: {rr.build_seconds:.2f}s\n- Total time: {rr.total_seconds:.2f}s\n",
                encoding="utf-8",
            )
            print(f"Report: {report_path_json} & {report_path_md}", flush=True)
        except Exception as e:
            logging.warning("Could not write run report for empty input: %s", e)
        return

    # ---------------------------
    # Build retrieval pipeline
    # ---------------------------
    print("RAG: building query engine…", flush=True)

    store_cfg = StoreConfig(
        kind="chroma" if args.use_chroma else "memory",
        chroma_path=Path(args.chroma_path),
        collection=args.chroma_collection,
        refresh=getattr(args, "refresh", False),
        load_if_exists=True,
    )
    embed_cfg = EmbeddingConfig(
        model=args.embed,
        multilingual=args.multilingual,
        cache_dir=args.cache_dir,
        batch_size=args.embed_batch_size or 16,
        device=args.embed_device,
    )
    split_cfg = SplitterConfig(chunk_size=1024, overlap=120)
    rerank_cfg = RerankerConfig(enabled=not args.no_reranker, top_n=args.top_k)
    retrieval_cfg = RetrievalConfig(top_k=args.top_k, similarity_cutoff=args.similarity_cutoff)

    try:
        build_res = build_retrieval_pipeline(
            docs,
            store=store_cfg,
            embed=embed_cfg,
            split=split_cfg,
            reranker=rerank_cfg,
            retrieval=retrieval_cfg,
        )
    except Exception as e:
        logging.exception("Unhandled error during pipeline build: %s", e)
        print("RAG: exception — see traceback above", flush=True)
        raise

    qe = build_res.qe
    retriever = build_res.retriever
    num_nodes = build_res.num_nodes
    build_seconds = build_res.build_seconds
    effective_model = build_res.effective_embed_model
    store_kind = build_res.store_kind

    logging.info("Query engine ready (elapsed %.2fs total so far)", time.time() - t0)
    print("RAG: query engine ready", flush=True)

    # ---------------------------
    # Answer questions
    # ---------------------------
    print("RAG: answering questions…", flush=True)
    questions = FAQS_ES if args.lang == "es" else FAQS_EN

    if args.llm == "openai":
        # Settings.llm already configured elsewhere when args.llm == "openai"
        results = answer_with_llm(qe, questions)
        mode_str = "openai"
    else:
        # context-only; enforce token cap inside this function if you implemented that
        results = answer_context_only(retriever, questions, per_q=min(5, args.top_k))
        mode_str = "none"

    # ---------------------------
    # Write output
    # ---------------------------
    md = render_markdown(day_label=in_path.stem, results=results)
    out_p = Path(args.out)
    if not args.dry_run:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md, encoding="utf-8")
    t_end = time.time()
    total_seconds = t_end - t0
    logging.info("Wrote %s (%d bytes)", out_p, out_p.stat().st_size if out_p.exists() else 0)
    print("RAG: done", flush=True)

    # ---------------------------
    # Run report (JSON + MD)
    # ---------------------------
    warnings_list: List[str] = []
    rr = RunReport(
        date=in_path.stem,
        input_file=str(in_path),
        output_file=str(out_p),
        docs_loaded=len(docs),
        nodes_embedded=num_nodes,
        top_k=args.top_k,
        similarity_cutoff=args.similarity_cutoff,
        token_cap=args.token_cap,
        llm=args.llm,
        embed_model=effective_model,
        cache_dir=args.cache_dir,
        reranker=(not args.no_reranker),
        started_at=t0,
        finished_at=t_end,
        build_seconds=build_seconds,
        total_seconds=total_seconds,
        warnings=warnings_list,
    )
    report_path_json = out_p.with_suffix(".run.json")
    report_path_md = out_p.with_suffix(".run.md")
    try:
        report_path_json.write_text(json.dumps(asdict(rr), indent=2), encoding="utf-8")
        report_path_md.write_text(
            f"# RAG run report — {rr.date}\n\n"
            f"- Input: `{rr.input_file}`\n"
            f"- Output: `{rr.output_file}` ({out_p.stat().st_size} bytes)\n"
            f"- Docs loaded: **{rr.docs_loaded}**\n"
            f"- Nodes embedded: **{rr.nodes_embedded}**\n"
            f"- Store: `{store_kind}`\n"
            f"- Embedding model: `{rr.embed_model}` (cache: `{rr.cache_dir}`)\n"
            f"- Retriever: top_k={rr.top_k}, cutoff={rr.similarity_cutoff}\n"
            f"- Token cap (answer path): {rr.token_cap}\n"
            f"- Reranker: {'on' if rr.reranker else 'off'}\n"
            f"- LLM: {rr.llm}\n\n"
            f"**Timings**\n"
            f"- Build time: {rr.build_seconds:.2f}s\n"
            f"- Total time: {rr.total_seconds:.2f}s\n\n"
            + ("**Warnings**\n- " + "\n- ".join(rr.warnings) + "\n" if rr.warnings else "")
            ,
            encoding="utf-8",
        )
        print(f"Report: {report_path_json} & {report_path_md}", flush=True)
    except Exception as e:
        logging.warning("Could not write run report: %s", e)

    # one-line digest for humans / CI
    print(
        f"Run: {in_path.name} -> {args.out} | "
        f"docs={len(docs)} nodes={num_nodes} store={store_kind} "
        f"model={effective_model} mode={mode_str} "
        f"build={build_seconds:.2f}s total={total_seconds:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    print("RAG: starting main module import…", flush=True)
    main()
