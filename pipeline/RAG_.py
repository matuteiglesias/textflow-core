# /home/matias/repos/textflow/pipeline/RAG.py
# -*- coding: utf-8 -*-
"""
RAG FAQs over a single day JSONL file of notes/responses.

JSONL schema (flexible): looks for any of these keys per line: 
  aiResponse | response | text
Optional metadata used if present: exerciseId, userId, timestamp

Examples:
  python RAG.py --input /home/matias/repos/textflow/ingest/2025-08-05.jsonl \
                --out   /home/matias/repos/textflow/exports/2025-08-05.md \
                --lang es --llm openai --openai-model gpt-4o-mini

  # No LLM: context-only answers (top passages per question)
  python RAG.py --input ... --out ... --llm none
"""
from __future__ import annotations
import argparse, json, textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---- LlamaIndex core
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

# # ---- Embeddings (local/HF)
# pip install llama-index-embeddings-huggingface
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---- Optional LLM (OpenAI); only used if --llm openai
try:
    from llama_index.llms.openai import OpenAI as OpenAILLM
except Exception:
    OpenAILLM = None  # allow running without openai deps

# ---- Token cap postprocessor
import tiktoken
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore



# -------- TokenCapPostprocessor (Pydantic-safe) --------
from typing import List, Optional, Any
from pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, MetadataMode

class TokenCapPostprocessor(BaseNodePostprocessor):
    """Keep retrieved nodes (in rank order) until a token budget is reached.

    - Counts tokens with tiktoken if available (default cl100k_base).
    - Falls back to a 4-chars-per-token heuristic.
    - Always keeps at least the first node to avoid empty context.
    """

    max_tokens: int = Field(default=2000, description="Maximum tokens across kept nodes")
    tokenizer_name: Optional[str] = Field(default=None, description="tiktoken encoding name")
    include_metadata: bool = Field(default=False, description="Count metadata too")
    _enc: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        try:
            import tiktoken  # type: ignore
            name = self.tokenizer_name or "cl100k_base"
            self._enc = tiktoken.get_encoding(name)
        except Exception:
            self._enc = None  # use heuristic

    def _count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        # heuristic: ~4 chars/token
        return max(1, (len(text) + 3) // 4)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        out: List[NodeWithScore] = []
        budget = int(self.max_tokens)

        for nws in nodes:
            node = getattr(nws, "node", None) or nws
            text = node.get_content(
                metadata_mode=MetadataMode.ALL if self.include_metadata else MetadataMode.NONE
            )
            cost = self._count(text)

            if not out:
                out.append(nws)   # keep first no matter what
                budget -= cost
                if budget <= 0:
                    break
                continue

            if cost <= budget:
                out.append(nws)
                budget -= cost
            else:
                break





FAQS_EN = [
    "Give a 5-bullet executive summary of what I worked on that day.",
    "What were the main technical problems or puzzles?",
    "Which algorithmic techniques appear (e.g., DP, greedy, MST, proofs)?",
    "What went wrong or remained unclear? Provide concrete examples.",
    "What did I validate with numeric micro-checks or unit tests?",
    "List decisions made and their rationale (one line each).",
    "Extract 5 reusable patterns or templates from the notes.",
    "What risks or TODOs remain? Prioritize them.",
    "What should be written up as a blog snippet (title + 3 bullets)?",
    "What are the next 3 actions for tomorrow with acceptance criteria?"
]
FAQS_ES = [
    "Resumen ejecutivo del día en 5 viñetas.",
    "¿Cuáles fueron los problemas técnicos principales?",
    "¿Qué técnicas algorítmicas aparecen (DP, greedy, MST, pruebas)?",
    "¿Qué salió mal o quedó poco claro? Ejemplos concretos.",
    "¿Qué validé con micro-checks numéricos o tests?",
    "Decisiones tomadas y justificación (una línea c/u).",
    "Extrae 5 patrones o plantillas reutilizables.",
    "Riesgos o pendientes; priorízalos.",
    "Propuesta de mini-post (título + 3 bullets).",
    "Próximas 3 acciones con criterios de aceptación."
]

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
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        logging.warning("No .jsonl files in %s", input_dir)
    for fname in files:
        cnt_total = cnt_kept = 0
        with open(fname, encoding="utf-8") as f:
            for j in iter_json_objects(f):
                cnt_total += 1
                if j.get("role") != "assistant":
                    continue
                docs.append(
                    Document(
                        text=j.get("content", "") or "",
                        metadata={k: j.get(k) for k in ("id","conversation_id","timestamp","title") if k in j},
                    )
                )
                cnt_kept += 1
        logging.info("Parsed %s | %d rows, kept %d assistant chunks", fname.name, cnt_total, cnt_kept)
    logging.info("Loaded %d assistant chunks total", len(docs))
    return docs




from huggingface_hub import snapshot_download
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def make_embedder(
    multilingual: bool,
    prefer: str = "instructor-large",
    cache_dir: str | None = None,
    force_local_repo: str | None = None,  # path to a pre-downloaded model
):
    # 1) if user gave a local model path, use it
    if force_local_repo:
        name = force_local_repo
        logging.info("Embedding model: %s", name)
        return HuggingFaceEmbedding(model_name=force_local_repo, cache_folder=cache_dir)

    # 2) multilingual path
    if multilingual:
        try:
            name = "intfloat/multilingual-e5-large"
            logging.info("Embedding model: %s", name)
            return HuggingFaceEmbedding(model_name=name,
                                        cache_folder=cache_dir)
        except Exception:
            name = "intfloat/multilingual-e5-base" 
            logging.info("Embedding model: %s", name)
            return HuggingFaceEmbedding(model_name=name,
                                        cache_folder=cache_dir)

    # 3) try INSTRUCTOR first
    if prefer.startswith("instructor"):
        try:
            name = "hkunlp/instructor-large"
            logging.info("Embedding model: %s", name)
            return HuggingFaceEmbedding(
                model_name=name,
                text_instruction="Represent this note by its algorithmic technique and decisions.",
                query_instruction="Represent this question about daily work and techniques for retrieval.",
                cache_folder=cache_dir,
            )
        except Exception as e:
            print("[warn] INSTRUCTOR failed, falling back to e5-large-v2. Error:", e)

    # 4) e5 fallback (great clustering, smaller/robust)
    try:
        name = "intfloat/e5-large-v2"
        logging.info("Embedding model: %s", name)
        return HuggingFaceEmbedding(model_name=name, cache_folder=cache_dir)
    except Exception:
        # 5) last resort: small BGE (fast, easy to pull)
        name = "BAAI/bge-small-en-v1.5"
        logging.info("Embedding model: %s", name)
        return HuggingFaceEmbedding(model_name=name, cache_folder=cache_dir)


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

import time
import sys, os, time, traceback, logging, signal, faulthandler
from pathlib import Path

# Always print a sentinel immediately (before logging config or heavy imports)
print("RAG: starting main module import…", flush=True)

# Unbuffer stdout/stderr just in case
if not sys.stdout.seekable():
    os.environ["PYTHONUNBUFFERED"] = "1"

# Enable faulthandler and set periodic stack dumps (every 30s)
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # override any previous basicConfig
    )
    # Tame noisy libs unless you asked for DEBUG
    if level > logging.DEBUG:
        for n in ("httpx","urllib3","transformers","huggingface_hub","sentence_transformers"):
            logging.getLogger(n).setLevel(logging.WARNING)



# NEW
from dataclasses import dataclass, asdict
import json, time, csv
from typing import List, Tuple


# make sure these exist already, otherwise add:
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import BaseRetriever


from llama_index.core.query_engine import RetrieverQueryEngine



from pathlib import Path
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker



from typing import Any, Tuple, List
from llama_index.core.retrievers import BaseRetriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import time, logging

def build_query_engine(
    docs: List[Document],
    *,
    multilingual: bool,
    top_k: int,
    token_cap: int,              # kept for signature parity (we apply cap later during answering)
    embed_model: str,
    cache_dir: str | None,
    use_reranker: bool,
    embed_batch_size: int | None,
    similarity_cutoff: float = 0.32,
) -> Tuple[Any, BaseRetriever, int, float, str]:
    """
    Returns: (qe, retriever, num_nodes, build_seconds, effective_model_id)
    """
    t_start = time.time()

    # choose model
    model_id = embed_model
    if multilingual and embed_model == "BAAI/bge-small-en-v1.5":
        model_id = "intfloat/multilingual-e5-small"  # smaller multilingual

    embed = HuggingFaceEmbedding(
        model_name=model_id,
        cache_folder=cache_dir,
        embed_batch_size=embed_batch_size or 16,
    )
    Settings.embed_model = embed

    # explicit split so we can count nodes
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=120)
    logging.info("Parsing documents -> nodes (chunk_size=1024 overlap=120)…")
    nodes = splitter.get_nodes_from_documents(docs)
    num_nodes = len(nodes)
    logging.info("Parsed %d nodes", num_nodes)

    # build index (embeds during build)
    index = VectorStoreIndex.from_nodes(nodes, show_progress=True)

    retriever = index.as_retriever(similarity_top_k=top_k)

    # postprocessors: cutoff + optional reranker
    post = [SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
    if use_reranker:
        post.insert(1, FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=top_k))
        logging.info("Reranker enabled (BAAI/bge-reranker-base).")
    else:
        logging.info("Reranker disabled.")

    # QE (passing retriever explicitly is fine on 0.10.x)
    qe = index.as_query_engine(
        retriever=retriever,
        node_postprocessors=post,
        # response_mode="compact",
    )

    build_seconds = time.time() - t_start
    return qe, retriever, num_nodes, build_seconds, model_id



# NEW ────────────────────────────────────────────────────────────────────────
@dataclass
class RunReport:
    date: str
    input_file: str
    output_file: str
    bytes_written: int
    docs_loaded: int
    nodes_embedded: int
    top_k: int
    similarity_cutoff: float
    token_cap: int
    llm: str
    embed_model: str
    embed_batch_size: int | None
    cache_dir: str | None
    reranker: bool
    started_at: float
    build_done_at: float
    finished_at: float
    build_seconds: float
    total_seconds: float
    warnings: list[str]

@dataclass
class QMetric:
    question: str
    n_candidates: int
    n_after_cutoff: int
    n_selected: int
    sim_min: float | None
    sim_max: float | None
    selected_token_sum: int

def _approx_tokens(text: str) -> int:
    # fast & dependency-free ≈4 chars/token heuristic
    return max(1, len(text) // 4)



def main():
    import argparse
    print("RAG: entering main()", flush=True)  # sentinel #2

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
    ap.add_argument("-v", "--verbose", action="count", default=0, help="-v=INFO, -vv=DEBUG")
    ap.add_argument("--dry-run", action="store_true", help="Load input and exit before embeddings/indexing")



    # argparse additions
    ap.add_argument("--embed", default="BAAI/bge-small-en-v1.5",
                    help="HF embedding model id; use a small model to avoid big downloads")
    ap.add_argument("--cache-dir", default=os.environ.get("TRANSFORMERS_CACHE"),
                    help="Where to store HF models (defaults to $TRANSFORMERS_CACHE)")
    ap.add_argument("--no-reranker", action="store_true", help="Disable reranker to avoid big model download")
    # add near your other args
    ap.add_argument("--embed-batch-size", type=int, default=None,
                    help="Batch size for sentence-transformer embedding (e.g., 16/32).")
    ap.add_argument("--no-report", action="store_true",
                    help="Disable writing .run.json/.run.md/.run.qmetrics.csv")


    args = ap.parse_args()

    configure_logging(args.verbose)
    log = logging.getLogger("RAG")
    log.info("Args: %s", vars(args))
    print("RAG: logging configured", flush=True)  # sentinel #3

    try:
        t0 = time.time()
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
        log.info("Input resolved to: %s", in_path)

        # LLM (optional)
        if args.llm == "openai":
            try:
                from llama_index.llms.openai import OpenAI as OpenAILLM
                from llama_index.core import Settings
            except Exception as e:
                log.error("OpenAI LLM import failed: %s", e)
                raise SystemExit("Install: pip install llama-index-llms-openai openai")
            Settings.llm = OpenAILLM(model=args.openai_model)
            log.info("LLM: OpenAI %s (API key present: %s)", args.openai_model, "yes" if os.getenv("OPENAI_API_KEY") else "NO")
        else:
            from llama_index.core import Settings  # ensures import errors surface here
            log.info("LLM: disabled (context-only)")

        # Load docs (use your loader)
        from llama_index.core import Document  # surface import timing
        t1 = time.time()
        docs = load_jsonl_as_documents(in_path)  # your existing function
        log.info("Loaded %d docs from %s in %.2fs", len(docs), in_path, time.time() - t1)
        if args.verbose >= 2 and docs:
            sample = docs[0].text[:200].replace("\n", " ")
            log.debug("Sample doc[0]: %r …", sample)

        if args.dry_run:
            log.warning("Dry-run: stopping after loading. Docs=%d", len(docs))
            print("RAG: dry-run complete", flush=True)
            return

        if not docs:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                f"---\nconfig:\n  theme: redux\n  layout: fixed\n---\n# Daily FAQ — {in_path.stem}\n\n_No notes found._\n",
                encoding="utf-8",
            )
            log.warning("No docs found; wrote empty report to %s", args.out)
            print("RAG: no docs, wrote empty report", flush=True)
            return

        # Build query engine
        log.info("Building query engine (top_k=%d, token_cap=%d, multilingual=%s)…",
                 args.top_k, args.token_cap, args.multilingual)
        print("RAG: building query engine…", flush=True)  # sentinel #4

        # Import here so if it hangs, faulthandler will dump stacks while you see this line
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.core.postprocessor import SimilarityPostprocessor
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        import tiktoken
        # your TokenCapPostprocessor / build_query_engine should be defined above
        qe, retriever = build_query_engine(
            docs,
            multilingual=args.multilingual,
            top_k=args.top_k,
            token_cap=args.token_cap,
            embed_model=args.embed,
            cache_dir=args.cache_dir,
            use_reranker=not args.no_reranker,
        )



        log.info("Query engine ready (elapsed %.2fs total so far)", time.time() - t0)
        print("RAG: query engine ready", flush=True)  # sentinel #5

        # Q&A
        questions = FAQS_ES if args.lang == "es" else FAQS_EN
        log.info("Answering %d questions (mode=%s)…", len(questions), args.llm)
        print("RAG: answering questions…", flush=True)  # sentinel #6
        if args.llm == "openai":
            results = answer_with_llm(qe, questions)
        else:
            results = answer_context_only(retriever, questions, per_q=min(5, args.top_k))

        # Render
        md = render_markdown(day_label=in_path.stem, results=results)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        log.info("Wrote %s (%d bytes)", out_path, out_path.stat().st_size)
        print("RAG: done", flush=True)  # sentinel #7

    except SystemExit:
        raise
    except Exception as e:
        logging.getLogger("RAG").error("Unhandled error: %s\n%s", e, traceback.format_exc())
        print("RAG: exception — see traceback above", flush=True)
        raise



import sys, os, time, traceback
from pathlib import Path

def eprint(msg):
    sys.stderr.write(msg + "\n"); sys.stderr.flush()
def sprint(msg):
    sys.stdout.write(msg + "\n"); sys.stdout.flush()

# ---------------- driver ----------------
if __name__ == "__main__":
    print("RAG: __main__ guard hit", flush=True)
    try:
        main()
    except SystemExit as e:
        print(f"RAG: sys.exit({e})", flush=True)
        raise
    except Exception as e:
        print("RAG: unhandled exception in main()", flush=True)
        raise
