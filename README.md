
# textflow-core

**textflow-core** is a lightweight, hackable pipeline for **Retrieval-Augmented Generation (RAG)** on personal or team knowledge. It ingests day files (JSONL), parses them to Markdown nodes, **embeds** them, stores vectors in **ChromaDB**, and exposes a fast **CLI** for retrieval, reporting, and observability.

Built on **LlamaIndex**, **HuggingFace / Jina / OpenAI embeddings**, and **Chroma**, with a focus on **repeatable runs, idempotent ingest, and rich run reports**.

> Use it to study repos, generate FAQs, synthesize daily notes, and produce “what happened today” digests—quickly.

---

## Features

* **Idempotent ingest**: JSONL ➝ Markdown ➝ Nodes with stable IDs, cached embeddings, and Chroma upserts.
* **Pluggable embeddings**: HuggingFace (BGE/E5), Jina, or OpenAI. SQLite caching layer avoids re-embedding.
* **Chroma vector store**: persistent, reusable collections; fast retrieval; telemetry-off option.
* **Observability**: every run writes `.run.json` + `.run.md` (counts, timings, config, warnings).
* **CLI workflow**: single-day focus (“work this date”), switch stores, toggle reranker, adjust cutoffs/top-k.
* **LLM-optional**: run pure retrieval or attach an LLM for polished answers.
* **Version-tolerant** LlamaIndex glue: resilient to minor API changes.

---

## Repo layout

```
.
├── ingest/                  # Day files (YYYY-MM-DD.jsonl)
├── pipeline/
│   ├── RAG.py               # Main RAG CLI
│   ├── export.py, loader.py, inspector.py, cluster.py, polish.py
├── snippetflow/
│   ├── ingest.py            # Ingest helpers (idempotent)
│   ├── storage.py           # Chroma helpers (upsert, load, sanitize)
│   ├── embedder.py          # Jina/OpenAI/HF factory + SQLite cache
│   └── retriever.py         # Builders/loaders to attach indexes
├── store/chroma/            # Chroma persistent store (created on demand)
├── exports/                 # Outputs + .run.{json,md} reports
├── requirements.txt
└── README.md
```

---

## Install

```bash
# 1) Create & activate env
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -U pip wheel
pip install -r requirements.txt

# Optional (FlagEmbedding reranker)
# pip install -U FlagEmbedding
```

> **Tip:** Set a cache for HuggingFace: `export HF_HOME=/tmp/hf_cache` (Linux/macOS).

Disable Chroma telemetry (recommended):

```bash
export CHROMA_DISABLE_TELEMETRY=1
```

---

## Data format (JSONL ➝ Markdown)

Place day files in `ingest/` as `YYYY-MM-DD.jsonl`. Each line is one record. Minimal schema:

```json
{"text": "Your note or message...", "title": "optional", "timestamp": 1723516800000}
```

The pipeline converts JSONL to Markdown blocks and then to LlamaIndex `TextNode`s with stable IDs and helpful metadata (`file`, `header_path`, `title`, `timestamp`).

---

## Quickstart

### 1) Ingest & embed (idempotent)

Use the **ingest helpers** if you want full control over caching & upserts (recommended for long-running corpora):

```bash
python scripts/run_ingest.py --input ingest --sqlite embeds.sqlite --chroma store/chroma --collection gpt_logs
```

* Embeddings are cached in `embeds.sqlite`.
* Vectors/documents upserted to Chroma (`store/chroma`, collection `gpt_logs`).
* Safe to re-run; only new content is embedded.

### 2) Query a single day (pure retrieval, no LLM)

```bash
rag --input ingest --date 2025-08-05 --out exports/2025-08-05.md \
    --lang es --llm none -vv \
    --use-chroma --chroma-path store/chroma --chroma-collection gpt_logs \
    --embed BAAI/bge-small-en-v1.5 --embed-batch-size 32 \
    --similarity-cutoff 0.32
```

Produces:

* `exports/2025-08-05.md` (answers + sources)
* `exports/2025-08-05.run.json` and `.run.md` (observability report)

### 3) Same, with LLM polish (OpenAI as example)

```bash
export OPENAI_API_KEY=sk-...
rag --input ingest --date 2025-08-06 --out exports/2025-08-06.md \
    --lang en --llm openai --openai-model gpt-4o-mini -v \
    --use-chroma --chroma-path store/chroma --chroma-collection gpt_logs \
    --similarity-cutoff 0.35 --top_k 8
```

---

## CLI options (high-signal ones)

```text
--input PATH             JSONL file or directory with day files
--date YYYY-MM-DD        Focus on a single day when --input is a directory
--out FILE               Output Markdown report

--lang {es,en}           Language for outputs
--llm {none,openai}      None = retrieval-only
--openai-model MODEL     e.g., gpt-4o-mini

--use-chroma             Persist & reuse vectors
--chroma-path DIR        Chroma data dir (default: store/chroma)
--chroma-collection NAME Collection name (e.g., gpt_logs)

--embed MODEL            HF model id (e.g., BAAI/bge-small-en-v1.5)
--embed-batch-size INT   Tune for CPU/RAM (32 is a good start)
--embed-device {cpu,cuda}
--cache-dir DIR          HF cache (e.g., /tmp/hf_cache)

--top_k INT              Retriever depth (default: 8)
--similarity-cutoff F    Filter weak hits (default: 0.32)
--no-reranker            Skip reranking (FlagEmbedding optional)

-v / -vv                 Verbose / debug logs
--dry-run                Build but do not write output
```

---

## Embedding choices

* **HuggingFace** (default):
  `BAAI/bge-small-en-v1.5` (fast, strong),
  `intfloat/e5-large-v2` (bigger),
  `intfloat/multilingual-e5-small` (multilingual).
* **Jina**: `jina-embeddings-v3` (good for multilingual; cached via SQLite).
* **OpenAI**: `text-embedding-3-small` (simple to use).

Switch providers via `snippetflow/embedder.py` or CLI flags. The pipeline is **version-tolerant** to common LlamaIndex changes.

---

## Observability

Each run writes a **RunReport**:

* `*.run.json`: all params (files, counts, timings, models, flags, warnings).
* `*.run.md`: human-readable summary.
* Useful for comparing runs across **days**, **models**, **cutoffs**, **top-k**, and **stores**.

Example fields:

* docs\_loaded, nodes\_embedded (or reused), top\_k, similarity\_cutoff
* token\_cap, llm, embed\_model, cache\_dir
* reranker on/off
* build\_seconds, total\_seconds
* warnings (e.g., reranker unavailable)

---

## Performance tips

* CPU-only? Use **`BAAI/bge-mini-en-v1.5`** or **`bge-small`**; set `--embed-batch-size 32`.
* Cache HF downloads with `--cache-dir /tmp/hf_cache` (or `HF_HOME`).
* Keep Chroma on SSD: `--chroma-path /fast/chroma`.
* Disable telemetry: `export CHROMA_DISABLE_TELEMETRY=1`.
* Reranker is optional; install `FlagEmbedding` for better ranking.

---

## Troubleshooting

* **“collection is empty”**: Run ingest first (populates Chroma).
* 
---

## Roadmap

* Multi-day, date-range retrieval & reports
* Query presets (FAQ, changelog, executive summary)
* Better inspector UI (source browsing)
* Optional **semantic filters** (by header/path/title)
* More rerankers & hybrid retrieval

---

## Contributing

PRs and issues welcome. 
