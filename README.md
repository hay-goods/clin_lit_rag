# Clinical Literature RAG Agent

A local, conversational RAG (Retrieval-Augmented Generation) tool that lets you ask
plain-English questions against a corpus of clinical PDFs and receive sourced,
cited answers — powered by Claude and ChromaDB.

Drop in any PDFs (trial publications, practice guidelines, epidemiology studies),
run the ingestion script once, and start querying.

---

## What It Does

1. **Ingests** PDFs from `data/pdfs/` — extracts text, chunks it, embeds it with
   `sentence-transformers`, and stores everything in a local ChromaDB vector store.
2. **Retrieves** the top-5 most relevant chunks for any query.
3. **Synthesizes** a sourced answer via `claude-sonnet-4-6`, citing the filename and
   page number for every claim.
4. **Displays** results in a Gradio chat interface that runs entirely on your machine.

No external databases, no cloud vector stores, no authentication required.

---

## Example Queries

> "What is the prevalence of severe aortic stenosis in adults over 65?"

> "What does the PARTNER 3 trial say about 2-year mortality for TAVR vs surgery?"

> "What are the ACC/AHA guideline recommendations for TAVR indication?"

> "How does EVOLUT Low Risk compare to PARTNER 3 on stroke outcomes?"

These are structural heart disease examples, but the tool works for **any clinical domain** —
just swap in your own PDFs.

---

## Stack

| Component | Library |
|---|---|
| Chat UI | Gradio |
| LLM | Anthropic `claude-sonnet-4-6` |
| Vector store | ChromaDB (local, no server) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| PDF parsing | pypdf |

---

## Project Structure

```
clin-lit-rag/
├── app.py           ← Gradio chat interface
├── ingest.py        ← PDF → chunk → embed → ChromaDB (run once)
├── retriever.py     ← Query embedding + ChromaDB search
├── requirements.txt
├── .env.example
├── README.md
└── data/
    └── pdfs/        ← Drop your PDFs here before running ingest.py
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/clin-lit-rag.git
cd clin-lit-rag

# with pip
pip install -r requirements.txt

# or with uv (faster)
uv pip install -r requirements.txt
```

### 2. Add your Anthropic API key

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Add PDFs

Drop any clinical PDFs into `data/pdfs/`. The tool works with any PDFs —
trial publications, guidelines, meta-analyses, epidemiology reports.

**Suggested seed corpus for structural heart disease:**
- ACC/AHA 2021 Valvular Heart Disease Guidelines
- PARTNER 3 trial — Mack et al., NEJM 2019
- PARTNER 3 5-year outcomes — NEJM 2023
- EVOLUT Low Risk trial — Popma et al., NEJM 2019
- EARLY TAVR trial — NEJM 2024
- COAPT trial — Stone et al., NEJM 2018
- ESC 2021 Valvular Heart Disease Guidelines
- Nkomo et al. AS prevalence study

All of these are freely available from their respective journal websites.

### 4. Ingest

```bash
python ingest.py
```

This chunks every PDF, embeds the chunks, and stores them in `data/chroma/`.
Re-running is safe — it clears and rebuilds the collection from scratch.

Expected output:
```
Loading embedding model: all-MiniLM-L6-v2
Processing: partner3_nejm_2019.pdf
  Extracted 14 pages
  Embedding 87 chunks...
  Stored 87 chunks
...
Ingestion complete. 642 total chunks across 8 PDFs.
```

### 5. Launch the app

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

---

## How It Works

```
User question
     │
     ▼
Embed with all-MiniLM-L6-v2
     │
     ▼
ChromaDB cosine similarity search → top 5 chunks
     │
     ▼
Chunks + question → Claude (claude-sonnet-4-6)
     │
     ▼
Sourced answer + citation list
```

The system prompt instructs Claude to cite every claim using the format
`[Source: filename.pdf, p.N]` and to refuse to answer if the corpus doesn't
contain relevant information — minimising hallucination.

---

## Configuration

All tunable constants live at the top of each file:

| File | Constant | Default | Description |
|---|---|---|---|
| `ingest.py` | `CHUNK_TOKENS` | `500` | Target chunk size in tokens |
| `ingest.py` | `OVERLAP_TOKENS` | `50` | Overlap between chunks |
| `ingest.py` | `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `retriever.py` | `TOP_K` | `5` | Chunks retrieved per query |
| `app.py` | `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model ID |

---

## Adapting to a New Domain

This tool is domain-agnostic. To repurpose it:

1. Replace the PDFs in `data/pdfs/` with documents from your domain.
2. Update the example queries in `app.py` (`gr.Examples`) to match your use case.
3. Optionally edit the `SYSTEM_PROMPT` in `app.py` to give Claude domain-specific
   instructions.
4. Re-run `python ingest.py`.

No other changes required.

---

## Stretch Goals / Roadmap

- [ ] Streaming responses
- [ ] Auto-ingest new PubMed abstracts on a schedule via PubMed API
- [ ] Structured extraction — pull numbers into a comparison table
- [ ] Multi-document comparison mode
- [ ] Deploy to Hugging Face Spaces

---

## Requirements

- Python 3.11+
- Anthropic API key ([get one here](https://console.anthropic.com/))
- ~500 MB disk for the embedding model (downloaded automatically on first run)

---

## License

MIT
