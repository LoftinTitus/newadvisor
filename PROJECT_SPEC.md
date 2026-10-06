# RAG Academic Advisor – Build Prompt for VS Code AI

## How to use this file

Save this file in your project folder as `PROJECT_SPEC.md` and point your VS Code AI assistant at it. Fill in the bracketed blanks in the last section first.

1. Create an empty folder (e.g. `advisor-bot`) and open it in VS Code.
2. Put this file there as `PROJECT_SPEC.md`. If you use Claude Code, also copy the "Coding rules" section into a `CLAUDE.md` file so it is read automatically every session.
3. Start the assistant with: "Read PROJECT_SPEC.md. Do Phase 1 only. Stop and show me how to test it before moving on."
4. After each phase, run the test listed under it yourself. Only then say "Phase N passed, start Phase N+1."
5. If it goes off track, say "Re-read PROJECT_SPEC.md and follow the current phase only."

Going one phase at a time matters: AI assistants do much better on small, testable steps than on "build the whole thing."

## Project goal (for the AI)

You are helping me build a retrieval-augmented generation (RAG) chatbot that answers academic advising questions for Chemical Engineering at The University of Texas at Austin. It must answer only from my documents and cite which document each answer came from. Do not fine-tune or train any model.

The source documents are: advising Canvas pages, past syllabi, the student handbook, the course catalog / degree requirements, and advising FAQs. They will be PDFs, .docx files, .html pages saved from Canvas, and .txt/.md files in `data/raw/`.

The system has two parts:

- **Ingestion (run occasionally):** load every file in `data/raw/`, extract text, clean it, split it into chunks, embed the chunks, and store them in a local vector database with metadata (source file, document type, catalog year, page or section).
- **Chat (run constantly):** take a student question, retrieve the most relevant chunks, send them with the question to an LLM, and return an answer with citations through a simple web chat UI.

The user is a student, not a professional developer. Explain what each file does in plain language, keep the code simple and well commented, and prefer fewer files over clever abstractions.

## Tech stack and constraints

Use plain Python with a few small libraries, no LangChain or LlamaIndex, so every step stays visible and debuggable.

| Part | Use | Why |
| --- | --- | --- |
| Language | Python 3.11+ in a `venv` virtual environment | Simple, best library support |
| PDF text | `pymupdf` | Fast, keeps page numbers |
| Word files | `python-docx` | Reads .docx paragraphs and tables |
| Canvas HTML | `beautifulsoup4` | Strips navigation and markup |
| Embeddings | `sentence-transformers` with `BAAI/bge-small-en-v1.5`, run locally | Free, no API key, good quality |
| Vector store | `chromadb` in persistent local mode (`data/chroma/`) | No server to run |
| LLM | Anthropic API via the `anthropic` SDK; model name read from `.env` (start with `claude-haiku-4-5-20251001`) | Cheap and fast; swappable later |
| Web UI | `streamlit` chat interface | Chat UI in pure Python |
| Config | `python-dotenv` reading `.env` | Keeps the API key out of code |

Hard constraints:

- The API key lives only in `.env`, which is listed in `.gitignore`. Never print it, log it, or hard-code it.
- `data/raw/` and `data/chroma/` are also in `.gitignore`, since the source documents may not be mine to publish.
- Everything must run on a normal laptop with no GPU.
- Pin library versions in `requirements.txt`.

## Repo structure

Create exactly this layout; add new files only if a phase truly needs one, and tell me why.

```
advisor-bot/
├── PROJECT_SPEC.md        # this document
├── README.md              # setup and run steps in plain language
├── requirements.txt
├── .env.example           # ANTHROPIC_API_KEY=, LLM_MODEL=, etc. (no real values)
├── .gitignore             # .env, venv/, data/raw/, data/chroma/
├── config.py              # loads .env; chunk size, top_k, paths, model names
├── ingest.py              # load -> clean -> chunk -> embed -> store
├── rag.py                 # retrieve() and answer() functions
├── prompts.py             # the system prompt (see below)
├── app.py                 # Streamlit chat UI
├── eval/
│   ├── questions.csv      # test questions + expected source + expected answer
│   └── run_eval.py
└── data/
    ├── raw/
    │   ├── handbook/
    │   ├── catalog/
    │   ├── syllabi/
    │   └── canvas/
    └── chroma/            # created by ingest.py
```

The subfolder a file sits in under `data/raw/` becomes its `doc_type` metadata. A year in the file name (e.g. `handbook_2025-26.pdf`) becomes its `catalog_year`.

## Build phases

Build in this order and stop after each phase until I confirm its test passes.

### Phase 0 — Setup

- Create the folder layout, `requirements.txt`, `.gitignore`, `.env.example`, `config.py`, and a README with setup steps (create venv, install, copy `.env.example` to `.env`).
- **Test:** `python -c "import config; print('ok')"` prints ok and no secrets.

### Phase 1 — Load and chunk (no embeddings yet)

- In `ingest.py`, write one loader per file type (PDF, .docx, .html, .txt/.md) that returns text plus metadata: `source`, `doc_type`, `catalog_year`, and `page` or `section`.
- Clean text: strip Canvas navigation, repeated headers/footers, and page numbers; collapse whitespace.
- Chunk by structure first (headings, paragraphs), then by size: about 300–400 words with about 50 words of overlap. Never split a table or a numbered requirements list mid-way if it fits in one chunk.
- Add a `--dry-run` flag that prints a summary (files read, chunks per file, average chunk length) and writes 10 random chunks to `chunks_preview.txt`.
- **Test:** I open `chunks_preview.txt` and every chunk reads as coherent text with correct metadata.

### Phase 2 — Embed, store, retrieve

- Embed chunks with the local model (normalized vectors, cosine distance) and store them in Chroma with their metadata.
- For queries, prepend the BGE query instruction: `Represent this sentence for searching relevant passages: `
- Make re-runs incremental: store a hash of each file and only re-process files that changed; remove chunks for deleted files.
- In `rag.py`, write `retrieve(question, top_k=6, catalog_year=None)` returning chunks, metadata, and scores.
- Add a CLI: `python rag.py --search "How many credits do I need to graduate?"` prints the top chunks with source and score.
- **Test:** for 5 questions I already know the answers to, the right document shows up in the top 3.

### Phase 3 — Answer with citations

- In `rag.py`, write `answer(question, history)` that retrieves chunks, builds a prompt with the system prompt from `prompts.py`, numbers each chunk as a source block, and calls the Anthropic API.
- The answer must cite sources inline like [1], [2] and end with a list mapping numbers to file name and page/section.
- If the best retrieval score is below a threshold in `config.py`, skip the LLM and reply that the documents don't cover it and to ask a human advisor.
- Include the last 4 chat turns for follow-up questions, and rewrite a follow-up into a standalone question before retrieving.
- **Test:** `python rag.py --ask "..."` gives a correct, cited answer, and an off-topic question ("best pizza in town?") is declined.

### Phase 4 — Chat UI

- `app.py`: a Streamlit chat with message history, a sidebar catalog-year selector, an expandable "Sources" panel under each answer showing the retrieved text, and a visible disclaimer: "Not official advice. Confirm with your advisor before registering."
- **Test:** `streamlit run app.py` works in the browser; sources expand; the year filter changes results.

### Phase 5 — Evaluation

- `eval/questions.csv` with columns `question, expected_source, expected_answer`; I will write 20–30 rows.
- `run_eval.py` reports retrieval hit rate (expected source in top 6) and saves each generated answer next to the expected one in `eval/results.csv` for me to grade.
- **Test:** hit rate is at least 85%. If not, suggest changes to chunk size, top_k, or cleaning, and re-run.

### Phase 6 — Deploy (only when I ask)

- Prepare for Streamlit Community Cloud or Render: secrets set in the host's dashboard, not in the repo.
- Because `data/raw/` is not committed, either commit the built `data/chroma/` folder (only if I confirm the content may be public) or add a password gate using a secret.
- Add simple rate limiting per session so a stranger can't run up my API bill.

## System prompt for the advisor bot

Put this text in `prompts.py` as `SYSTEM_PROMPT`, filling in the brackets from config.

```
You are an academic advising assistant for Chemical Engineering students at The University of Texas at Austin.

You answer ONLY from the numbered source excerpts provided with each question.
- Cite every factual claim with its source number, like [2].
- If the sources do not contain the answer, say so plainly and suggest
  contacting an academic advisor. Never fill gaps from general knowledge.
- If sources disagree (e.g. different catalog years), point out the
  conflict, give both, and note which catalog year each is from.
- Requirements depend on the student's catalog year. If it matters and the
  student hasn't said, ask which year they entered.
- Be concise and practical: lead with the direct answer, then details.
- Never invent course numbers, credit counts, deadlines, or policies.
- Do not give advice about a specific student's grades, records, financial
  aid, or personal circumstances; refer them to an advisor.
- For anything that affects graduation or registration, end with:
  "Please confirm with your advisor before acting on this."
```

## Coding rules for the AI assistant

Follow these in every session.

- Work on the current phase only. When it's done, stop, list the files you changed, and tell me exactly how to test it.
- Before writing code for a phase, give me a 3–5 line plan and wait for "go".
- Explain new concepts (embeddings, chunking, vector search) in one or two plain sentences the first time they come up.
- Keep functions short with docstrings and comments on the non-obvious lines. Use type hints.
- Put every tunable number (chunk size, overlap, top_k, score threshold, model names) in `config.py`, never scattered in code.
- Handle errors with clear messages: a corrupt PDF gets skipped with a warning, not a crash; a missing API key says which variable to set.
- Never commit, print, or log secrets or the contents of `data/raw/`.
- Don't add dependencies beyond the stack table without asking me first.
- Don't rewrite working code from earlier phases unless the current phase requires it; if it does, say what changed and why.
- If you're unsure what I want, ask one question instead of guessing.

## Fill in before starting

Replace the brackets throughout the file, then tick these off.

- [ ] University name: The University of Texas at Austin
- [ ] Major / program: Chemical Engineering
- [ ] Catalog years to include: [e.g. 2023-24, 2024-25, 2025-26]
- [ ] Anthropic API key created at console.anthropic.com and pasted into `.env` (with a monthly spend limit set)
- [ ] Documents downloaded into `data/raw/` subfolders, with the year in each file name
- [ ] Removed anything containing student names, grades, or other personal records (FERPA)
- [ ] Checked whether I'm allowed to use or publish the Canvas content and syllabi before deploying publicly
- [ ] Written 20–30 test questions with known answers for Phase 5
