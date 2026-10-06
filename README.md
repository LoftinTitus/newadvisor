# UT Chemical Engineering Advising Bot

A chatbot that answers academic advising questions for Chemical Engineering
students at UT Austin. It answers only from the documents in `documents/` and
cites which document each answer came from. See `PROJECT_SPEC.md` for the full
plan.

## One-time setup

You need Python 3.13. On this Mac it's at `/opt/homebrew/bin/python3.13`.
Newer versions like 3.14 may not work with every library yet.

1. **Create a virtual environment.** This is a private folder of libraries
   just for this project:
   ```
   /opt/homebrew/bin/python3.13 -m venv venv
   ```
2. **Turn it on.** Do this every time you open a new terminal:
   ```
   source venv/bin/activate
   ```
   Your prompt will start with `(venv)`.
3. **Install the libraries:**
   ```
   pip install -r requirements.txt
   ```
4. **Add your API key.** Copy the example settings file:
   ```
   cp .env.example .env
   ```
   Then open `.env` and paste your key from console.anthropic.com after
   `ANTHROPIC_API_KEY=`. `.env` is in `.gitignore`, so it never gets committed.

## Check that it works

```
python -c "import config; print('ok')"
```

This should print `ok` and nothing else.

## Using it

Run these with the venv turned on (`source venv/bin/activate`).

| Command | What it does |
| --- | --- |
| `python ingest.py` | Builds or updates the search database from `documents/`. Run it again whenever you add, change, or delete a document; only the changed files get re-processed. |
| `python ingest.py --dry-run` | Only splits the documents into chunks and writes 10 random ones to `chunks_preview.txt`, so you can check them. Doesn't touch the database. |
| `python rag.py --search "your question"` | Shows the passages that best match a question, with their scores. No API key needed. |
| `python rag.py --ask "your question"` | Answers the question with citations, using Claude. Needs your API key in `.env`. |

Add `--year 2026-28` to `--search` or `--ask` to limit results to one catalog year. Documents with no year are always included.

## Where things are

| File / folder | What it is |
| --- | --- |
| `documents/` | Your source documents. Each subfolder name (Curriculum, Policies, Syllabi) becomes the document type. Not committed to git. |
| `config.py` | All settings in one place: paths, chunk size, how many results to retrieve, model names. |
| `ingest.py` | Reads the documents, cleans them, splits them into chunks, and stores them in the database. |
| `rag.py` | Finds the chunks that match a question (`retrieve`) and asks Claude to answer from them (`answer`). |
| `prompts.py` | The instructions given to Claude. |
| `requirements.txt` | The libraries and exact versions this project uses. |
| `.env.example` | Template for `.env`, which holds your API key. |
| `data/chroma/` | The search database, built later by `ingest.py`. Not committed. |
| `eval/` | Test questions for checking answer quality (Phase 5). |
