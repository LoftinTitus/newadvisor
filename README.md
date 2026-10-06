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

## Where things are

| File / folder | What it is |
| --- | --- |
| `documents/` | Your source documents. Each subfolder name (Curriculum, Policies, Syllabi) becomes the document type. Not committed to git. |
| `config.py` | All settings in one place: paths, chunk size, how many results to retrieve, model names. |
| `requirements.txt` | The libraries and exact versions this project uses. |
| `.env.example` | Template for `.env`, which holds your API key. |
| `data/chroma/` | The search database, built later by `ingest.py`. Not committed. |
| `eval/` | Test questions for checking answer quality (Phase 5). |
