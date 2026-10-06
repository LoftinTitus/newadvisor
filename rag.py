"""Retrieval and answering.

retrieve() turns a question into an embedding and asks the vector database
for the chunks whose embeddings are closest, i.e. the passages most similar
in meaning to the question. answer() hands those chunks to Claude as numbered
sources and returns a cited answer.

Run:  python rag.py --search "How many credits do I need to graduate?"
      python rag.py --ask "How do I claim AP credit?"
"""

import argparse
import re
from functools import lru_cache

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

import config
import prompts


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once (first call downloads it, ~130 MB)."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    """Open (or create) the Chroma collection stored in data/chroma/.

    We compute embeddings ourselves, so Chroma's built-in embedder is turned off.
    "cosine" means similarity is measured by the angle between vectors.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks. Vectors are normalized to length 1 so cosine
    similarity is a simple dot product."""
    return get_model().encode(texts, normalize_embeddings=True, batch_size=32,
                              show_progress_bar=len(texts) > 100).tolist()


def embed_query(question: str) -> list[float]:
    """Embed a question. BGE expects queries (not passages) to carry an instruction prefix."""
    return get_model().encode(config.QUERY_INSTRUCTION + question,
                              normalize_embeddings=True).tolist()


def retrieve(question: str, top_k: int = config.TOP_K,
             catalog_year: str | None = None) -> list[dict]:
    """Return the top_k most relevant chunks for a question, best first.

    Each result is {"text", "metadata", "score"}; score is cosine similarity
    (1.0 = identical meaning, around 0.5 or below = weakly related).
    If catalog_year is given, only chunks from that year, plus chunks whose
    year is unknown (most policies and handouts), are searched.
    """
    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("The database is empty. Run: python ingest.py")

    where = {"catalog_year": {"$in": [catalog_year, "unknown"]}} if catalog_year else None
    result = collection.query(query_embeddings=[embed_query(question)],
                              n_results=top_k, where=where)
    return [
        {"text": text, "metadata": meta, "score": 1 - dist}  # Chroma returns distance = 1 - similarity
        for text, meta, dist in zip(result["documents"][0], result["metadatas"][0],
                                    result["distances"][0])
    ]


def catalog_years() -> list[str]:
    """All catalog_year values in the database (for the UI's year selector)."""
    metas = get_collection().get(include=["metadatas"])["metadatas"]
    return sorted({m["catalog_year"] for m in metas} - {"unknown"})


def describe(meta: dict) -> str:
    """Short human-readable location, e.g. 'Advising (p. 3-4)'."""
    where = f"p. {meta['page']}" if meta.get("page") else meta.get("section", "")
    return f"{meta['title']} ({where})" if where else meta["title"]


# --------------------------------------------------------------------------
# Answering (Phase 3)
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """The Anthropic API client (needs ANTHROPIC_API_KEY in .env)."""
    return anthropic.Anthropic(api_key=config.get_api_key())


def ask_llm(system: str, messages: list[dict], max_tokens: int) -> str:
    """Send one request to Claude and return its text, with readable errors."""
    try:
        response = get_client().messages.create(
            model=config.LLM_MODEL, max_tokens=max_tokens,
            system=system, messages=messages,
        )
    except anthropic.AuthenticationError:
        raise SystemExit("The API key was rejected. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.NotFoundError:
        raise SystemExit(f"Model '{config.LLM_MODEL}' not found. Check LLM_MODEL in .env.")
    except anthropic.RateLimitError:
        raise RuntimeError("Too many requests right now. Wait a minute and try again.")
    except anthropic.APIConnectionError:
        raise RuntimeError("Couldn't reach the Anthropic API. Check your internet connection.")
    except anthropic.APIStatusError as err:
        raise RuntimeError(f"The Anthropic API returned an error ({err.status_code}). Try again shortly.")
    return "".join(b.text for b in response.content if b.type == "text").strip()


def recent_history(history: list[dict]) -> list[dict]:
    """The last HISTORY_TURNS exchanges (one user + one assistant message each)."""
    return [{"role": m["role"], "content": m["content"]}
            for m in history[-2 * config.HISTORY_TURNS:]]


def standalone_question(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up ("what about spring?") into a self-contained question
    so retrieval searches for the right thing. No history means no rewrite."""
    if not history:
        return question
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    return ask_llm(prompts.REWRITE_PROMPT,
                   [{"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"}],
                   max_tokens=200) or question


def format_sources(hits: list[dict]) -> str:
    """Number each retrieved chunk as a source block for the prompt."""
    blocks = []
    for n, hit in enumerate(hits, start=1):
        m = hit["metadata"]
        blocks.append(f'<source id="{n}" document="{describe(m)}" type="{m["doc_type"]}" '
                      f'catalog_year="{m["catalog_year"]}">\n{hit["text"]}\n</source>')
    return "\n\n".join(blocks)


def source_list(text: str, hits: list[dict]) -> str:
    """The 'Sources' footer: only the numbers the answer actually cited."""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    if not cited:
        return ""
    return "\n\nSources:\n" + "\n".join(f"[{n}] {describe(hits[n - 1]['metadata'])}" for n in cited)


def answer(question: str, history: list[dict] | None = None,
           catalog_year: str | None = None) -> dict:
    """Answer a student question from the documents.

    history is the chat so far as [{"role": "user"|"assistant", "content": str}].
    Returns {"answer": str, "sources": the retrieved chunks, "search_query": str,
    "covered": False if nothing relevant was found and the LLM was skipped}.
    """
    history = recent_history(history or [])
    query = standalone_question(question, history)
    hits = retrieve(query, catalog_year=catalog_year)

    if not hits or hits[0]["score"] < config.MIN_SCORE:
        return {"answer": prompts.NOT_COVERED, "sources": [], "search_query": query, "covered": False}

    year_note = f"\n(The student says their catalog year is {catalog_year}.)" if catalog_year else ""
    user_msg = f"{format_sources(hits)}\n\nStudent question: {question}{year_note}"
    text = ask_llm(prompts.SYSTEM_PROMPT, history + [{"role": "user", "content": user_msg}],
                   max_tokens=config.MAX_ANSWER_TOKENS)
    return {"answer": text + source_list(text, hits), "sources": hits,
            "search_query": query, "covered": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the advising documents.")
    parser.add_argument("--search", metavar="QUESTION", help="print the top matching chunks")
    parser.add_argument("--ask", metavar="QUESTION", help="answer a question with citations")
    parser.add_argument("--year", help="limit to one catalog year, e.g. '2026-28'")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    if args.ask:
        result = answer(args.ask, catalog_year=args.year)
        print(result["answer"])
        return
    if not args.search:
        parser.print_help()
        return
    for rank, hit in enumerate(retrieve(args.search, args.top_k, args.year), start=1):
        m = hit["metadata"]
        print(f"\n[{rank}] score {hit['score']:.3f}  {describe(m)}  "
              f"[{m['doc_type']}, {m['catalog_year']}]")
        print("    " + hit["text"][:400].replace("\n", "\n    ") + ("..." if len(hit["text"]) > 400 else ""))


if __name__ == "__main__":
    main()
