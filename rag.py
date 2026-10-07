"""Retrieval and answering.

retrieve() turns a question into an embedding and asks the vector database
for the chunks whose embeddings are closest, i.e. the passages most similar
in meaning to the question. answer() hands those chunks to Claude as numbered
sources and returns a cited answer.

Run:  python rag.py --search "How many credits do I need to graduate?"
      python rag.py --ask "How do I claim AP credit?"
"""

import argparse
import math
from datetime import date
import re
from collections import Counter
from functools import lru_cache

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

import config
import prompts


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once. Uses the copy already downloaded to this
    computer, so a slow or missing internet connection can't hang startup;
    only the very first run downloads it (~130 MB)."""
    try:
        return SentenceTransformer(config.EMBEDDING_MODEL, local_files_only=True)
    except OSError:  # not downloaded yet
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


# A course number like "CHE 339", "ChE 339", "E S 333T", or the catalog's
# "CHE X39" (where X stands for the credit-hours digit).
# Subjects are 1-4 letters, or two single letters with a space ("E S", "C E").
COURSE_PATTERN = re.compile(r"\b([A-Z] [A-Z]|[A-Z]{1,4}) ?([X\d])(\d\d[A-Z]{0,2})\b")
# Words that look like a subject in "or 320M" or "Room 222" but aren't one.
NOT_SUBJECTS = {"A", "AN", "AND", "OR", "TO", "IN", "OF", "ON", "AT", "BY", "FOR", "THE",
                "IS", "BE", "AS", "IF", "NO", "SO", "UP", "WE", "IT", "MY", "PAGE", "ROOM",
                "RM", "PM", "AM", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "SEPT", "OCT", "WEEK", "HW", "EXAM", "QUIZ", "CH."}


def course_keys(text: str) -> set[str]:
    """Normalized course numbers mentioned in text: subject + last two digits +
    suffix, so 'CHE 339', 'ChE339' and 'CHE X39' all become 'CHE39'."""
    return {subject.replace(" ", "") + number
            for subject, _, number in COURSE_PATTERN.findall(text.upper().replace("_", " "))
            if subject not in NOT_SUBJECTS}


STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "can", "do", "does", "for", "how",
             "i", "if", "in", "is", "it", "my", "of", "on", "or", "should", "the", "to",
             "what", "when", "where", "which", "who", "will", "with", "you", "your", "me"}


# Different spellings of the same idea, mapped to one word for keyword search.
SYNONYMS = [(re.compile(r"\bpre[- ]?req(uisite)?s?\b", re.I), "prerequisite"),
            (re.compile(r"\bco[- ]?req(uisite)?s?\b", re.I), "corequisite"),
            (re.compile(r"\bq[- ]?drop", re.I), "qdrop")]


def tokenize(text: str) -> list[str]:
    """Words for keyword search: lowercase, no stopwords, a trailing plural 's'
    removed, plus normalized course numbers (so 'CHE X39' matches 'CHE 339')."""
    for pattern, word in SYNONYMS:
        text = pattern.sub(word, text)
    words = [singular(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
    return words + [k.lower() for k in course_keys(text)]


def singular(word: str) -> str:
    """Rough plural -> singular: classes -> class, electives -> elective,
    studies -> study. Leaves words like 'process', 'status', 'basis' alone."""
    if len(word) <= 3:
        return word
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


class KeywordIndex:
    """BM25, the classic search-engine scoring: a chunk scores higher the more
    often it contains the question's words, with rare words counting more and
    long chunks counting less. Complements embeddings, which match meaning but
    can miss exact terms like 'AP Chemistry' or 'repeat'."""

    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids, self.k1, self.b = ids, k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(word for d in self.docs for word in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def search(self, question: str, allowed: set[str] | None, limit: int) -> list[str]:
        """Ids of the best-matching chunks, best first (only ids in `allowed`, if given)."""
        terms = set(tokenize(question))
        scores = []
        for cid, doc, length in zip(self.ids, self.docs, self.lengths):
            if allowed is not None and cid not in allowed:
                continue
            score = sum(self.idf[t] * doc[t] * (self.k1 + 1)
                        / (doc[t] + self.k1 * (1 - self.b + self.b * length / self.avg_len))
                        for t in terms if t in doc)
            if score > 0:
                scores.append((score, cid))
        return [cid for _, cid in sorted(scores, reverse=True)[:limit]]


@lru_cache(maxsize=1)
def keyword_index(chunk_count: int) -> tuple[KeywordIndex, dict[str, dict]]:
    """Build the keyword index over every chunk (title + text), plus {id: metadata}.
    chunk_count is part of the cache key so the index rebuilds after re-ingesting."""
    data = get_collection().get(include=["documents", "metadatas"])
    texts = [f"{m['title']} {d}" for d, m in zip(data["documents"], data["metadatas"])]
    return KeywordIndex(data["ids"], texts), dict(zip(data["ids"], data["metadatas"]))


def retrieve(question: str, top_k: int = config.TOP_K,
             catalog_year: str | None = None) -> list[dict]:
    """Return the top_k most relevant chunks for a question, best first.

    Each result is {"text", "metadata", "score"}; score is cosine similarity
    (1.0 = identical meaning, around 0.5 or below = weakly related).

    Hybrid search: one ranking by meaning (embeddings) and one by keywords
    (BM25) are merged with reciprocal rank fusion, which rewards chunks that
    rank well in either list. At most config.MAX_PER_SOURCE chunks come from
    any one file, and config.MAX_PER_DOC_TYPE limits how many come from
    syllabi, so five syllabi for the same course can't crowd out the degree plan.

    If catalog_year is given (e.g. "2026-28"), chunks tied to a different
    catalog are left out; see year_matches().
    """
    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("The database is empty. Run: python ingest.py")

    pool = top_k * config.CANDIDATE_MULTIPLIER
    index, metas = keyword_index(collection.count())
    years = {m["catalog_year"] for m in metas.values()}
    ok_years = [y for y in years if year_matches(y, catalog_year)]
    where = {"catalog_year": {"$in": ok_years}} if catalog_year else None
    query_vec = embed_query(question)
    result = collection.query(query_embeddings=[query_vec], n_results=pool, where=where,
                              include=["documents", "metadatas", "distances", "embeddings"])
    hits = {cid: {"text": text, "metadata": meta, "score": 1 - dist, "vec": vec}  # distance = 1 - similarity
            for cid, text, meta, dist, vec in zip(result["ids"][0], result["documents"][0],
                                                  result["metadatas"][0], result["distances"][0],
                                                  result["embeddings"][0])}
    by_meaning = list(hits)

    allowed = ({cid for cid, m in metas.items() if m["catalog_year"] in ok_years}
               if catalog_year else None)
    by_keyword = index.search(question, allowed, pool)

    # Keyword-only finds still need a similarity score: fetch their vectors.
    missing = [cid for cid in by_keyword if cid not in hits]
    if missing:
        got = collection.get(ids=missing, include=["documents", "metadatas", "embeddings"])
        for cid, text, meta, vec in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]):
            hits[cid] = {"text": text, "metadata": meta, "vec": vec,
                         "score": float(sum(a * b for a, b in zip(query_vec, vec)))}

    # Reciprocal rank fusion: 1/(k + rank) from each list, summed.
    fused: Counter[str] = Counter()
    for ranking, weight in ((by_meaning, 1.0), (by_keyword, config.KEYWORD_WEIGHT)):
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] += weight / (config.RRF_K + rank)

    # Keyword matches on common words can drag in unrelated chunks; drop any
    # chunk much less similar in meaning than the best one.
    floor = max(h["score"] for h in hits.values()) - config.MAX_SCORE_GAP

    results, per_source, per_type = [], Counter(), Counter()
    for cid, _ in fused.most_common():
        if hits[cid]["score"] < floor:
            continue
        meta = hits[cid]["metadata"]
        type_cap = config.MAX_PER_DOC_TYPE.get(meta["doc_type"], top_k)
        # Two sections' syllabi often share whole paragraphs; keep one copy.
        duplicate = any(sum(a * b for a, b in zip(hits[cid]["vec"], kept["vec"])) > config.DUPLICATE_SIMILARITY
                        for kept in results)
        if (not duplicate and per_source[meta["source"]] < config.MAX_PER_SOURCE
                and per_type[meta["doc_type"]] < type_cap):
            results.append(hits[cid])
            per_source[meta["source"]] += 1
            per_type[meta["doc_type"]] += 1
        if len(results) == top_k:
            break
    return [{k: v for k, v in hit.items() if k != "vec"} for hit in results]


def catalog_range(year: str) -> tuple[int, int] | None:
    """'2026-28' -> (2026, 2028). None for terms like 'Fall 2026', 'unknown', etc."""
    m = re.fullmatch(r"(20\d\d)-(\d\d)", year)
    return (int(m.group(1)), 2000 + int(m.group(2))) if m else None


def year_matches(chunk_year: str, selected: str | None) -> bool:
    """Should a chunk be searched when the student picked catalog `selected`?

    Only chunks tied to a *different* catalog are excluded. Chunks with a
    catalog range overlapping the selection are kept (Tech Electives
    "2020-28" covers every catalog from 2020 on), and so is everything not
    tied to a catalog: syllabi (labeled by term) and undated handouts.
    """
    chunk, picked = catalog_range(chunk_year), catalog_range(selected or "")
    if picked is None or chunk is None:
        return True
    return chunk[0] < picked[1] and picked[0] < chunk[1]


def catalog_years() -> list[str]:
    """Catalog choices for the UI's year selector, newest first."""
    return config.CATALOG_YEARS


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
    except anthropic.BadRequestError as err:
        if "usage limits" in str(err):  # the spend limit you set in the Console
            raise RuntimeError("The monthly spend limit set in the Claude Console has been "
                               "reached. Raise it there (Settings > Billing) or wait until next month.")
        raise RuntimeError(f"The Anthropic API rejected the request: {err.message}")
    except anthropic.RateLimitError as err:
        if "enforced_spend_limit_reached" in str(err.body):  # the tier's monthly cap
            raise RuntimeError("This account's monthly API spend cap has been reached; "
                               "access resumes at the start of next month.")
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

    if not hits or max(h["score"] for h in hits) < config.MIN_SCORE:
        return {"answer": prompts.NOT_COVERED, "sources": [], "search_query": query, "covered": False}

    notes = [f"Today's date: {date.today():%B %d, %Y}."]
    if catalog_year:
        notes.append(f"The student says their catalog year is {catalog_year}.")
    user_msg = f"{format_sources(hits)}\n\n{' '.join(notes)}\n\nStudent question: {question}"
    text = ask_llm(prompts.SYSTEM_PROMPT, history + [{"role": "user", "content": user_msg}],
                   max_tokens=config.MAX_ANSWER_TOKENS)
    return {"answer": text + source_list(text, hits), "sources": hits,
            "search_query": query, "covered": True}


def chat(catalog_year: str | None, debug: bool = False) -> None:
    """Simple terminal chat for testing follow-up questions. Empty line quits."""
    history: list[dict] = []
    print("Ask a question (press Enter on an empty line to quit).")
    while question := input("\nYou: ").strip():
        try:
            result = answer(question, history, catalog_year)
        except RuntimeError as err:  # temporary API problems: report and keep chatting
            print(f"\n[{err}]")
            continue
        if debug:
            print(f"  (searched for: {result['search_query']})")
        print(f"\nAdvisor: {result['answer']}")
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": result["answer"]}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the advising documents.")
    parser.add_argument("--search", metavar="QUESTION", help="print the top matching chunks")
    parser.add_argument("--ask", metavar="QUESTION", help="answer a question with citations")
    parser.add_argument("--chat", action="store_true", help="multi-turn chat in the terminal")
    parser.add_argument("--debug", action="store_true", help="with --chat, show the search query")
    parser.add_argument("--year", help="student's catalog, e.g. '2026-28' (leaves out other catalogs' documents)")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    if args.ask:
        result = answer(args.ask, catalog_year=args.year)
        print(result["answer"])
        return
    if args.chat:
        chat(args.year, args.debug)
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
