"""Web chat interface for the advising bot.

Run:  streamlit run app.py
Then open http://localhost:8501 in your browser.

Streamlit re-runs this whole file top to bottom every time you send a
message; st.session_state is what remembers the conversation between runs.
"""

import os

import streamlit as st

import config
import prompts
import rag

DISCLAIMER = "Not official advice. Confirm with your advisor before registering."
ANY_YEAR = "Not sure / show all"
EXAMPLES = [
    "What are the prerequisites for CHE 354?",
    "What AP Chemistry score do I need for credit?",
    "Which technical areas does CHE 339 count for?",
    "When is the latest I can take CHE 319 and still graduate in 4 years?",
]

st.set_page_config(page_title="ChE Advising Assistant", page_icon="🎓")


@st.cache_resource(show_spinner="Loading the search model...")
def warm_up() -> int:
    """Load the embedding model and keyword index once, not on every message.
    Returns how many documents are searchable."""
    rag.get_model()
    _, metas = rag.keyword_index(rag.get_collection().count())
    return len({m["source"] for m in metas.values()})


def has_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


def search_only(question: str, year: str | None) -> dict:
    """Without an API key: show the best-matching passages instead of an answer."""
    hits = rag.retrieve(question, catalog_year=year)
    if not hits or max(h["score"] for h in hits) < config.MIN_SCORE:
        return {"answer": prompts.NOT_COVERED, "sources": [], "search_query": question}
    return {"answer": "**No API key is set, so I can't write an answer yet.** "
                      "These are the passages I would answer from. Add your key "
                      "to `.env` and restart to get real answers.",
            "sources": hits, "search_query": question}


def show_sources(sources: list[dict], expanded: bool = False) -> None:
    """Expandable panel listing the retrieved passages, numbered like the citations."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=expanded):
        for n, hit in enumerate(sources, start=1):
            m = hit["metadata"]
            st.markdown(f"**[{n}] {rag.describe(m)}**  \n"
                        f"{m['doc_type'].title()} · {m['catalog_year']} · "
                        f"match {hit['score']:.2f}")
            st.code(hit["text"], language=None, wrap_lines=True)


def show_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("search_query"):
            st.caption(f"Searched for: {msg['search_query']}")
        show_sources(msg.get("sources", []), expanded=msg.get("expanded", False))


# --- Sidebar ---------------------------------------------------------------
doc_count = warm_up()
with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Your catalog year", [ANY_YEAR] + rag.catalog_years(),
                          help="The catalog in effect when you entered UT. Documents "
                               "for other catalogs are left out of the search.")
    year = None if choice == ANY_YEAR else choice
    if st.button("Clear chat"):
        st.session_state.messages = []
    if not has_api_key():
        st.warning("No API key found in `.env`, so answers are turned off. "
                   "You can still see which documents match your question.")
    st.caption(f"Searching {doc_count} advising documents · model {config.LLM_MODEL}")

# --- Main page -------------------------------------------------------------
st.title("ChE Advising Assistant")
st.caption(f"{config.MAJOR} · {config.UNIVERSITY}")
st.info(DISCLAIMER)

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources", ...}]

for msg in st.session_state.messages:
    show_message(msg)

# A question comes from the chat box, or from clicking an example.
question = st.chat_input("Ask about courses, prerequisites, AP credit, policies...")
if not st.session_state.messages and not question:
    st.markdown("**Try asking:**")
    for example in EXAMPLES:
        if st.button(example, key=example):
            question = example

if question:
    show_message({"role": "user", "content": question})
    # Earlier turns, text only, for follow-up questions.
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.spinner("Searching the advising documents..."):
        try:
            result = (rag.answer(question, history, year) if has_api_key()
                      else search_only(question, year))
        except (RuntimeError, SystemExit) as err:  # API problems: show, don't crash
            result = {"answer": f"Sorry, something went wrong: {err}", "sources": []}

    reply = {"role": "assistant", "content": result["answer"], "sources": result["sources"],
             "expanded": not has_api_key(),  # in search-only mode the passages ARE the answer
             # Only show the search when a follow-up was rewritten into something new.
             "search_query": (result.get("search_query")
                              if result.get("search_query", question) != question else None)}
    show_message(reply)
    st.session_state.messages += [{"role": "user", "content": question}, reply]
    st.rerun()  # redraw so the example buttons disappear after the first question
