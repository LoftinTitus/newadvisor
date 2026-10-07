"""Web chat interface for the advising bot.

Run:  streamlit run app.py
Then open http://localhost:8501 in your browser.

Streamlit re-runs this whole file top to bottom every time you send a
message; st.session_state is what remembers the conversation between runs.
Colors and fonts are set in .streamlit/config.toml.
"""

import os
import re
import time

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
    "What's the latest I can take CHE 319 and still graduate on time?",
]
AVATARS = {"user": ":material/person:", "assistant": ":material/school:"}

st.set_page_config(page_title="ChE Advising Assistant", page_icon="🎓")


@st.cache_resource(show_spinner="Loading the search model...")
def warm_up() -> int:
    """Load the embedding model and keyword index once, not on every message.
    Returns how many documents are searchable."""
    rag.get_model()
    _, metas = rag.keyword_index(rag.database_version())
    return len({m["source"] for m in metas.values()})


def has_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


def search_only(question: str, year: str | None) -> dict:
    """Without an API key: show the best-matching passages instead of an answer."""
    hits = rag.retrieve(question, catalog_year=year)
    if not hits or max(h["score"] for h in hits) < config.MIN_SCORE:
        return {"answer": prompts.NOT_COVERED, "sources": [], "search_query": question}
    return {"answer": "Answers are turned off until an API key is added, but here "
                      "are the passages I'd answer from.",
            "sources": hits, "search_query": question}


def plain(text: str) -> str:
    """Show document text as-is: escape characters Markdown would treat as
    formatting (like | tables, * bullets, $ math) and keep line breaks."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|~<>$])", r"\\\1", text).replace("\n", "  \n")


def show_sources(sources: list[dict], expanded: bool = False) -> None:
    """Expandable list of the retrieved passages, numbered like the citations."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/description:", expanded=expanded):
        for n, hit in enumerate(sources, start=1):
            m = hit["metadata"]
            with st.container(border=True):
                st.markdown(f"**{n}.  {plain(rag.describe(m))}**")
                st.caption(f"{m['doc_type'].title()}  ·  {m['catalog_year']}  ·  "
                           f"match {hit['score']:.2f}")
                st.markdown(plain(hit["text"]))


def rate_limit_message() -> str | None:
    """Record this question; return a message if the session is over its limits.

    Times are kept in st.session_state, separate from the chat, so "New chat"
    doesn't reset them.
    """
    now = time.time()
    asked = st.session_state.setdefault("asked_at", [])
    if len(asked) >= config.RATE_LIMIT_PER_SESSION:
        return (f"You've reached the limit of {config.RATE_LIMIT_PER_SESSION} questions "
                "for this session. Please come back later.")
    recent = [t for t in asked if now - t < 60]
    if len(recent) >= config.RATE_LIMIT_PER_MINUTE:
        wait = int(60 - (now - recent[0])) + 1
        return f"You're asking questions quickly. Please wait {wait} seconds and try again."
    asked.append(now)
    return None


def show_message(msg: dict) -> None:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg.get("search_query"):
            st.caption(f":material/search: Searched for: {msg['search_query']}")
        show_sources(msg.get("sources", []), expanded=msg.get("expanded", False))


# --- Sidebar ---------------------------------------------------------------
doc_count = warm_up()
with st.sidebar:
    st.markdown("### :material/tune: Settings")
    choice = st.selectbox("Your catalog year", [ANY_YEAR] + rag.catalog_years(),
                          help="The catalog in effect when you entered UT. Documents "
                               "for other catalogs are left out of the search.")
    year = None if choice == ANY_YEAR else choice
    if st.button("New chat", icon=":material/add_comment:", width="stretch"):
        st.session_state.messages = []
    st.divider()
    if has_api_key():
        st.caption(f":material/check_circle: Answers on · {config.LLM_MODEL}")
    else:
        st.caption(":material/key_off: Answers off: no API key in `.env`. "
                   "Search still works.")
    st.caption(f":material/folder: {doc_count} advising documents")

# --- Main page -------------------------------------------------------------
st.markdown("## ChE Advising Assistant")
st.caption(f"{config.MAJOR}  ·  {config.UNIVERSITY}")
st.caption(f":material/info: {DISCLAIMER}")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources", ...}]

for msg in st.session_state.messages:
    show_message(msg)

# A question comes from the chat box, or from clicking an example.
question = st.chat_input("Ask about courses, prerequisites, AP credit, policies...")
if not st.session_state.messages and not question:
    st.write("")
    st.markdown("**Not sure where to start? Try one of these:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=example, width="stretch"):
            question = example

if question:
    show_message({"role": "user", "content": question})
    # Earlier turns, text only, for follow-up questions (rate-limit notices left out).
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages if not m.get("notice")]
    if limited := rate_limit_message():
        result = {"answer": limited, "sources": [], "notice": True}
    else:
        with st.spinner("Searching the advising documents..."):
            try:
                result = (rag.answer(question, history, year) if has_api_key()
                          else search_only(question, year))
            except (RuntimeError, SystemExit) as err:  # API problems: show, don't crash
                result = {"answer": f"Sorry, something went wrong: {err}", "sources": []}

    reply = {"role": "assistant", "content": result["answer"], "sources": result["sources"],
             "notice": result.get("notice", False),
             "expanded": not has_api_key(),  # in search-only mode the passages ARE the answer
             # Only show the search when a follow-up was rewritten into something new.
             "search_query": (result.get("search_query")
                              if result.get("search_query", question) != question else None)}
    show_message(reply)
    st.session_state.messages += [{"role": "user", "content": question,
                                   "notice": reply["notice"]}, reply]
    st.rerun()  # redraw so the example buttons disappear after the first question
