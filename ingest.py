"""Ingestion: load every document, clean it, and split it into chunks.

Pipeline: load -> clean -> chunk -> embed -> store.

A "chunk" is a passage of a few hundred words. The bot searches over chunks
instead of whole files, so it can hand the LLM just the relevant paragraphs.

Run:  python ingest.py            (build or update the database)
      python ingest.py --dry-run  (only chunk; write chunks_preview.txt)
"""

import argparse
import hashlib
import random
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import pymupdf
from bs4 import BeautifulSoup
from docx import Document

import config
import rag

# A "unit" is the smallest piece we never split if we can help it: one
# paragraph, one list, or one table. Each is a dict:
#   {"text": str, "page": int | None, "section": str, "kind": "text" | "table"}
Unit = dict

# Invisible characters that PDFs and Word sprinkle into text.
INVISIBLE = re.compile(r"[​‌‍⁠﻿­]")
BULLETS = re.compile(r"^\s*[●■•▪◦▫○◆➢➤►]\s*")
PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,3}(\s*(of|/)\s*\d{1,3})?$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Metadata from the file path
# --------------------------------------------------------------------------

def readable_name(path: Path) -> str:
    """File name without extension, with %20-style escapes turned back into text."""
    return unquote(path.stem).strip()


def doc_type_for(path: Path) -> str:
    """The subfolder under documents/ (e.g. 'syllabi'); 'general' if none."""
    rel = path.relative_to(config.RAW_DIR)
    return rel.parts[0].lower() if len(rel.parts) > 1 else "general"


TERMS = {"fall": "Fall", "fa": "Fall", "f": "Fall", "spring": "Spring",
         "sp": "Spring", "summer": "Summer", "su": "Summer"}
UT_SEMESTER_DIGIT = {"2": "Spring", "6": "Summer", "9": "Fall"}


def catalog_year_for(name: str) -> str:
    """Pull a year or term out of a file name, or return 'unknown'.

    Handles the styles in our files, checked in this order:
      'CHE3542022215430'   -> 'Spring 2022'  (UT semester code 20222 + unique #)
      'Fall2026', 'F26', 'SP 2026', 'Sp26' -> 'Fall 2026' / 'Spring 2026'
      '2026-2028', '2020-2028'            -> '2026-28'
      '26-27'                             -> '2026-27'
      a bare '2025'                       -> '2025'
    """
    m = re.match(r"^[A-Za-z]+\d{3}[A-Za-z]?(20\d\d)([269])\d{5}$", name)
    if m:
        return f"{UT_SEMESTER_DIGIT[m.group(2)]} {m.group(1)}"

    m = re.search(r"(?<![A-Za-z])(fall|fa|f|spring|sp|summer|su)\s*(20)?(\d\d)(?!\d)",
                  name, re.IGNORECASE)
    if m:
        return f"{TERMS[m.group(1).lower()]} 20{m.group(3)}"

    m = re.search(r"(?<!\d)(20\d\d)\s*[-–]\s*(?:20)?(\d\d)(?!\d)", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    m = re.search(r"(?<!\d)(\d\d)-(\d\d)(?!\d)", name)
    if m and int(m.group(2)) == int(m.group(1)) + 1:
        return f"20{m.group(1)}-{m.group(2)}"

    m = re.search(r"(?<!\d)(20\d\d)(?!\d)", name)
    if m:
        return m.group(1)
    return "unknown"


def term_from_text(text: str) -> str:
    """Fallback when the file name has no year: the first 'Fall 2026'-style
    term mentioned near the start of the document (syllabi state it up top)."""
    m = re.search(r"\b(Fall|Spring|Summer)\s*,?\s*(20\d\d)\b", text[:3000], re.IGNORECASE)
    return f"{m.group(1).title()} {m.group(2)}" if m else "unknown"


# --------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------

def clean_line(line: str) -> str:
    """Remove invisible characters, turn odd bullet symbols into '- ', squash spaces."""
    line = INVISIBLE.sub("", line)
    if BULLETS.match(line):
        line = BULLETS.sub("- ", line)
    return re.sub(r"[ \t\xa0]+", " ", line).strip()


def join_block_lines(lines: list[str]) -> str:
    """Join the lines of one PDF text block.

    PDFs break lines wherever the page edge is, so ordinary lines are glued
    back together with spaces. Bullet items keep their own line.
    """
    out: list[str] = []
    for line in lines:
        if not line:
            continue
        if out and not line.startswith("- "):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def continues_sentence(prev: str, nxt: str) -> bool:
    """True if `nxt` looks like the rest of `prev`'s sentence: prev doesn't end
    in punctuation and nxt starts with a lowercase letter."""
    return not re.search(r"[.!?:;]\s*$", prev) and bool(re.match(r"[a-z(]", nxt))


def repeated_edge_lines(pages: list[list[str]]) -> set[str]:
    """Find header/footer lines: text that shows up near the top or bottom of
    most pages. Digits are masked so 'Page 3' and 'Page 4' count as the same."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        edge = lines[:3] + lines[-3:]
        counts.update({mask_digits(l) for l in edge if len(l) >= 4})
    threshold = max(3, len(pages) // 2)
    return {line for line, n in counts.items() if n >= threshold}


def mask_digits(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower())


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

CHECKMARKS = {"✓", "✔", "\uf0fc", "x", "X"}
# Short words that are real on their own, so they are never glued onto the word before.
SHORT_WORDS = {"a", "an", "as", "at", "be", "by", "do", "if", "in", "is", "it",
               "no", "of", "on", "or", "so", "to", "up", "we", "&"}


def fix_cell(cell: str | None) -> str:
    """Clean one table cell. Narrow columns wrap text mid-word, so this glues
    the pieces back: '1646\\n0' -> '16460', 'Uniq\\nue' -> 'Unique'."""
    raw = INVISIBLE.sub("", cell or "").strip()
    if raw in CHECKMARKS:
        return "✓"
    text = clean_line(raw.replace("\n", " "))
    if re.fullmatch(r"\d{3,4} \d{1,2}", text):  # a wrapped number like a unique #
        return text.replace(" ", "")
    words = text.split(" ")
    for i in range(len(words) - 1, 0, -1):
        w = words[i]
        if len(w) <= 2 and w.isalpha() and w.islower() and w not in SHORT_WORDS:
            words[i - 1] += words.pop(i)
    return " ".join(words)


def is_header_row(row: list[str], headers: list[str]) -> bool:
    """True if at least two cells repeat their column's header text (the header
    row printed again on a new page, possibly with a new first-column name)."""
    matches = sum(1 for i, c in enumerate(row)
                  if c and i < len(headers) and headers[i] and c in headers[i])
    return matches >= 2


def nearest_header(headers: list[str], i: int) -> str:
    """Header for column i. Merged header cells often sit a column or two to
    the side of their values, so if column i has no header, borrow the closest
    one within 2 columns."""
    for dist in range(3):
        for j in (i + dist, i - dist):
            if 0 <= j < len(headers) and headers[j]:
                return headers[j]
    return ""


def table_to_text(rows: list[list[str]], state: dict) -> str:
    """Turn a table into one line per row, like 'Day: MWF | Room: ECJ 1.214'.

    - A lone cell at the top is a table title. Short titles become the label
      (see below) and mean the table has no header row.
    - Otherwise the first row is the column headers. If the next row
      only adds short bits under existing headers (e.g. 'Area' over '1', '2',
      '4A'), it is merged into them.
    - When the header row repeats later, it's skipped. If it has a new name in
      some column (e.g. 'Other engineering courses' replacing 'ChE courses'),
      that column's header is updated.
    - A row with one filled cell of 3-12 words (like 'CHE 319 TRANSPORT
      PHENOMENA') is a group label, prefixed to the rows under it so each line
      stands alone. A shorter lone cell is wrapped text and joins the line
      above; a longer one is kept as its own line.
    - A blank first cell under a merged (multi-row) cell, like 'Math' spanning
      four AP Calculus rows, is filled in from the row above.
    `state` carries the headers and label across pages of the same document.
    """
    rows = [[fix_cell(c) for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    lines: list[str] = []

    if not state.get("headers"):
        titled = False
        while rows and sum(bool(c) for c in rows[0]) < 2:  # title row(s) above the table
            title = next(c for c in rows.pop(0) if c)
            if 3 <= len(title.split()) <= 12:
                state["label"] = title
            else:
                lines.append(title)
            titled = True
        # A titled table (like 'CH 302 Requirement' over plain rows) has no header row.
        if rows and not titled:
            headers, rows = rows[0], rows[1:]
            if rows:
                second = rows[0]
                filled = [i for i, c in enumerate(second) if c]
                if (len(filled) >= 2 and all(len(second[i]) <= 4 for i in filled)
                        and all(i < len(headers) and headers[i] for i in filled)):
                    headers = [f"{h} {s}".strip() for h, s in zip(headers, second)]
                    rows = rows[1:]
            state["headers"] = headers
    headers = state.get("headers", [])
    # Column 0 can only be a merged cell if column 1 has a header of its own
    # (otherwise column 1 is just overflow from column 0, as in Tech Electives).
    fill_down = len(headers) > 1 and bool(headers[0]) and bool(headers[1])

    for row in rows:
        filled = [c for c in row if c]
        if is_header_row(row, headers):
            for i, c in enumerate(row):
                if c and i < len(headers) and c not in headers[i]:
                    headers[i] = c
            continue
        if len(filled) == 1 and len(row) > 2:
            words = len(filled[0].split())
            if 3 <= words <= 12:
                state["label"] = filled[0]
            elif words > 12 or not lines:
                lines.append(filled[0])
            else:
                lines[-1] += " " + filled[0]
            continue
        if fill_down:
            if row[0]:
                state["above"] = row[0]
            elif state.get("above"):
                row = [state["above"]] + row[1:]
        parts = []
        for i, cell in enumerate(row):
            if not cell:
                continue
            head = nearest_header(headers, i)
            parts.append(f"{head}: {cell}" if head and head != cell else cell)
        prefix = f"{state['label']} — " if state.get("label") else ""
        lines.append(prefix + " | ".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Loaders: one per file type. Each returns a list of units.
# --------------------------------------------------------------------------

def load_pdf(path: Path) -> list[Unit]:
    """Read a PDF page by page. Tables become row-per-line text; everything
    else is read as paragraph blocks in reading order."""
    doc = pymupdf.open(path)
    table_state: dict = {}
    raw_pages: list[list[tuple[float, str, str]]] = []  # (y, kind, text) per page

    for page in doc:
        items: list[tuple[float, str, str]] = []
        tables = page.find_tables().tables
        table_boxes = [pymupdf.Rect(t.bbox) for t in tables]
        for t, box in zip(tables, table_boxes):
            text = table_to_text(t.extract(), table_state)
            if text:
                items.append((box.y0, "table", text))
        # Text blocks: (x0, y0, x1, y1, text, block_no, block_type). Type 0 is text.
        for x0, y0, x1, y1, text, _, btype in page.get_text("blocks", sort=True):
            if btype != 0 or any(pymupdf.Rect(x0, y0, x1, y1).intersects(b) for b in table_boxes):
                continue
            items.append((y0, "text", text))
        items.sort(key=lambda it: it[0])
        raw_pages.append(items)

    # Header/footer detection looks at the first and last lines of each page.
    page_lines = [[clean_line(l) for _, kind, txt in items if kind == "text"
                   for l in txt.splitlines() if clean_line(l)] for items in raw_pages]
    repeated = repeated_edge_lines(page_lines)

    units: list[Unit] = []
    for page_no, items in enumerate(raw_pages, start=1):
        page_units: list[Unit] = []
        for _, kind, text in items:
            if kind == "table":
                page_units.append({"text": text, "page": page_no, "section": "", "kind": "table"})
                continue
            lines = [clean_line(l) for l in text.splitlines()]
            lines = [l for l in lines if l and mask_digits(l) not in repeated]
            joined = join_block_lines(lines)
            if not joined:
                continue
            prev = page_units[-1] if page_units else None
            if prev and prev["kind"] == "text" and continues_sentence(prev["text"], joined):
                prev["text"] += " " + joined  # PDF split one paragraph into blocks
            else:
                page_units.append({"text": joined, "page": page_no, "section": "", "kind": "text"})
        # Drop a bare page number at the very top or bottom of the page.
        for idx in (0, -1):
            if page_units and PAGE_NUMBER.match(page_units[idx]["text"]):
                page_units.pop(idx)
        units.extend(page_units)
    return units


def load_docx(path: Path) -> list[Unit]:
    """Read a Word file: paragraphs (tracking the latest heading as the section) and tables."""
    doc = Document(str(path))
    units: list[Unit] = []
    section = ""
    # doc.element.body lists paragraphs and tables in document order.
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = next(p for p in doc.paragraphs if p._p is child)
            text = clean_line(para.text)
            if not text:
                continue
            if para.style is not None and para.style.name.lower().startswith(("heading", "title")):
                section = text
            units.append({"text": text, "page": None, "section": section, "kind": "text"})
        elif tag == "tbl":
            table = next(t for t in doc.tables if t._tbl is child)
            rows = [[c.text for c in row.cells] for row in table.rows]
            text = table_to_text(rows, {})
            if text:
                units.append({"text": text, "page": None, "section": section, "kind": "table"})
    return units


# Parts of a saved Canvas page that are menus, not content.
CANVAS_JUNK = ["script", "style", "nav", "header", "footer", "noscript", "#header",
               "#left-side", "#breadcrumbs", "#footer", ".ic-app-nav-toggle-and-crumbs",
               "#right-side-wrapper", ".screenreader-only", "#flash_message_holder"]


def load_html(path: Path) -> list[Unit]:
    """Read a saved Canvas/web page, keeping only the main content area."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for selector in CANVAS_JUNK:
        for tag in soup.select(selector):
            tag.decompose()
    root = soup.select_one("#content") or soup.select_one("main") or soup.body or soup

    units: list[Unit] = []
    section = ""
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if el.find_parent("table") is not None:
            continue  # cells are handled with their table
        if el.name == "table":
            rows = [[c.get_text(" ") for c in tr.find_all(["th", "td"])] for tr in el.find_all("tr")]
            text = table_to_text(rows, {})
            if text:
                units.append({"text": text, "page": None, "section": section, "kind": "table"})
            continue
        text = clean_line(el.get_text(" "))
        if not text:
            continue
        if el.name.startswith("h"):
            section = text
        if el.name == "li":
            text = "- " + text
        units.append({"text": text, "page": None, "section": section, "kind": "text"})
    return units


def load_text(path: Path) -> list[Unit]:
    """Read .txt/.md: blank lines separate paragraphs; '#' lines are headings.
    A heading is kept in the same unit as the paragraph after it, so a chunk
    never ends up with a list but not the heading saying what the list is."""
    units: list[Unit] = []
    section = ""
    pending_heading = ""
    for para in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="ignore")):
        lines = [clean_line(l) for l in para.splitlines() if clean_line(l)]
        if not lines:
            continue
        if lines[0].startswith("#"):
            section = lines[0].lstrip("#").strip()
            if len(lines) == 1:  # heading on its own: attach it to what follows
                pending_heading = section
                continue
        text = "\n".join(lines)
        if pending_heading:
            text, pending_heading = f"{pending_heading}\n{text}", ""
        units.append({"text": text, "page": None, "section": section, "kind": "text"})
    if pending_heading:
        units.append({"text": pending_heading, "page": None, "section": section, "kind": "text"})
    return units


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".html": load_html,
           ".htm": load_html, ".txt": load_text, ".md": load_text}


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def split_big_unit(unit: Unit, size: int) -> list[Unit]:
    """Split a unit longer than one chunk. Tables and lists split between lines;
    plain paragraphs split between sentences."""
    if "\n" in unit["text"]:
        pieces = unit["text"].split("\n")
    else:
        pieces = re.split(r"(?<=[.!?])\s+", unit["text"])
    out: list[Unit] = []
    current: list[str] = []
    for piece in pieces:
        if current and word_count(" ".join(current + [piece])) > size:
            out.append({**unit, "text": "\n".join(current)})
            current = []
        current.append(piece)
    if current:
        out.append({**unit, "text": "\n".join(current)})
    return out


def location(units: list[Unit]) -> tuple[str, str]:
    """Page range (e.g. '3' or '3-4') and section name covered by these units."""
    pages = [u["page"] for u in units if u["page"] is not None]
    page = "" if not pages else (str(pages[0]) if pages[0] == pages[-1] else f"{pages[0]}-{pages[-1]}")
    sections = [u["section"] for u in units if u["section"]]
    return page, (sections[0] if sections else "")


def chunk_units(units: list[Unit], size: int, overlap: int) -> list[dict]:
    """Pack units into chunks of about `size` words.

    Units are added whole until the next one would overflow. The next chunk
    then starts with the last unit(s) of the previous one (up to `overlap`
    words) so a sentence near a boundary is never stranded without context.
    """
    pieces: list[Unit] = []
    for u in units:
        pieces.extend(split_big_unit(u, size) if word_count(u["text"]) > size else [u])

    groups: list[list[Unit]] = []  # the units that make up each chunk
    current: list[Unit] = []
    carried = 0  # how many units at the start of `current` are overlap
    for piece in pieces:
        new_words = sum(word_count(u["text"]) for u in current[carried:])
        too_full = sum(word_count(u["text"]) for u in current) + word_count(piece["text"]) > size
        # A new heading (docx/html/md files) starts a new chunk, so each chunk
        # covers one topic, e.g. one semester of the degree plan.
        new_section = bool(current) and piece["section"] != current[-1]["section"]
        # Close the chunk when it's full or a section ends, unless it's still
        # tiny (e.g. a title right before a big table); then it absorbs the
        # next piece instead. A finished section may be half the usual minimum.
        big_enough = new_words >= (config.MIN_CHUNK_WORDS // 2 if new_section else config.MIN_CHUNK_WORDS)
        if (too_full or new_section) and big_enough:
            groups.append(current)
            # Carry trailing units forward as overlap, as long as they fit.
            carry: list[Unit] = []
            for u in reversed(current):
                if sum(word_count(c["text"]) for c in carry) + word_count(u["text"]) > overlap:
                    break
                carry.insert(0, u)
            if not carry and current[-1]["kind"] == "text":
                # Last paragraph is too long to carry whole: carry its tail.
                tail = " ".join(current[-1]["text"].split()[-overlap:])
                carry = [{**current[-1], "text": "..." + tail}]
            if new_section:
                carry = []  # no overlap across a section boundary
            current, carried = carry, len(carry)
        current.append(piece)

    new_units = current[carried:]
    if new_units:
        # A short leftover at the end of a file is folded into the previous chunk
        # instead of becoming a tiny chunk of its own.
        if groups and sum(word_count(u["text"]) for u in new_units) < config.MIN_CHUNK_WORDS:
            groups[-1].extend(new_units)
        else:
            groups.append(current)
    return [make_chunk(g) for g in groups]


def make_chunk(units: list[Unit]) -> dict:
    page, section = location(units)
    return {"text": "\n\n".join(u["text"] for u in units), "page": page, "section": section}


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------

def find_files() -> list[Path]:
    """Every file under the documents folder, skipping hidden files like .DS_Store."""
    if not config.RAW_DIR.exists():
        raise SystemExit(f"Document folder not found: {config.RAW_DIR}")
    return sorted(p for p in config.RAW_DIR.rglob("*")
                  if p.is_file() and not p.name.startswith("."))


def process_file(path: Path) -> list[dict]:
    """Load, clean and chunk one file. Returns chunks with full metadata."""
    units = LOADERS[path.suffix.lower()](path)
    chunks = chunk_units(units, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
    year = catalog_year_for(readable_name(path))
    if year == "unknown" and doc_type_for(path) == "syllabi":
        year = term_from_text("\n".join(u["text"] for u in units[:20]))
    meta = {
        "source": str(path.relative_to(config.RAW_DIR)),
        "title": readable_name(path),
        "doc_type": doc_type_for(path),
        "catalog_year": year,
    }
    return [{**meta, **c, "chunk_index": i,
             "courses": " ".join(sorted(rag.course_keys(f"{meta['title']} {c['text']}")))}
            for i, c in enumerate(chunks)]


def supported_files() -> list[Path]:
    """Files we have a loader for; anything else is reported and skipped."""
    files = []
    for path in find_files():
        if str(path.relative_to(config.RAW_DIR)) in config.SKIP_FILES:
            continue
        if path.suffix.lower() in LOADERS:
            files.append(path)
        else:
            print(f"  skipped (unsupported file type): {path.relative_to(config.RAW_DIR)}")
    return files


def process_file_safely(path: Path) -> list[dict]:
    """process_file(), but a corrupt or empty file gives a warning, not a crash."""
    try:
        chunks = process_file(path)
    except Exception as err:
        print(f"  WARNING: could not read {path.relative_to(config.RAW_DIR)}: {err}")
        return []
    if not chunks:
        print(f"  WARNING: no text found in {path.relative_to(config.RAW_DIR)}")
    return chunks


def load_all() -> list[dict]:
    """Load and chunk every supported file."""
    return [c for path in supported_files() for c in process_file_safely(path)]


# --------------------------------------------------------------------------
# Embed and store (Phase 2)
# --------------------------------------------------------------------------

# Bump this when ingest.py changes how chunks are made, so every file is
# re-processed on the next run even though the files themselves didn't change.
PIPELINE_VERSION = "4"


def file_hash(path: Path) -> str:
    """Fingerprint of a file plus the settings used to chunk and embed it.
    It changes when the file changes, or when chunking settings, the
    embedding model, or PIPELINE_VERSION change."""
    settings = (f"{PIPELINE_VERSION}|{config.CHUNK_SIZE_WORDS}|{config.CHUNK_OVERLAP_WORDS}|"
                f"{config.MIN_CHUNK_WORDS}|{config.EMBEDDING_MODEL}")
    return hashlib.sha256(path.read_bytes() + settings.encode()).hexdigest()


def stored_hashes(collection) -> dict[str, str]:
    """{source: file_hash} for every file already in the database."""
    metas = collection.get(include=["metadatas"])["metadatas"]
    return {m["source"]: m["file_hash"] for m in metas}


def embed_and_store() -> None:
    """Bring the database in line with the documents folder.

    Only new or changed files are chunked and embedded (that's the slow part);
    chunks belonging to deleted files are removed.
    """
    collection = rag.get_collection()
    already = stored_hashes(collection)
    files = supported_files()
    current = {str(p.relative_to(config.RAW_DIR)) for p in files}

    for source in sorted(set(already) - current):
        collection.delete(where={"source": source})
        print(f"  removed (deleted or in SKIP_FILES): {unquote(source)}")

    added = unchanged = 0
    for path in files:
        source, digest = str(path.relative_to(config.RAW_DIR)), file_hash(path)
        if already.get(source) == digest:
            unchanged += 1
            continue
        chunks = process_file_safely(path)
        collection.delete(where={"source": source})  # drop the old version, if any
        if not chunks:
            continue
        # The title goes into the embedded text so that, e.g., "CHE 360 syllabus"
        # matches chunks from that file even when the chunk never says "CHE 360".
        vectors = rag.embed_passages([f"{c['title']}\n\n{c['text']}" for c in chunks])
        collection.add(
            ids=[f"{source}::{c['chunk_index']}" for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=vectors,
            metadatas=[{k: v for k, v in c.items() if k != "text"} | {"file_hash": digest}
                       for c in chunks],
        )
        added += 1
        print(f"  {'updated' if source in already else 'added'}: {unquote(source)} ({len(chunks)} chunks)")

    print(f"\nDone. {added} file(s) embedded, {unchanged} unchanged. "
          f"Database now holds {collection.count()} chunks.")


def dry_run(chunks: list[dict]) -> None:
    """Print a summary and write 10 random chunks to chunks_preview.txt."""
    per_file = Counter(c["source"] for c in chunks)
    print(f"\nFiles read: {len(per_file)}   Chunks: {len(chunks)}   "
          f"Average chunk length: {sum(word_count(c['text']) for c in chunks) / max(len(chunks), 1):.0f} words\n")
    print(f"{'chunks':>6}  {'year':<12} file")
    for c in sorted({c["source"]: c for c in chunks}.values(), key=lambda c: c["source"]):
        print(f"{per_file[c['source']]:>6}  {c['catalog_year']:<12} {unquote(c['source'])}")

    sample = random.sample(chunks, min(10, len(chunks)))
    out = config.PROJECT_DIR / "chunks_preview.txt"
    with out.open("w", encoding="utf-8") as f:
        for c in sample:
            f.write("=" * 78 + "\n")
            f.write(f"source: {unquote(c['source'])}\ndoc_type: {c['doc_type']}   "
                    f"catalog_year: {c['catalog_year']}   page: {c['page'] or '-'}   "
                    f"section: {c['section'] or '-'}   words: {word_count(c['text'])}\n")
            f.write("-" * 78 + "\n" + c["text"] + "\n\n")
    print(f"\nWrote {len(sample)} random chunks to {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load, chunk, embed and store the documents.")
    parser.add_argument("--dry-run", action="store_true",
                        help="only load and chunk; print a summary and write chunks_preview.txt")
    args = parser.parse_args()

    print(f"Reading documents from {config.RAW_DIR} ...")
    chunks = load_all()
    if args.dry_run:
        dry_run(chunks)
    else:
        embed_and_store()


if __name__ == "__main__":
    main()
