"""
chunking.py
─────────────────────────────────────────────────────────────────────────────
Document chunking and legal identifier extraction.

Responsibilities:
  - Score and select the most relevant excerpt per source document
  - Build a multi-document context block for the extraction prompt
  - Extract Italian/other legal identifiers from text (regex, fully mechanical)
  - Parse AFG Source column to map table references → specific laws

No Claude API calls here. To tune relevance scoring: edit only this file.
─────────────────────────────────────────────────────────────────────────────
"""

import re
from config import CHUNK_RADIUS, LEGAL_PATTERNS_BY_COUNTRY


# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE SCORING AND CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def _score_window(window_text: str, full_window: str, terms: list[str]) -> float:
    """
    Scores a text window for relevance to a fiscal measure.
    Rewards keyword matches, euro amounts, and Italian threshold language.
    """
    keyword_score   = sum(1 for t in terms if t in window_text)
    euro_score      = 2.0 * len(re.findall(
        r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:euro|EUR|%)', full_window
    ))
    threshold_score = 1.5 * len(re.findall(
        r'(?:non superiori|fino a|inferiori|oltre|redditi)', full_window, re.IGNORECASE
    ))
    return keyword_score + euro_score + threshold_score


def extract_relevant_chunk(full_text: str, measure: str,
                           radius: int = CHUNK_RADIUS) -> str:
    """
    Finds the highest-scoring window in `full_text` for the given `measure`
    and returns a labelled excerpt.
    Falls back to the document start if no strong match is found.
    """
    terms = [w.lower() for w in re.findall(r'\b\w{5,}\b', measure)]
    text_lower = full_text.lower()
    best_pos, best_score = -1, 0.0
    step = max(1, radius // 4)

    for pos in range(0, max(1, len(text_lower) - radius), step):
        score = _score_window(
            text_lower[pos: pos + radius],
            full_text[pos: pos + radius],
            terms,
        )
        if score > best_score:
            best_score, best_pos = score, pos

    if best_pos >= 0 and best_score >= 2:
        start = max(0, best_pos - radius // 2)
        end   = min(len(full_text), best_pos + int(radius * 1.5))
        return f"[RELEVANT EXCERPT — score: {best_score:.1f}]\n{full_text[start:end]}"

    return f"[DOCUMENT START — no strong match]\n{full_text[:radius * 2]}"


def _expand_chunk(chunk: str, original_text: str, score: float) -> str:
    """
    For high-scoring chunks, expands the window to ±7 500 chars
    to capture complete sentences and nearby tables.
    """
    preview = chunk[chunk.find("\n") + 1: chunk.find("\n") + 200]
    pos = original_text.find(preview[:100])
    if pos >= 0:
        start = max(0, pos - 3_500)
        end   = min(len(original_text), pos + 7_500)
        return f"[EXPANDED EXCERPT — score: {score:.1f}]\n{original_text[start:end]}"
    return chunk


def build_source_context(source_docs: dict[str, str], measure: str) -> str:
    """
    Searches ALL source documents, scores each, and returns labelled excerpts
    from the top-3 most relevant documents (most → least relevant).

    High-relevance documents (score ≥ 5) get an expanded window, ensuring
    that income thresholds split across paragraphs are captured.
    """
    scored: list[tuple[str, str, float]] = []

    for name, text in source_docs.items():
        chunk = extract_relevant_chunk(text, measure)
        score_match = re.search(r'score:\s*([\d.]+)', chunk)
        score = float(score_match.group(1)) if score_match else 0.0
        if score >= 5 and "[RELEVANT EXCERPT" in chunk:
            chunk = _expand_chunk(chunk, text, score)
        scored.append((name, chunk, score))

    scored.sort(key=lambda x: x[2], reverse=True)

    parts = [
        f"{'─' * 50}\nDOCUMENT: {name}\n{'─' * 50}\n{chunk}"
        for name, chunk, _ in scored[:3]
    ]
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LEGAL IDENTIFIER EXTRACTION  (fully mechanical regex)
# ─────────────────────────────────────────────────────────────────────────────

def extract_law_identifiers(text: str, country: str = "ITA") -> str:
    """
    Extracts Italian (or other country) legal references from text.
    Returns a comma-separated string like: "DL 90/2014, L 147/2013, Art. 5"
    """
    patterns = LEGAL_PATTERNS_BY_COUNTRY.get(country, {})
    if not patterns:
        return "NA"

    identifiers: list[str] = []

    for m in re.finditer(patterns.get("decreto_legge", r"(?!x)x"), text):
        identifiers.append(f"DL {m.group(1)}/{m.group(2)}")

    for m in re.finditer(patterns.get("decreto_legislativo", r"(?!x)x"), text):
        identifiers.append(f"D.Lgs. {m.group(1)}/{m.group(2)}")

    # Legge: skip numbers already captured as DL
    seen = {id_.split()[1] for id_ in identifiers if id_.startswith("DL")}
    for m in re.finditer(patterns.get("legge", r"(?!x)x"), text):
        ny = f"{m.group(1)}/{m.group(2)}"
        if ny not in seen:
            identifiers.append(f"L {ny}")

    for m in list(re.finditer(patterns.get("articolo", r"(?!x)x"), text))[:3]:
        identifiers.append(f"Art. {m.group(1)}")

    # Deduplicate preserving order
    seen_all: set[str] = set()
    unique: list[str] = []
    for id_ in identifiers:
        if id_ not in seen_all:
            seen_all.add(id_)
            unique.append(id_)

    return ", ".join(unique) if unique else "NA"


# ─────────────────────────────────────────────────────────────────────────────
# AFG TABLE → LAW MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def build_afg_table_law_mapping(afg_source: str) -> dict[str, str]:
    """
    Parses the AFG Source column (M) to build a dict mapping table names
    and page numbers to the law they correspond to.

    Example input:
      "DEF 2014, Table V.7 pag 91 (Legge di Stabilità 2014);
       Table A2 pag 134 (Decree Law 90/2014, Jun 2014)"

    Returns:
      {"V7": "Legge di Stabilità 2014", "91": "Legge di Stabilità 2014",
       "A2": "Decree Law 90/2014, Jun 2014", "134": "Decree Law 90/2014, Jun 2014"}
    """
    if not afg_source or afg_source == "NA":
        return {}

    mapping: dict[str, str] = {}
    # Split on semicolons/commas that are NOT inside parentheses
    entries = re.split(r'[;,](?![^()]*\))', afg_source)

    for entry in entries:
        m = re.search(
            r'Table\s*([A-Z]\.?\d+)\s*pag\s*(\d+)\s*\(([^)]+)\)',
            entry, re.IGNORECASE,
        )
        if m:
            tname = m.group(1).replace('.', '')
            page  = m.group(2)
            law   = m.group(3).strip()
            mapping[tname] = law
            mapping[page]  = law
            # Also store with period: "V7" → "V.7"
            if '.' not in tname and len(tname) >= 2:
                mapping[f"{tname[0]}.{tname[1:]}"] = law

    return mapping


def find_measure_law_in_table_pages(
    measure: str,
    source_docs: dict[str, str],
    afg_source: str,
    country: str = "ITA",
) -> str:
    """
    Searches the table pages referenced in `afg_source` for the measure text.
    If found, returns the law associated with that table from the AFG Source column.

    This is more reliable than asking Claude to identify the law from the excerpt.
    """
    if not afg_source or afg_source == "NA" or not source_docs:
        return "NA"

    # Build page → law map
    table_map: dict[int, dict] = {}
    for m in re.finditer(
        r'Table\s*([A-Z]\.?\d+)\s*pag\s*(\d+)\s*\(([^)]+)\)',
        afg_source, re.IGNORECASE,
    ):
        table_map[int(m.group(2))] = {"table": m.group(1), "law": m.group(3).strip()}

    if not table_map:
        return "NA"

    terms = [w.lower() for w in re.findall(r'\b\w{5,}\b', measure)]
    if len(terms) < 2:
        return "NA"

    for doc_text in source_docs.values():
        for page_num, info in table_map.items():
            pm = re.search(rf"\[PAGE {page_num}\]", doc_text, re.IGNORECASE)
            if pm:
                start   = max(0, pm.start() - 1_000)
                end     = min(len(doc_text), pm.end() + 3_000)
                excerpt = doc_text[start:end].lower()
                if sum(1 for t in terms if t in excerpt) >= 3:
                    return info["law"]

    return "NA"


def match_measure_to_law(
    source_page_reference: str,
    source_document_name: str,
    table_law_mapping: dict[str, str],
) -> str:
    """
    Given a page reference string and the table-law mapping, returns the
    specific law identifier. Used as a fallback when page-level search fails.
    """
    if not source_page_reference or source_page_reference == "NA":
        return "NA"
    if not table_law_mapping:
        return "NA"

    # Try by table name first
    tm = re.search(r'Table\s*([A-Z]\.?\d+)', source_page_reference, re.IGNORECASE)
    if tm:
        tname = tm.group(1).replace('.', '')
        if tname in table_law_mapping:
            return table_law_mapping[tname]
        tp = f"{tname[0]}.{tname[1:]}" if len(tname) >= 2 else tname
        if tp in table_law_mapping:
            return table_law_mapping[tp]

    # Then by page number
    pm = re.search(r'(?:Page|pag)\s*(\d+)', source_page_reference, re.IGNORECASE)
    if pm and pm.group(1) in table_law_mapping:
        return table_law_mapping[pm.group(1)]

    return "NA"