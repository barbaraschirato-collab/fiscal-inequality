"""
config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for all configuration, constants, and taxonomies.

To update models, paths, or the motivation taxonomy:  edit ONLY this file.
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (override via environment variable FISCAL_BASE_DIR if needed)
# ─────────────────────────────────────────────────────────────────────────────
import os

BASE = Path(
    os.environ.get(
        "FISCAL_BASE_DIR",
        r""
        r"",
    )
)

AFG_INPUT_DIR = BASE / "Excel Input AFG"
SOURCES_DIR   = BASE / "Sources"
MICRODATA_DIR = BASE / "Microdata"
EXAMPLE_DIR   = BASE / "Excel Example"
OUTPUT_DIR    = BASE / "Excel Output"
SEED_DIR      = EXAMPLE_DIR            # seed files live alongside the example workbooks

AFG_FILENAME  = "AppendixTables_AFG.xlsx"
AFG_SHEET     = "Input"

# ─────────────────────────────────────────────────────────────────────────────
# MODELS  ← update here when Anthropic releases new versions
# ─────────────────────────────────────────────────────────────────────────────
MODEL_MAIN    = "claude-sonnet-4-6"          # document extraction
MODEL_MAPPING = "claude-haiku-4-5-20251001"  # Eurostat variable mapping

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_RADIUS        = 4500    # chars for initial keyword window
SOURCE_CACHE_LIMIT  = 40_000  # chars of source text sent to cached prefix
CHECKPOINT_EVERY    = 5       # save checkpoint every N measures
API_SLEEP_SECONDS   = 0.8     # polite pause between API calls

# ─────────────────────────────────────────────────────────────────────────────
# MOTIVATION TAXONOMY  (single definition — injected into prompts by extraction.py)
# ─────────────────────────────────────────────────────────────────────────────
MOTIVATION_TAXONOMY: dict[int, tuple[str, str]] = {
    1: ("Fiscal consolidation",      "reduce deficit/debt (revenue increases, spending cuts)"),
    2: ("Economic expansion",        "boost growth/employment (tax cuts, incentives, stimulus)"),
    3: ("Redistribution",            "help poor/vulnerable, reduce inequality"),
    4: ("Public investment",         "infrastructure/human capital (education, health, R&D)"),
    5: ("Administrative/Regulatory", "efficiency, governance, simplification"),
    6: ("Foreign policy",            "international commitments, peacekeeping"),
    7: ("Other",                     "miscellaneous, unclassifiable, one-off events"),
}

# Human-readable string injected into prompts
MOTIVATION_CATEGORIES_TEXT: str = "\n".join(
    f"{k}. {label}  — {desc}"
    for k, (label, desc) in MOTIVATION_TAXONOMY.items()
) + "\nReturn ONLY the integer 1-7 based on the STATED motivation."

# Default motivation when category/subcategory is known but text is absent
CATEGORY_DEFAULTS: dict[str, tuple[str, int]] = {
    "Government consumption": ("Government spending / Public services", 1),
    "Government investment":  ("Public investment / Infrastructure",    4),
    "Transfers":              ("Social support / Welfare spending",      3),
    "Direct taxes":           ("Tax revenue / Direct taxation",         1),
    "Indirect taxes":         ("Tax revenue / Indirect taxation",       1),
    "Income taxes":           ("Tax revenue / Income taxation",         1),
    "Corporate taxes":        ("Tax revenue / Corporate taxation",      1),
    "VAT":                    ("Tax revenue / Value added tax",         1),
}

# ─────────────────────────────────────────────────────────────────────────────
# AFG COLUMN NAMES  (letters A–M → descriptive names)
# ─────────────────────────────────────────────────────────────────────────────
AFG_COLUMN_NAMES: dict[str, str] = {
    "A": "Country",    "B": "Year",      "C": "Category",   "D": "Components",
    "E": "Measure",    "F": "Type",      "G": "Impact t",   "H": "Impact t+1",
    "I": "Impact t+2", "J": "Impact t+3","K": "Impact t+4", "L": "Impact t+5",
    "M": "Source",
}

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT COLUMN GROUPS  (order preserved in Excel)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_POPULATION_COLS = [
    "target_population_found", "target_population_summary_en",
    "target_age_min", "target_age_max",
    "target_income_min", "target_income_max", "target_income_type",
    "target_employment_status", "target_household_type",
    "target_sector", "target_geographic", "target_other_criteria",
]
EUROSTAT_MAPPING_COLS = [
    "eurostat_mapping_found", "primary_survey",
    "eurostat_variables", "secondary_surveys", "filter_conditions_combined",
]
MOTIVATION_COLS = ["motivation_specific", "motivation_category"]
SOURCE_REFERENCE_COLS = [
    "source_document_found", "source_document_name",
    "source_page_reference", "source_excerpt_original", "source_law_identifier",
]
EXTRACTION_DIAGNOSTIC_COLS = [
    "confidence_score", "confidence_grade", "confidence_signals",
    "extraction_confidence_raw", "extraction_notes",
]

# ─────────────────────────────────────────────────────────────────────────────
# LEGAL REFERENCE PATTERNS  (keyed by country ISO-3)
# Add new countries here; extraction.py picks the right one automatically.
# ─────────────────────────────────────────────────────────────────────────────
LEGAL_PATTERNS_BY_COUNTRY: dict[str, dict[str, str]] = {
    "ITA": {
        "decreto_legge":       r"[Dd](?:ecreto\s*)?[Ll](?:egge|\.)\s*(?:n\.|n°|n)?\s*(\d+)\s*/\s*(\d{4})",
        "decreto_legislativo": r"[Dd](?:ecreto\s*)?[Ll](?:egislativo|gs\.?)\s*(?:n\.|n°|n)?\s*(\d+)\s*/\s*(\d{4})",
        "legge":               r"[Ll](?:egge|\.)\s*(?:n\.|n°|n)?\s*(\d+)\s*/\s*(\d{4})",
        "articolo":            r"[Aa]rt(?:icolo|\.)?\s*(\d+(?:[-,]\w+)?)",
    },
    # ── extend for other countries ──────────────────────────────────────────
    # "FRA": {
    #     "loi":        r"[Ll]oi\s+n°\s*(\d{4})-(\d+)",
    #     "ordonnance": r"[Oo]rdonnance\s+n°\s*(\d{4})-(\d+)",
    #     "article":    r"[Aa]rticle\s+(\d+(?:[- ]\w+)?)",
    # },
    # "DEU": {
    #     "gesetz":     r"[Gg]esetz\s+(?:vom\s+)?(\d{1,2})\.\s*(\w+)\s+(\d{4})",
    #     "artikel":    r"[Aa]rt(?:ikel|\.)?\s+(\d+(?:[a-z])?)",
    # },
    # "ESP": {
    #     "ley":        r"[Ll]ey\s+(\d+)/(\d{4})",
    #     "real_decreto": r"[Rr]eal\s+[Dd]ecreto\s+(\d+)/(\d{4})",
    #     "articulo":   r"[Aa]rt(?:ículo|\.)?\s+(\d+(?:[.- ]\w+)?)",
    # },
}

# ─────────────────────────────────────────────────────────────────────────────
# REGIONAL (NUTS-2) MAPPINGS  (keyed by country ISO-3)
# ─────────────────────────────────────────────────────────────────────────────
REGIONAL_MAPPINGS: dict[str, dict[str, str]] = {
    "ITA": {
        "piemonte": "ITC1",       "piedmont": "ITC1",
        "valle d'aosta": "ITC2",  "aosta valley": "ITC2",
        "liguria": "ITC3",
        "lombardia": "ITC4",      "lombardy": "ITC4",
        "milano": "ITC4",         "milan": "ITC4",
        "bolzano": "ITH1",        "south tyrol": "ITH1",
        "trento": "ITH2",         "trentino": "ITH2",
        "veneto": "ITH3",         "venezia": "ITH3",
        "friuli venezia giulia": "ITH4",
        "emilia romagna": "ITH5", "emilia-romagna": "ITH5",
        "toscana": "ITI1",        "tuscany": "ITI1",
        "umbria": "ITI2",         "marche": "ITI3",
        "lazio": "ITI4",          "roma": "ITI4",
        "abruzzo": "ITF1",        "molise": "ITF2",
        "campania": "ITF3",       "napoli": "ITF3",
        "puglia": "ITF4",         "apulia": "ITF4",
        "basilicata": "ITF5",     "calabria": "ITF6",
        "sicilia": "ITG1",        "sicily": "ITG1",
        "sardegna": "ITG2",       "sardinia": "ITG2",
    },
    # "FRA": { "île-de-france": "FR10", "bretagne": "FRH0", ... },
    # "DEU": { "bayern": "DE2", "nordrhein-westfalen": "DEA", ... },
}

# ─────────────────────────────────────────────────────────────────────────────
# MECHANICAL CONFIDENCE WEIGHTS
# Keys match signal names in validation.compute_mechanical_confidence()
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "source_found":        12,
    "law_found":           12,
    "excerpt_found":        8,
    "target_found":        15,
    "specific_criterion":  18,
    "income_numeric":      15,
    "eurostat_mapped":     10,
    "motivation_explicit":  6,
    "claude_confidence":    4,
}  # sum = 100

# ─────────────────────────────────────────────────────────────────────────────
# QUALITY SCORE WEIGHTS  (used in output.py)
# ─────────────────────────────────────────────────────────────────────────────
QUALITY_SCORE_WEIGHTS: dict[str, float] = {
    "target_coverage":     0.25,
    "income_detail":       0.25,
    "eurostat_mapping":    0.20,
    "motivation_explicit": 0.12,
    "source_coverage":     0.10,
    "high_conf_share":     0.08,
}  # sum = 1.00
