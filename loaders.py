"""
loaders.py
─────────────────────────────────────────────────────────────────────────────
All data loading: AFG measures, source PDFs/DOCX, Eurostat dictionaries,
and the hand-coded example/seed workbook.

The example and seed are the SAME file (e.g. Italy_2013_2014.xlsx) serving
two different purposes:
  example_as_prompt() → str   fed to Sonnet as a few-shot reference
  example_as_seed()   → dict  used by validation.py to benchmark accuracy

Nothing in this module calls the Claude API.
─────────────────────────────────────────────────────────────────────────────
"""

import re
import time
import pandas as pd
import pdfplumber
from pathlib import Path

from config import (
    AFG_INPUT_DIR, AFG_FILENAME, AFG_SHEET,
    SOURCES_DIR, MICRODATA_DIR, EXAMPLE_DIR, SEED_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# AFG MEASURES
# ─────────────────────────────────────────────────────────────────────────────

def load_afg_measures(country: str, year: str) -> pd.DataFrame:
    afg_path = AFG_INPUT_DIR / AFG_FILENAME
    if not afg_path.exists():
        afg_path = AFG_INPUT_DIR / AFG_FILENAME.replace(".xlsx", ".xls")
    if not afg_path.exists():
        raise FileNotFoundError(f"AFG file not found: {AFG_FILENAME}")

    print(f"  Reading: {afg_path.name}  sheet='{AFG_SHEET}'")
    df = pd.read_excel(afg_path, sheet_name=AFG_SHEET, header=0, dtype=str)

    col_names = list(df.columns)
    df.rename(
        columns={col_names[i]: letter
                 for i, letter in enumerate("ABCDEFGHIJKLM")
                 if i < len(col_names)},
        inplace=True,
    )

    df["A"] = df["A"].astype(str).str.strip().str.upper()
    df["B"] = df["B"].astype(str).str.strip()

    filtered = df[
        (df["A"] == country.upper()) & (df["B"] == str(year))
    ].copy()

    filtered = filtered[
        filtered["E"].notna()
        & (filtered["E"].str.strip() != "")
        & (filtered["E"].str.lower() != "nan")
    ].reset_index(drop=True)

    print(f"  Found {len(filtered)} measures for {country} {year}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# PDF / DOCX READERS
# ─────────────────────────────────────────────────────────────────────────────

def read_pdf(path: Path, max_pages: int = 120) -> str:
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                text = page.extract_text()
                if text:
                    pages.append(f"[PAGE {i + 1}]\n{text}")
    except Exception as e:
        print(f"    WARNING: cannot read {path.name}: {e}")
    return "\n\n".join(pages)


def read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in (".docx", ".doc"):
        return read_docx(path)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────

def find_source_documents(country: str, year: str) -> dict[str, str]:
    prefix = f"{country.upper()}_{year}_"
    docs: dict[str, str] = {}

    for f in sorted(SOURCES_DIR.iterdir()):
        if (
            f.name.upper().startswith(prefix)
            and f.suffix.lower() in (".pdf", ".docx", ".doc")
        ):
            print(f"    Loading: {f.name}")
            text = load_document(f)
            if text.strip():
                docs[f.name] = text
                wc = len(text.split())
                has_nums = bool(re.findall(r'\d+[.,]\d+\s*euro', text[:10_000]))
                print(
                    f"      -> {len(text):,} chars  ~{wc:,} words  "
                    f"{'OK' if has_nums else 'no'} euro amounts"
                )

    if not docs:
        print(f"    WARNING: No source documents with prefix '{prefix}'")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# EUROSTAT DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────

_KEY_COLS = [
    "Variable identifier", "Variable name",
    "Codes", "Labels", "Filter", "Topic", "Detailed topic",
]


def load_eurostat_dictionaries() -> str:
    cache_file = MICRODATA_DIR / "_dictionary_cache.txt"

    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86_400
        if age_days < 30:
            print(f"    Using cached dictionaries ({age_days:.0f} days old)")
            return cache_file.read_text(encoding="utf-8")

    combined: list[str] = []
    for f in sorted(MICRODATA_DIR.iterdir()):
        if f.suffix.lower() in (".xlsx", ".xls") and "_Dictionary_Variables" in f.name:
            print(f"    Loading: {f.name}")
            try:
                xl = pd.ExcelFile(f)
                for sheet in xl.sheet_names:
                    df = pd.read_excel(f, sheet_name=sheet, dtype=str)
                    cols = [c for c in _KEY_COLS if c in df.columns]
                    if "Variable identifier" not in cols:
                        continue
                    df = df[cols].dropna(subset=["Variable identifier"])
                    survey_name = f.stem.replace("_Dictionary_Variables", "")
                    combined.append(
                        f"\n{'=' * 60}\n"
                        f"SURVEY: {survey_name}  |  Sheet: {sheet}  |  Variables: {len(df)}\n"
                        f"{'=' * 60}\n{df.to_csv(index=False)}"
                    )
            except Exception as e:
                print(f"    WARNING: could not load {f.name}: {e}")

    full_text = "\n".join(combined)
    cache_file.write_text(full_text, encoding="utf-8")
    print(f"    Cached: {cache_file.name} ({len(full_text):,} chars)")
    return full_text


def invalidate_dictionary_cache() -> None:
    cache_file = MICRODATA_DIR / "_dictionary_cache.txt"
    if cache_file.exists():
        cache_file.unlink()
        print("    Dictionary cache cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE / SEED  --  same file, one read, two derived representations
#
#   load_example_sheet()  ->  (pd.DataFrame, sheet_name)
#   example_as_prompt()   ->  str    few-shot text injected into Sonnet prompt
#   example_as_seed()     ->  dict   ground-truth dict for validation.py
#
# Your workbook (e.g. Italy_2013_2014.xlsx) has one sheet per year ("2013",
# "2014"). load_example_sheet() picks the right sheet automatically.
# ─────────────────────────────────────────────────────────────────────────────

def _pick_sheet(xl: pd.ExcelFile, country: str, year: str) -> str:
    """Priority: exact year -> Country_Year -> contains year -> first sheet."""
    names = xl.sheet_names
    if str(year) in names:
        return str(year)
    candidate = f"{country.upper()}_{year}"
    if candidate in names:
        return candidate
    for s in names:
        if str(year) in s:
            return s
    print(f"    WARNING: no sheet matching year '{year}' -- falling back to '{names[0]}'")
    return names[0]


def load_example_sheet(country: str, year: str) -> tuple[pd.DataFrame, str]:
    """
    Finds the hand-coded workbook in SEED_DIR, picks the right sheet,
    and returns (DataFrame, sheet_name). File is read once; both
    example_as_prompt() and example_as_seed() work from the result.
    Returns (empty DataFrame, "") if no file is found.
    """
    if not SEED_DIR.exists():
        print(f"    WARNING: SEED_DIR not found: {SEED_DIR}")
        return pd.DataFrame(), ""

    for f in sorted(SEED_DIR.iterdir()):
        if f.suffix.lower() not in (".xlsx", ".xls"):
            continue
        try:
            xl = pd.ExcelFile(f)
            sheet = _pick_sheet(xl, country, year)
            df = pd.read_excel(f, sheet_name=sheet, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            print(f"    Example/Seed: {f.name}  ->  sheet '{sheet}'")
            return df, sheet
        except Exception as e:
            print(f"    WARNING loading {f.name}: {e}")

    print("    WARNING: no example/seed file found in SEED_DIR")
    return pd.DataFrame(), ""


def example_as_prompt(df: pd.DataFrame, sheet: str) -> str:
    """
    Formats the DataFrame as a string for the Sonnet few-shot prompt.
    Shows all columns and up to 30 rows so Claude can calibrate its output.
    """
    if df.empty:
        return "No example found."
    return (
        f"Sheet: {sheet}\n"
        f"Columns: {list(df.columns)}\n"
        f"{df.to_string(max_rows=30)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEED COLUMN STRUCTURE
#
# Your workbook columns (from Italy_2013_2014.xlsx):
#
#   "Motivation "           -- text label, e.g. "Redistribution"
#   "Motivation Specific"   -- e.g. "Social support / Housing emergency"
#   "Target Income - HY010" -- free text, e.g. "Income <13,000 euros/year"
#   "Target Age - RB081"    -- free text, e.g. "18-67"
#   ... (one column per EU-SILC variable)
#
# The pipeline outputs structured numeric fields (target_income_max=15000).
# Validation therefore compares:
#   - motivation_category: seed text label -> int via MOTIVATION_LABEL_TO_INT
#   - target_population_found: True if ANY Target column is non-empty
#   - Per-target presence: was the right field filled by the pipeline?
#
# To add a new target column to your workbook: add it to _SEED_TARGET_COLS.
# To handle a new motivation label: add it to MOTIVATION_LABEL_TO_INT.
# ─────────────────────────────────────────────────────────────────────────────

# Motivation text labels -> integer 1-7 taxonomy code
# These match all unique values found in Italy_2013_2014.xlsx (both sheets)
MOTIVATION_LABEL_TO_INT: dict[str, int] = {
    "Fiscal consolidation":      1,
    "Economic expansion":        2,
    "Redistribution":            3,
    "Public investment":         4,
    "Administrative/Regulatory": 5,
    "Foreign policy":            6,
    "Other":                     7,
}

# (logical_name, [accepted column variants]) for each target field
# logical_name is used as the key in the seed dict and by validation.py
_SEED_TARGET_COLS: list[tuple[str, list[str]]] = [
    ("target_income",     ["Target Income - HY010",                         "Target Income"]),
    ("target_fin_income", ["Target Financial Income - HY090"]),
    ("target_age",        ["Target Age - RB081",                            "Target Age"]),
    ("target_labour",     ["Target Labour Market Participation - PL032"]),
    ("target_employment", ["Target Employment Status - PL040A",
                           "Target Employment Status"]),
    ("target_sector",     ["Target Occupation Sector - PL111A",             "Target Occupation Sector"]),
    ("target_tenure",     ["Target Tenure status - HH021",                  "Target Tenure status"]),
    ("target_rental",     ["Target Rental Income from Property/Land HY040"]),
    ("target_secondary",  ["Target With Secondary Property"]),
    ("target_region",     ["Target Region DB040",                           "Target Region"]),
    ("target_urban",      ["Target Urban/Rural Status - DB100",             "Target Urban/Rural Status"]),
]


def _find_col(cols: list[str], variants: list[str]) -> str | None:
    """Returns the first variant present in cols, or None."""
    for v in variants:
        if v in cols:
            return v
    return None


def _is_filled(val) -> bool:
    return val is not None and str(val).strip() not in ("", "nan", "NA", "N/A", "na", "None", "NaN")


def example_as_seed(df: pd.DataFrame) -> dict[str, dict]:
    """
    Converts the example DataFrame into a seed dict keyed by normalised
    (stripped, lowercase) measure text.

    Reads your workbook column structure directly -- no reformatting needed.
    The "Motivation " text label is converted to integer via
    MOTIVATION_LABEL_TO_INT. Target column text is stored as-is alongside
    a boolean target_population_found flag.

    Returns {} if the Measure column cannot be found.
    """
    if df.empty:
        return {}

    cols = list(df.columns)

    # Measure column (required)
    measure_col = _find_col(cols, ["Measure", "measure", "MEASURE"])
    if not measure_col:
        print(f"    [Seed] Cannot find Measure column. Columns: {cols}")
        return {}

    # Motivation columns (needed for accuracy scoring)
    mot_label_col    = _find_col(cols, ["Motivation", "motivation"])
    mot_specific_col = _find_col(cols, ["Motivation Specific", "Motivation specific", "motivation_specific"])

    if not mot_label_col:
        print(f"    [Seed] Cannot find Motivation column -- motivation accuracy will not be computed.")

    # Resolve target columns
    resolved_targets: list[tuple[str, str]] = []
    for logical, variants in _SEED_TARGET_COLS:
        actual = _find_col(cols, variants)
        if actual:
            resolved_targets.append((logical, actual))

    seed: dict[str, dict] = {}

    for _, row in df.iterrows():
        key = str(row.get(measure_col, "")).strip().lower()
        if not key or key in ("nan", "none", ""):
            continue

        # Motivation label -> integer
        mot_label = str(row[mot_label_col]).strip() if mot_label_col else ""
        mot_int   = MOTIVATION_LABEL_TO_INT.get(mot_label)
        if mot_int is None and mot_label:
            for label, code in MOTIVATION_LABEL_TO_INT.items():
                if label.lower() in mot_label.lower():
                    mot_int = code
                    break

        mot_specific = str(row[mot_specific_col]).strip() if mot_specific_col else ""

        # Target columns: raw text + presence flag
        target_values: dict[str, str] = {}
        for logical, actual_col in resolved_targets:
            val = row.get(actual_col, "")
            target_values[logical] = str(val).strip() if _is_filled(val) else ""

        target_population_found = any(
            _is_filled(row.get(actual_col, ""))
            for _, actual_col in resolved_targets
        )

        seed[key] = {
            "Measure":                 row[measure_col],
            "motivation_category":     mot_int,
            "motivation_label":        mot_label,
            "motivation_specific":     mot_specific,
            "target_population_found": target_population_found,
            **target_values,
        }

    n_targeted = sum(1 for v in seed.values() if v["target_population_found"])
    n_mot_miss = sum(1 for v in seed.values() if v["motivation_category"] is None)

    print(
        f"    [Seed] {len(seed)} measures  "
        f"({n_targeted} with targeting,  "
        f"{len(seed) - n_mot_miss}/{len(seed)} motivation resolved)"
    )
    if n_mot_miss:
        bad = {v["motivation_label"] for v in seed.values() if v["motivation_category"] is None}
        print(f"           Unresolved labels: {bad}")
        print(f"           Add them to MOTIVATION_LABEL_TO_INT in loaders.py")

    return seed