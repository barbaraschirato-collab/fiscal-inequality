"""
output.py
─────────────────────────────────────────────────────────────────────────────
File export, checkpoint management, and quality/cost report.

  save_checkpoint()        - JSON snapshot every N measures (resume support)
  load_latest_checkpoint() - loads the most recent snapshot
  export_to_excel()        - multi-year Excel: Dictionary | Data | per-year
                             sheets | Seed Validation
  print_quality_report()   - token costs, coverage, weighted quality score

export_to_excel() accepts results_by_year = {"2013": [...], "2014": [...]}
so a single file ITA_2013_2014_Extended.xlsx is produced with:
  - a combined Data sheet across all years
  - one sheet per year ("2013", "2014")
  - a Seed Validation sheet covering all years that had seed data
─────────────────────────────────────────────────────────────────────────────
"""

import json
import time as tm
import pandas as pd
from pathlib import Path

from config import (
    OUTPUT_DIR,
    AFG_COLUMN_NAMES,
    MOTIVATION_TAXONOMY,
    QUALITY_SCORE_WEIGHTS,
    TARGET_POPULATION_COLS,
    EUROSTAT_MAPPING_COLS,
    MOTIVATION_COLS,
    SOURCE_REFERENCE_COLS,
    EXTRACTION_DIAGNOSTIC_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(results: list[dict], country: str, year: str, step: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{country}_{year}_checkpoint_{step:04d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"    checkpoint saved ({step} measures): {path.name}")


def load_latest_checkpoint(country: str, year: str) -> tuple[list[dict], int]:
    checkpoints = sorted(OUTPUT_DIR.glob(f"{country}_{year}_checkpoint_*.json"))
    if not checkpoints:
        print("  No checkpoint found - starting from scratch.")
        return [], 0
    latest = checkpoints[-1]
    print(f"  Resuming from checkpoint: {latest.name}")
    with open(latest, encoding="utf-8") as f:
        results = json.load(f)
    n = len(results)
    print(f"  {n} measures already completed.")
    return results, n


# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_VARIABLE_LABELS = {
    "RB081": "Age", "RB090": "Sex",
    "RB211": "Main activity status (self-defined)",
    "PL031": "Self-defined current economic status",
    "PL111": "Industry (NACE Rev. 2)",
    "HY020": "Total disposable household income",
    "DB040": "Region (NUTS 2)", "HB110": "Household type",
    "AGE": "Age", "ILOSTAT": "ILO labour status",
    "MAINSTAT": "Main activity status",
}

_SURVEY_LABELS = {
    "EUSILC": "EU-SILC", "LFS": "LFS", "HBS": "HBS",
    "SES": "SES", "EHIS": "EHIS", "ECHP": "ECHP",
}


def _prepare_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """Serialise list/dict columns, rename A-M, fill NA in Eurostat cols."""
    all_new_cols = (
        TARGET_POPULATION_COLS + EUROSTAT_MAPPING_COLS
        + MOTIVATION_COLS + SOURCE_REFERENCE_COLS + EXTRACTION_DIAGNOSTIC_COLS
    )
    for col in all_new_cols:
        if col not in df.columns:
            df[col] = "NA"
        else:
            def _ser(v):
                if isinstance(v, list):
                    return " | ".join(
                        json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
                        for x in v
                    )
                if isinstance(v, dict):
                    return json.dumps(v, ensure_ascii=False)
                return "NA" if v is None else v
            df[col] = df[col].apply(_ser)

    rename_map = {k: v for k, v in AFG_COLUMN_NAMES.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    euro_cols = [c for c in df.columns
                 if c.startswith(("EUSILC_","LFS_","HBS_","SES_","EHIS_","ECHP_"))]
    for col in euro_cols:
        df[col] = df[col].fillna("NA")

    return df


def _get_euro_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns
                   if c.startswith(("EUSILC_","LFS_","HBS_","SES_","EHIS_","ECHP_"))])


def _column_order(df: pd.DataFrame, euro_val_cols: list[str]) -> list[str]:
    afg_basic  = ["Country","Year","Category","Components","Measure"]
    afg_fiscal = ["Type","Impact t","Impact t+1","Impact t+2",
                  "Impact t+3","Impact t+4","Impact t+5"]
    afg_source = ["Source"]
    ordered = (
          [c for c in afg_basic              if c in df.columns]
        + [c for c in TARGET_POPULATION_COLS if c in df.columns]
        + [c for c in EUROSTAT_MAPPING_COLS  if c in df.columns]
        + sorted(euro_val_cols)
        + [c for c in MOTIVATION_COLS        if c in df.columns]
        + [c for c in SOURCE_REFERENCE_COLS  if c in df.columns]
        + [c for c in afg_fiscal             if c in df.columns]
        + [c for c in afg_source             if c in df.columns]
        + [c for c in EXTRACTION_DIAGNOSTIC_COLS if c in df.columns]
    )
    leftover = [c for c in df.columns if c not in ordered]
    return ordered + sorted(leftover)


def _build_dictionary_sheet(euro_val_cols: list[str]) -> pd.DataFrame:
    rows = [{"Section": "EUROSTAT SURVEY VARIABLES",
             "Code": "", "Description": "", "Survey": ""}]
    seen: set[str] = set()
    for col in euro_val_cols:
        parts = col.split("_", 1)
        if len(parts) != 2:
            continue
        survey_abbr, vcode = parts
        if vcode in seen:
            continue
        seen.add(vcode)
        rows.append({
            "Section": "", "Code": vcode,
            "Description": _VARIABLE_LABELS.get(vcode, f"Variable {vcode}"),
            "Survey": _SURVEY_LABELS.get(survey_abbr, survey_abbr),
        })
    rows += [
        {"Section": "", "Code": "", "Description": "", "Survey": ""},
        {"Section": "MOTIVATION CATEGORIES",
         "Code": "Category", "Description": "Label", "Survey": "Description"},
    ]
    for k, (label, desc) in MOTIVATION_TAXONOMY.items():
        rows.append({"Section": "", "Code": str(k),
                     "Description": label, "Survey": desc})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL EXPORT  (multi-year)
# ─────────────────────────────────────────────────────────────────────────────

def export_to_excel(
    results_by_year: dict[str, list[dict]],
    country: str,
    validation_by_year: dict[str, dict],
) -> Path:
    """
    Builds one Excel file covering all processed years for a country.

    Sheet layout:
      Dictionary       - variable codes + motivation taxonomy
      Data             - all measures combined across all years
      {year}           - one sheet per year, e.g. "2014", "2013"
      Seed Validation  - per-measure seed comparisons, all years

    Args:
      results_by_year    dict mapping year string -> list of output row dicts
      country            ISO-3 country code, used in filename
      validation_by_year dict mapping year string -> validation report dict

    Output filename: {country}_{year1}_{year2}_Extended.xlsx
    e.g. ITA_2013_2014_Extended.xlsx
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    years_sorted = sorted(results_by_year.keys())
    output_path  = OUTPUT_DIR / f"{country}_{'_'.join(years_sorted)}_Extended.xlsx"

    # Combined DataFrame across all years
    all_results  = [row for yr in years_sorted for row in results_by_year[yr]]
    combined_df  = _prepare_output_df(pd.DataFrame(all_results))
    euro_all     = _get_euro_cols(combined_df)
    combined_df  = combined_df[_column_order(combined_df, euro_all)]

    # Per-year DataFrames
    year_dfs: dict[str, pd.DataFrame] = {}
    for yr in years_sorted:
        yr_df  = _prepare_output_df(pd.DataFrame(results_by_year[yr]))
        yr_euro = _get_euro_cols(yr_df)
        year_dfs[yr] = yr_df[_column_order(yr_df, yr_euro)]

    dict_df = _build_dictionary_sheet(euro_all)

    # Seed validation: combine rows from all years, tag with year column
    all_val_rows: list[dict] = []
    for yr in years_sorted:
        for row in validation_by_year.get(yr, {}).get("matched_measures", []):
            all_val_rows.append({"year": yr, **row})
    val_df = (pd.DataFrame(all_val_rows) if all_val_rows
              else pd.DataFrame([{"note": "No seed data for any year."}]))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        dict_df.to_excel(writer,     sheet_name="Dictionary",     index=False)
        combined_df.to_excel(writer, sheet_name="Data",           index=False)
        for yr in years_sorted:
            year_dfs[yr].to_excel(writer, sheet_name=str(yr),     index=False)
        val_df.to_excel(writer,      sheet_name="Seed Validation", index=False)

    sheet_list = "Dictionary | Data | " + " | ".join(years_sorted) + " | Seed Validation"
    print(f"  Saved : {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Sheets: {sheet_list}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY / COST REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_quality_report(
    results: list[dict],
    token_tracker: dict,
    start_time: float,
    label: str = "",
) -> None:
    """
    Prints runtime, token cost, coverage stats, weighted quality score,
    confidence distribution, and motivation breakdown for one year's results.
    Pass label="ITA 2013" etc. for a clear header.
    """
    n       = len(results)
    runtime = tm.time() - start_time
    hdr     = f"QUALITY REPORT{' -- ' + label if label else ''}"

    print(f"\n{'=' * 65}")
    print(f"  {hdr}")
    print(f"{'=' * 65}\n")

    print("RUNTIME")
    print(f"  Total      : {runtime / 60:.1f} min  ({runtime:.0f} s)")
    print(f"  Per measure: {runtime / max(n,1):.1f} s\n")

    ext_in  = token_tracker.get("extraction_input", 0)
    ext_out = token_tracker.get("extraction_output", 0)
    map_in  = token_tracker.get("mapping_input", 0)
    map_out = token_tracker.get("mapping_output", 0)
    cached  = (token_tracker.get("extraction_cached_input", 0)
               + token_tracker.get("mapping_cached_input", 0))
    cost    = (ext_in/1e6*3.00 + ext_out/1e6*15.00
               + map_in/1e6*0.80 + map_out/1e6*4.00)

    print("TOKEN USAGE")
    print(f"  Extraction input  : {ext_in:>8,}")
    print(f"  Extraction output : {ext_out:>8,}")
    print(f"  Mapping input     : {map_in:>8,}")
    print(f"  Mapping output    : {map_out:>8,}")
    print(f"  Cached reads      : {cached:>8,}")
    print(f"  Estimated cost    : ${cost:.3f}\n")

    _na = (None, "NA", "", "null")
    src_found    = sum(1 for r in results if r.get("source_document_found") is True)
    target_found = sum(1 for r in results if r.get("target_population_found") is True)
    euro_found   = sum(1 for r in results if r.get("eurostat_mapping_found") is True)
    income_found = sum(1 for r in results if r.get("target_income_max") not in _na)
    age_found    = sum(1 for r in results if
                       r.get("target_age_min") not in _na or
                       r.get("target_age_max") not in _na)
    law_found    = sum(1 for r in results if
                       r.get("source_law_identifier") not in _na)

    print("COVERAGE")
    print(f"  Source found      : {src_found:>3}/{n}")
    print(f"  Target identified : {target_found:>3}/{n}")
    print(f"    With income     : {income_found:>3}/{n}  <- drives s_{{im}}")
    print(f"    With age        : {age_found:>3}/{n}")
    print(f"  Eurostat mapping  : {euro_found:>3}/{n}")
    print(f"  Law identifiers   : {law_found:>3}/{n}\n")

    qm = {
        "source_coverage":     src_found    / n if n else 0,
        "target_coverage":     target_found / n if n else 0,
        "income_detail":       income_found / target_found if target_found else 0,
        "eurostat_mapping":    euro_found   / target_found if target_found else 0,
        "high_conf_share":     sum(1 for r in results
                                   if r.get("confidence_grade") == "high") / n if n else 0,
        "motivation_explicit": 1 - sum(
            1 for r in results
            if "motivation inferred" in str(r.get("extraction_notes","")).lower()
        ) / n if n else 0,
    }
    overall = sum(QUALITY_SCORE_WEIGHTS[k] * qm[k] for k in QUALITY_SCORE_WEIGHTS) * 100
    grade   = ("EXCELLENT"  if overall >= 80 else
               "GOOD"       if overall >= 70 else
               "ACCEPTABLE" if overall >= 60 else
               "FAIR"       if overall >= 50 else "POOR")

    print("QUALITY SCORE  [scientifically weighted]")
    for k, w in QUALITY_SCORE_WEIGHTS.items():
        print(f"  {k:<22} ({w:.0%} weight): {qm[k]:.1%}")
    print(f"\n  Overall: {overall:.1f}/100  [{grade}]\n")

    print("MECHANICAL CONFIDENCE")
    for g in ("high", "medium", "low"):
        cnt = sum(1 for r in results if r.get("confidence_grade") == g)
        bar = "=" * (cnt * 20 // n) if n else ""
        print(f"  {g.capitalize():<8}: {cnt:>3}/{n}  {bar}")
    scores = [r["confidence_score"] for r in results
              if isinstance(r.get("confidence_score"), (int, float))]
    if scores:
        print(f"  Mean : {sum(scores)/len(scores):.1f}/100\n")

    print("MOTIVATION CATEGORIES")
    cat_counts: dict = {}
    for r in results:
        c = r.get("motivation_category", "NA")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    for cid in sorted(cat_counts, key=lambda x: (not isinstance(x, int), x)):
        label = MOTIVATION_TAXONOMY.get(cid, ("?",""))[0] if isinstance(cid, int) else "?"
        print(f"  {cid}. {label:<30}: {cat_counts[cid]:>3}/{n}")
    print()