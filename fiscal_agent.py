"""
fiscal_agent.py
─────────────────────────────────────────────────────────────────────────────
Pipeline orchestrator.

Entry point: run_multi_year(country, years)
  Default: python fiscal_agent.py  ->  ITA 2013 + 2014

CHECKPOINT BEHAVIOUR
─────────────────────
A checkpoint is saved every CHECKPOINT_EVERY measures AND once more when
a year finishes completely (the "final checkpoint").

When run_multi_year is called:
  - Year already done  (final checkpoint covers all N measures)
    -> extraction loop SKIPPED ENTIRELY, zero API calls, <1 second
  - Year interrupted   (partial checkpoint)
    -> resumes from last checkpoint row
  - Year not started   (no checkpoint)
    -> runs from scratch

Practical effect:
  python fiscal_agent.py            # runs 2013 then 2014 fresh
  python fiscal_agent.py            # re-runs: both skipped instantly, Excel rebuilt
  python fiscal_agent.py --years 2012 2013 2014  # 2013+2014 skipped, 2012 extracted

Output: {country}_{year1}_{year2}_Extended.xlsx
  Sheets: Dictionary | Data | 2013 | 2014 | Seed Validation
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import time
import argparse
import anthropic

from config import (
    SOURCE_CACHE_LIMIT,
    CHECKPOINT_EVERY,
    API_SLEEP_SECONDS,
    TARGET_POPULATION_COLS,
    EUROSTAT_MAPPING_COLS,
    MOTIVATION_COLS,
    SOURCE_REFERENCE_COLS,
    EXTRACTION_DIAGNOSTIC_COLS,
)
from loaders import (
    load_afg_measures,
    find_source_documents,
    load_eurostat_dictionaries,
    load_example_sheet,
    example_as_prompt,
    example_as_seed,
)
from chunking import build_source_context
from extraction import extract_from_sources, enrich_with_fallback_numbers
from mapping import (
    map_to_eurostat,
    force_obvious_mappings,
    build_eurostat_value_columns,
    EMPTY_MAPPING,
)
from validation import (
    compute_mechanical_confidence,
    validate_against_seed,
    print_seed_validation_report,
)
from output import (
    save_checkpoint,
    load_latest_checkpoint,
    export_to_excel,
    print_quality_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED PREFIX BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_source_prefix(source_docs: dict) -> list:
    """Ephemeral Sonnet cache block for this year's source documents."""
    full_text = "\n\n".join(
        f"{'─'*50}\nDOCUMENT: {name}\n{'─'*50}\n{text}"
        for name, text in source_docs.items()
    )
    return [{
        "type": "text",
        "text": (
            "You are a senior fiscal policy researcher expert in European public finance, "
            "Italian budget documents, and Eurostat microdata.\n\n"
            "FULL SOURCE DOCUMENTS FOR THIS COUNTRY-YEAR:\n"
            + full_text[:SOURCE_CACHE_LIMIT]
        ),
        "cache_control": {"type": "ephemeral"},
    }]


def _build_dict_prefix(eurostat_text: str) -> list:
    """Ephemeral Haiku cache block for Eurostat dictionaries (shared across years)."""
    return [{
        "type": "text",
        "text": (
            "You are a Eurostat microdata expert. "
            "Use these survey dictionaries to map target populations to variables.\n\n"
            + eurostat_text
        ),
        "cache_control": {"type": "ephemeral"},
    }]


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT ROW BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_output_row(afg_row, extracted, mapping, confidence):
    extracted["confidence_score"]   = confidence["confidence_score"]
    extracted["confidence_grade"]   = confidence["confidence_grade"]
    extracted["confidence_signals"] = json.dumps(
        confidence["confidence_signals"], ensure_ascii=False
    )
    combined  = {**extracted, **mapping}
    all_cols  = (TARGET_POPULATION_COLS + EUROSTAT_MAPPING_COLS
                 + MOTIVATION_COLS + SOURCE_REFERENCE_COLS
                 + EXTRACTION_DIAGNOSTIC_COLS)
    output_row = dict(afg_row)

    for col in all_cols:
        val = combined.get(col, "NA")
        if isinstance(val, list):
            val = " | ".join(
                json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v)
                for v in val
            )
        elif isinstance(val, dict):
            val = json.dumps(val, ensure_ascii=False)
        elif val is None:
            val = "NA"
        output_row[col] = val

    for col, val in build_eurostat_value_columns(
        mapping.get("eurostat_variables", [])
    ).items():
        output_row[col] = val

    return output_row


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-YEAR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(country, year, client, cached_dict_prefix):
    """
    Processes all AFG measures for one country-year.

    Returns (results, validation_report).

    Skip logic:
      If the final checkpoint for (country, year) covers all N AFG measures,
      the extraction loop is skipped entirely and results are loaded from disk.
      Only seed validation and the quality report are re-run (fast, no API cost).
    """
    year_start = time.time()
    print(f"\n{'─'*65}")
    print(f"  Processing: {country} {year}")
    print(f"{'─'*65}")

    token_tracker = {
        "extraction_input": 0, "extraction_output": 0,
        "extraction_cached_input": 0,
        "mapping_input": 0, "mapping_output": 0,
        "mapping_cached_input": 0,
    }

    # AFG measures
    afg_df = load_afg_measures(country, year)
    n_afg  = len(afg_df)
    print()

    # Source documents (year-specific)
    print(f"  Source documents ({country} {year})...")
    source_docs = find_source_documents(country, year)
    print(f"  {len(source_docs)} document(s)\n")

    # Example + seed from the same workbook, year-specific sheet
    print(f"  Example/seed workbook ({country} {year})...")
    example_df, example_sheet = load_example_sheet(country=country, year=year)
    example_text = example_as_prompt(example_df, example_sheet)
    seed_data    = example_as_seed(example_df)
    print(f"  Prompt example : {len(example_text):,} chars  |  Seed: {len(seed_data)}\n")

    # Sonnet source prefix (year-specific)
    cached_source_prefix = _build_source_prefix(source_docs)
    print(f"  Source prefix  : {len(cached_source_prefix[0]['text']):,} chars\n")

    # Load checkpoint
    results, already_done = load_latest_checkpoint(country, year)

    # ── Skip if already complete ─────────────────────────────────────────────
    if already_done >= n_afg:
        print(f"  Already complete ({already_done}/{n_afg} measures). Skipping extraction.\n")

    else:
        # ── Extraction loop ───────────────────────────────────────────────────
        remaining = n_afg - already_done
        print(f"  Extracting {remaining} measures (resuming from {already_done})...\n")

        for i, (_, row) in enumerate(afg_df.iterrows()):
            if i < already_done:
                continue

            measure    = str(row.get("E", "")).strip()
            category1  = str(row.get("C", "")).strip()
            category2  = str(row.get("D", "")).strip()
            afg_source = str(row.get("M", "")).strip()

            if not measure or measure.lower() == "nan":
                continue

            print(f"  [{i+1:>3}/{n_afg}] {measure[:70]}...")

            source_chunk = build_source_context(source_docs, measure)

            extracted, ext_usage = extract_from_sources(
                measure=measure, category1=category1, category2=category2,
                source_chunk=source_chunk, example_text=example_text,
                cached_prefix=cached_source_prefix, client=client,
                country=country, year=year,
                afg_source=afg_source, source_docs=source_docs,
            )
            token_tracker["extraction_input"]        += ext_usage["input"]
            token_tracker["extraction_output"]       += ext_usage["output"]
            token_tracker["extraction_cached_input"] += ext_usage["cache_read"]

            extracted = enrich_with_fallback_numbers(measure, source_docs, extracted)

            if extracted.get("target_population_found") is True:
                mapping, map_usage = map_to_eurostat(
                    target_summary=extracted.get("target_population_summary_en", ""),
                    target_details=extracted,
                    cached_dict_prefix=cached_dict_prefix,
                    client=client, country=country,
                )
                mapping = force_obvious_mappings(extracted, mapping)
                token_tracker["mapping_input"]        += map_usage["input"]
                token_tracker["mapping_output"]       += map_usage["output"]
                token_tracker["mapping_cached_input"] += map_usage["cache_read"]
            else:
                mapping = dict(EMPTY_MAPPING)

            confidence = compute_mechanical_confidence(extracted, mapping)
            results.append(
                _build_output_row(row.to_dict(), extracted, mapping, confidence)
            )

            # Periodic checkpoint
            if (i + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(results, country, year, i + 1)

            time.sleep(API_SLEEP_SECONDS)

        # Final checkpoint - marks this year as done for all future runs
        save_checkpoint(results, country, year, n_afg)
        print(f"\n  Done: {len(results)} measures extracted.")

    # Validation and quality report always run (fast, no API cost)
    print(f"\n  Seed validation ({country} {year})...")
    validation_report = validate_against_seed(results, seed_data)
    print_seed_validation_report(validation_report)

    print_quality_report(
        results, token_tracker, year_start, label=f"{country} {year}"
    )

    return results, validation_report


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-YEAR ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_year(country="ITA", years=None):
    """
    Processes multiple years for one country, writing a single combined Excel.

    Shared across ALL years (loaded once):
      - Anthropic API client
      - Eurostat dictionaries + Haiku cached prefix

    Per year (reloaded each iteration):
      - AFG measures filtered to that year
      - Source documents  (ITA_2013_*.pdf  vs  ITA_2014_*.pdf)
      - Example/seed sheet (correct tab of Italy_2013_2014.xlsx)
      - Sonnet source prefix
    """
    if years is None:
        years = ["2013", "2014"]

    overall_start = time.time()
    print(f"\n{'='*65}")
    print(f"  FISCAL PIPELINE  v2  |  {country}  {' + '.join(years)}")
    print(f"{'='*65}\n")

    # Shared setup (once for all years)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
    client = anthropic.Anthropic(api_key=api_key)

    print("Loading Eurostat dictionaries (shared across all years)...")
    eurostat_text      = load_eurostat_dictionaries()
    cached_dict_prefix = _build_dict_prefix(eurostat_text)
    print(f"  {len(eurostat_text):,} chars\n")

    # Process each year
    results_by_year    = {}
    validation_by_year = {}

    for year in years:
        results, val_report = run_pipeline(
            country=country,
            year=year,
            client=client,
            cached_dict_prefix=cached_dict_prefix,
        )
        results_by_year[year]    = results
        validation_by_year[year] = val_report

    # Combined Excel export
    print(f"\n{'='*65}")
    print(f"  Exporting combined Excel  ({country} {' + '.join(years)})...")
    output_path = export_to_excel(
        results_by_year=results_by_year,
        country=country,
        validation_by_year=validation_by_year,
    )

    # Combined quality summary
    all_results  = [r for yr in years for r in results_by_year[yr]]
    empty_tokens = {k: 0 for k in [
        "extraction_input", "extraction_output", "extraction_cached_input",
        "mapping_input", "mapping_output", "mapping_cached_input",
    ]}
    print_quality_report(
        all_results, empty_tokens, overall_start,
        label=f"{country} ALL YEARS ({' + '.join(years)})",
    )

    total_time = time.time() - overall_start
    print(f"  Measures : {len(all_results)}")
    print(f"  Runtime  : {total_time/60:.1f} min")
    print(f"  Output   : {output_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fiscal Measure Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fiscal_agent.py                          # ITA 2013+2014\n"
            "  python fiscal_agent.py --years 2014             # ITA 2014 only\n"
            "  python fiscal_agent.py --country FRA --years 2010 2011\n"
        ),
    )
    parser.add_argument("--country", default="ITA")
    parser.add_argument("--years", default=["2013", "2014"], nargs="+")
    args = parser.parse_args()
    run_multi_year(country=args.country, years=args.years)