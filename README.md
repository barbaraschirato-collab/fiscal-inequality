# Fiscal Policy Extraction Agent

A Python pipeline that uses Claude to automatically extract, classify, and map fiscal policy measures from government budget documents to Eurostat microdata survey variables. Built for the [AFG Dataset](https://igier.unibocconi.eu/research/datasets/fiscal-adjustment-plans/dataset) extension project on fiscal policy and inequality.

---

## What it does

Given a list of fiscal policy measures from the AFG dataset (Alesina, Favero & Giavazzi 2019) and a set of official government budget PDFs, the agent:

1. **Extracts** the target population of each measure (income thresholds, age, employment status, household type, sector, geography) from the source documents
2. **Classifies** the motivation behind each measure into a 7-category taxonomy (fiscal consolidation, redistribution, public investment, etc.)
3. **Maps** each measure's target population to the appropriate Eurostat microdata survey variable(s) — EU-SILC, LFS, HBS, SES, EHIS, or ECHP — with exact code values
4. **Handles indirect taxes** (VAT, excise duties) as expenditure-based targets, mapping them deterministically to HBS COICOP expenditure categories without an LLM call
5. **Validates** results against a hand-coded seed case, reporting motivation accuracy, income threshold, and target recall
6. **Scores** each extracted measure with a mechanical confidence score (0–100) based on observable signals (source found, law identified, income numeric, etc.)

Output is a multi-sheet Excel workbook covering multiple country-years, with one sheet per year plus a combined data sheet and seed validation report.

---

## Architecture

```
fiscal_agent.py   ← orchestrator, checkpoint logic
├── loaders.py    ← reads AFG Excel, source PDFs, Eurostat dictionaries
├── chunking.py   ← scores and selects relevant document excerpts per measure
├── extraction.py ← Claude Sonnet: extracts target population + motivation
│                   + fallback regex for income thresholds
│                   + deterministic COICOP detection for indirect taxes
├── mapping.py    ← Claude Haiku: maps target population → Eurostat variables
│                   + deterministic HBS mapping for expenditure-based measures
├── validation.py ← mechanical confidence scoring + seed case validation
├── output.py     ← checkpointing, Excel export, quality report
└── config.py     ← all paths, model names, taxonomy, weights (edit here)
```

**Two-model design:** Claude Sonnet (`claude-sonnet-4-6`) handles the harder extraction task (reading dense legal/budget text); Claude Haiku (`claude-haiku-4-5-20251001`) handles the structured mapping task (matching variables in a dictionary). Source documents are cached as prompt prefixes so the full PDF text is only tokenised once per year, not once per measure.

---

## Folder structure

The pipeline expects the following layout (configurable in `config.py` or via the `FISCAL_BASE_DIR` environment variable):

```
<base_dir>/
├── Excel Input AFG/
│   └── AppendixTables_AFG.xlsx      ← AFG measures input (country, year, category, etc.)
├── Sources/
│   ├── ITA_2013_Stability_Programme.pdf
│   ├── ITA_2013_NADEF.pdf
│   ├── ITA_2014_Stability_Programme.pdf
│   └── ...                          ← named {COUNTRY}_{YEAR}_*.pdf
├── Microdata/
│   ├── EU_SILC_Dictionary_Variables.xlsx
│   ├── LFS_Dictionary_Variables.xlsx
│   ├── HBS_Dictionary_Variables.xlsx
│   └── ...                          ← Eurostat survey variable dictionaries
├── Excel Example/
│   └── Italy_2013_2014.xlsx         ← hand-coded examples (one sheet per year)
├── Seed/
│   └── Italy_2013_2014.xlsx         ← hand-coded seed for validation
└── Excel Output/
    └── ITA_2013_2014_Extended.xlsx  ← generated output
```

Source documents are matched automatically by the `{COUNTRY}_{YEAR}_` filename prefix. Any PDF or DOCX with this prefix in the `Sources/` folder is loaded for the corresponding country-year.

---

## Installation

```bash
pip install anthropic pandas openpyxl pdfplumber python-docx tenacity
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Optionally override the base directory:

```bash
export FISCAL_BASE_DIR=/path/to/your/data
```

---

## Usage

```bash
# Default: ITA 2013 + 2014
python fiscal_agent.py

# Specific years
python fiscal_agent.py --years 2011 2012 2013

# WIP: Different country (requires AFG rows and source PDFs for that country, currently code is highly specific for Italy)
python fiscal_agent.py --country FRA --years 2010 2011
```

### Checkpoint and resume

A checkpoint is saved every 5 measures and once more when a year completes. On re-run:

| State | Behaviour |
|---|---|
| Year complete | Skipped instantly (0 API calls), Excel rebuilt |
| Year interrupted mid-run | Resumes from last saved measure |
| Year not started | Runs from scratch |

Delete `{COUNTRY}_{YEAR}_checkpoint_*.json` from `Excel Output/` to force a full rerun of a year.

---

## Output

The output Excel file (`ITA_2013_2014_Extended.xlsx`) contains:

| Sheet | Contents |
|---|---|
| **Dictionary** | Eurostat variable dictionary (all surveys) |
| **Data** | All years combined, all extracted columns |
| **2013**, **2014**, … | Per-year data |
| **Seed Validation** | Comparison against hand-coded reference cases |

### Extracted columns

**Target population**

| Column | Description |
|---|---|
| `target_population_found` | Whether a specific target population was identified |
| `target_population_summary_en` | English prose description |
| `target_age_min` / `target_age_max` | Age eligibility bounds |
| `target_income_min` / `target_income_max` | Income threshold (€) |
| `target_income_type` | Gross / net / ISEE / other |
| `target_employment_status` | e.g. employed, self-employed, unemployed, all |
| `target_household_type` | e.g. single, couple with children |
| `target_sector` | NACE sector if sector-specific |
| `target_geographic` | NUTS region if geographically targeted |
| `target_other_criteria` | Any other eligibility criteria |
| `target_expenditure_category` | COICOP code for expenditure-based (indirect tax) measures |

**Eurostat mapping**

| Column | Description |
|---|---|
| `eurostat_mapping_found` | Whether a mapping was identified |
| `primary_survey` | EU-SILC / LFS / HBS / SES / EHIS / ECHP |
| `eurostat_variables` | JSON list of variables with codes, values, and rationale |
| `secondary_surveys` | Additional surveys |
| `filter_conditions_combined` | Full filter expression (e.g. `HY020 <= 15000 AND RB081 < 65`) |

**Motivation**

| Column | Description |
|---|---|
| `motivation_specific` | Free-text motivation extracted from the document |
| `motivation_category` | Integer 1–7 (see taxonomy below) |

**Source reference**

| Column | Description |
|---|---|
| `source_document_found` | Whether the measure was found in a source PDF |
| `source_document_name` | Filename |
| `source_page_reference` | Page/table reference |
| `source_excerpt_original` | Verbatim excerpt (Italian/original language) |
| `source_law_identifier` | Law/decree identifier (e.g. DL 76/2013) |

**Diagnostics**

| Column | Description |
|---|---|
| `confidence_score` | Mechanical score 0–100 |
| `confidence_grade` | high / medium / low |
| `confidence_signals` | JSON breakdown of each signal |
| `extraction_confidence_raw` | Claude's self-reported confidence |
| `extraction_notes` | Claude's free-text notes |

---

## Motivation taxonomy

| Code | Category | Description |
|---|---|---|
| 1 | Fiscal consolidation | Reduce deficit/debt: revenue increases, spending cuts |
| 2 | Economic expansion | Boost growth/employment: tax cuts, incentives, stimulus |
| 3 | Redistribution | Help poor/vulnerable, reduce inequality |
| 4 | Public investment | Infrastructure/human capital: education, health, R&D |
| 5 | Administrative/Regulatory | Efficiency, governance, simplification |
| 6 | Foreign policy | International commitments, peacekeeping |
| 7 | Other | Miscellaneous, unclassifiable, one-off events |

---

## Handling indirect taxes

Measures with AFG category code `INDT` (indirect taxes: VAT, excise duties, consumption taxes) receive a different treatment from income-targeted measures. Their target population is all consumers of a specific good or service — defined by expenditure, not income.

The pipeline handles these deterministically, without an LLM call:

1. **Detection**: `_fix_expenditure_target()` in `extraction.py` fires when `category1 = INDT`. It uses a keyword table to identify the relevant COICOP category from the measure text (e.g. "tabacchi" → `02.2 Tobacco`, "benzina/gasolio" → `07.2 Operation of personal transport equipment`).
2. **Mapping**: `_map_expenditure_hbs()` in `mapping.py` looks up the COICOP code in the full HBS variable dictionary and returns the HBS variable directly. No Haiku call is made.
3. **Result**: `primary_survey = HBS`, `variable_code = 02.2`, `filter_conditions_combined = HBS.02.2 > 0`.

> **Note**: Only AFG category `INDT` triggers this path. Government consumption measures (category `CONS`) share the label "Goods and Services" in the AFG but are correctly excluded — they have no household expenditure target.

The COICOP keyword matcher covers all 12 COICOP divisions at both division and sub-division level (e.g. `01.1.1` Bread, `01.1.2` Meat, `07.1` Vehicle purchase vs `07.2` Fuel/operation) and uses a prefix regex `(?<![a-z])` to avoid false substring matches (e.g. `gasolio` triggering `olio`/oils).

---

## Confidence scoring

Rather than relying on Claude's self-reported confidence, every result is scored mechanically from observable signals:

| Signal | Weight |
|---|---|
| Source document found | 12 |
| Law/decree identifier found | 12 |
| Verbatim excerpt found | 8 |
| Target population identified | 15 |
| Specific eligibility criterion | 18 |
| Income threshold is numeric | 15 |
| Eurostat variable mapped | 10 |
| Motivation explicitly stated | 6 |
| Claude's self-reported confidence | 4 |
| **Total** | **100** |

Grades: **high** ≥ 65, **medium** ≥ 40, **low** < 40.

---

## Income threshold fallback

If Claude confirms income-based targeting exists but fails to extract the numeric threshold, a regex fallback scans a ±1,500 character window around measure keywords in the source PDFs. It only fires when both conditions hold:

1. `target_population_found = True` and at least one income field is non-empty (Claude confirmed targeting exists)
2. `target_income_max` is still null (the specific number is missing)

This two-condition gate prevents the fallback from picking up spurious amounts (e.g. the IRPEF top bracket ceiling of €150,000) from nearby fiscal tables when no income criterion actually exists.

---

## Configuration

All tuneable parameters are in `config.py`:

```python
MODEL_MAIN        = "claude-sonnet-4-6"          # extraction model
MODEL_MAPPING     = "claude-haiku-4-5-20251001"  # mapping model
CHUNK_RADIUS      = 4500    # chars of source text window around keyword match
SOURCE_CACHE_LIMIT = 40_000 # chars of source text in cached prompt prefix
CHECKPOINT_EVERY  = 5       # save checkpoint every N measures
API_SLEEP_SECONDS = 0.8     # pause between API calls
```

To extend to a new country, add entries to `LEGAL_PATTERNS_BY_COUNTRY` (law identifier regex patterns) and `REGIONAL_MAPPINGS` (NUTS code → region name) in `config.py`. Everything else (source document discovery, AFG filtering, year iteration) is automatic.

---

## Extending to new years

For years without a hand-coded seed reference, the pipeline:
- Uses the nearest available seed year as the few-shot example for Claude (falls back to the first sheet found)
- Skips seed validation for that year (`No seed data found — skipping validation`)
- Runs extraction and mapping as normal

No code changes are needed. Just add the source PDFs to `Sources/` with the correct naming convention and run:

```bash
python fiscal_agent.py --years 2011 2012 2013 2014
```

Years with completed checkpoints are skipped instantly; only new years consume API tokens.

---

## Reference

This pipeline extends the dataset from:

> Alesina, A., Favero, C., & Giavazzi, F. (2019). *Effects of Austerity: Expenditure- and Tax-based Approaches*. Journal of Economic Perspectives, 33(2), 141–162.

> Alesina, A., Favero, C., & Giavazzi, F. (2019). *Austerity: When It Works and When It Doesn't*. Princeton University Press.

The AFG dataset identifies fiscal consolidation episodes for 16 OECD countries (1978–2014) using a narrative approach based on official budget documents.
