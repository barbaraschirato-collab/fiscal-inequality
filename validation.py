"""
validation.py
─────────────────────────────────────────────────────────────────────────────
Two independent validation layers:

  1. compute_mechanical_confidence()
       Scores each extracted measure from 0–100 using observable signals
       (source found, law found, income numeric, etc.).
       Replaces Claude's self-reported confidence, which is unreliable.

  2. validate_against_seed() / print_seed_validation_report()
       Compares pipeline output to the hand-coded ITA 2013 seed case and
       reports motivation accuracy, income MAE, target recall.

To add a new confidence signal: add it to _compute_signals() and
CONFIDENCE_WEIGHTS in config.py.
To change what the seed validation measures: edit validate_against_seed().
─────────────────────────────────────────────────────────────────────────────
"""

from config import CONFIDENCE_WEIGHTS


# ─────────────────────────────────────────────────────────────────────────────
# MECHANICAL CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _compute_signals(extracted: dict, mapping: dict) -> dict[str, float]:
    """
    Computes individual confidence signals (0–1 each) from observable facts.
    Each signal name must correspond to a key in config.CONFIDENCE_WEIGHTS.
    """
    signals: dict[str, float] = {}

    # Source document found
    signals["source_found"] = 1.0 if extracted.get("source_document_found") is True else 0.0

    # Legal reference extracted
    law = str(extracted.get("source_law_identifier", "NA") or "NA")
    signals["law_found"] = 0.0 if law in ("NA", "", "null", "None") else 1.0

    # Non-trivial excerpt present
    excerpt = str(extracted.get("source_excerpt_original", "") or "")
    signals["excerpt_found"] = 0.0 if excerpt in ("NA", "", "null", "None") else 1.0

    # Target population identified
    signals["target_found"] = 1.0 if extracted.get("target_population_found") is True else 0.0

    # At least one specific eligibility criterion coded
    def _has_value(key: str) -> bool:
        v = extracted.get(key)
        return v not in (None, "NA", "", "null", "None", "all")

    has_criterion = (
        _has_value("target_income_max")
        or _has_value("target_age_max")
        or _has_value("target_age_min")
        or (
            _has_value("target_employment_status")
            and str(extracted.get("target_employment_status", "")).lower()
               not in ("na", "all", "null", "none", "")
        )
    )
    signals["specific_criterion"] = 1.0 if has_criterion else 0.0

    # Income threshold is a real positive number
    inc_max = extracted.get("target_income_max")
    try:
        inc_val = float(inc_max)
        signals["income_numeric"] = 1.0 if inc_val > 0 else 0.5
    except (TypeError, ValueError):
        signals["income_numeric"] = 0.0

    # Eurostat mapping found with at least one variable
    variables = mapping.get("eurostat_variables", [])
    signals["eurostat_mapped"] = (
        1.0 if mapping.get("eurostat_mapping_found") and variables else 0.0
    )

    # Motivation not flagged as inferred
    notes = str(extracted.get("extraction_notes", "")).lower()
    signals["motivation_explicit"] = 0.5 if "motivation inferred" in notes else 1.0

    # Claude's own self-report (kept as weak signal only)
    raw_conf = str(extracted.get("extraction_confidence_raw", "")).lower()
    signals["claude_confidence"] = {"high": 1.0, "medium": 0.5, "low": 0.0}.get(raw_conf, 0.0)

    return signals


def compute_mechanical_confidence(extracted: dict, mapping: dict) -> dict:
    """
    Computes a mechanical confidence score from observable signals.

    Returns:
      {
        "confidence_score"  : float   0–100
        "confidence_grade"  : str     "high" | "medium" | "low"
        "confidence_signals": dict    signal_name → value (audit trail)
      }

    The weights are defined in config.CONFIDENCE_WEIGHTS so they can be
    adjusted without touching this function.
    """
    signals = _compute_signals(extracted, mapping)

    score = sum(
        CONFIDENCE_WEIGHTS.get(k, 0) * v
        for k, v in signals.items()
    )

    grade = "high" if score >= 70 else ("medium" if score >= 45 else "low")

    return {
        "confidence_score":   round(score, 1),
        "confidence_grade":   grade,
        "confidence_signals": signals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEED-CASE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_against_seed(results: list[dict], seed: dict[str, dict]) -> dict:
    """
    Compares pipeline output rows against the hand-coded seed case.

    Matching is by normalised (stripped, lowercase) measure text.

    Returns a validation report dict:
      n_matched          : int    — measures matched
      motivation_accuracy: float  — % exact match on motivation_category (1–7)
      income_mae         : float  — mean |pipeline - seed| on income threshold
      income_coverage    : float  — % of seed income measures where pipeline found income
      target_recall      : float  — % of seed targeted measures pipeline also targeted
      matched_measures   : list   — per-measure comparison rows (written to Excel)
    """
    if not seed:
        return {"n_matched": 0, "note": "No seed data available."}

    matched: list[dict] = []

    for row in results:
        # Look up by normalised measure text
        measure_key = str(
            row.get("E", row.get("Measure", ""))
        ).strip().lower()
        if measure_key not in seed:
            continue

        ref = seed[measure_key]

        comparison: dict = {
            "measure":              measure_key[:80],
            # Motivation
            "pipeline_motivation":  row.get("motivation_category", "NA"),
            "seed_motivation":      ref.get("motivation_category", "NA"),
            "motivation_match":     (
                str(row.get("motivation_category", ""))
                == str(ref.get("motivation_category", ""))
            ),
            # Income
            "pipeline_income_max":  row.get("target_income_max", "NA"),
            "seed_income_max":      ref.get("target_income_max", "NA"),
            # Target found
            "pipeline_target":      row.get("target_population_found", False),
            "seed_target":          str(ref.get("target_population_found", "")).lower()
                                    in ("true", "1", "yes"),
            # Confidence
            "confidence_score":     row.get("confidence_score", "NA"),
        }

        # Income absolute error (where both have numeric values)
        pip_inc = row.get("target_income_max")
        ref_inc = ref.get("target_income_max")
        try:
            comparison["income_abs_error"] = abs(float(pip_inc) - float(ref_inc))
        except (TypeError, ValueError):
            comparison["income_abs_error"] = None

        matched.append(comparison)

    if not matched:
        return {"n_matched": 0, "note": "No measures overlapped with seed."}

    n = len(matched)

    # ── Motivation accuracy ───────────────────────────────────────────────────
    mot_correct   = sum(1 for m in matched if m["motivation_match"])
    mot_accuracy  = 100 * mot_correct / n

    # ── Income MAE ───────────────────────────────────────────────────────────
    income_errors = [m["income_abs_error"] for m in matched if m["income_abs_error"] is not None]
    income_mae    = (sum(income_errors) / len(income_errors)) if income_errors else None

    # ── Income coverage ───────────────────────────────────────────────────────
    seed_has_income  = [m for m in matched
                        if m["seed_income_max"] not in (None, "NA", "", "null")]
    pip_found_income = [m for m in seed_has_income
                        if m["pipeline_income_max"] not in (None, "NA", "", "null")]
    income_coverage  = (
        100 * len(pip_found_income) / len(seed_has_income) if seed_has_income else None
    )

    # ── Target recall ─────────────────────────────────────────────────────────
    seed_targeted = [m for m in matched if m["seed_target"]]
    pip_targeted  = [m for m in seed_targeted if m["pipeline_target"] is True]
    target_recall = (
        100 * len(pip_targeted) / len(seed_targeted) if seed_targeted else None
    )

    return {
        "n_matched":           n,
        "motivation_accuracy": round(mot_accuracy, 1),
        "income_mae":          round(income_mae, 0) if income_mae is not None else "N/A",
        "income_coverage":     round(income_coverage, 1) if income_coverage is not None else "N/A",
        "target_recall":       round(target_recall, 1) if target_recall is not None else "N/A",
        "matched_measures":    matched,
    }


def print_seed_validation_report(report: dict) -> None:
    """Prints the seed validation summary to stdout."""
    print(f"\n{'=' * 65}")
    print("  SEED-CASE VALIDATION  (vs hand-coded ITA 2013)")
    print(f"{'=' * 65}\n")

    if report.get("n_matched", 0) == 0:
        print(f"  {report.get('note', 'No matched measures.')}\n")
        return

    n = report["n_matched"]
    print(f"  Measures matched   : {n}")
    print(f"  Motivation accuracy: {report['motivation_accuracy']}%"
          f"  (exact match on 1-7 code)")
    print(f"  Income coverage    : {report['income_coverage']}%"
          f"  (pipeline found income where seed has it)")
    print(f"  Income MAE         : €{report['income_mae']}"
          f"  (mean absolute error on threshold)")
    print(f"  Target recall      : {report['target_recall']}%"
          f"  (pipeline found target where seed has it)\n")

    # Surface motivation disagreements
    disagreements = [m for m in report["matched_measures"] if not m["motivation_match"]]
    if disagreements:
        print(f"  MOTIVATION DISAGREEMENTS  ({len(disagreements)}/{n}):")
        for m in disagreements[:10]:
            print(f"    '{m['measure'][:65]}'")
            print(f"      Pipeline: {m['pipeline_motivation']}"
                  f"  |  Seed: {m['seed_motivation']}")
    else:
        print("  ✓ All motivation categories agree with seed.")
    print()