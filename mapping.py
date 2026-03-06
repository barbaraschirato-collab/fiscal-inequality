"""
mapping.py
─────────────────────────────────────────────────────────────────────────────
Eurostat variable mapping: Haiku API call + deterministic override rules.

Two layers:
  1. map_to_eurostat()        — Haiku LLM call using survey dictionaries
  2. force_obvious_mappings() — mechanical rules that override/supplement Haiku
                                when a target criterion is present but Haiku
                                missed the mandatory variable

To update the mapping prompt:       edit MAPPING_PROMPT_TEMPLATE below.
To add a new mandatory variable:    add a block to force_obvious_mappings().
To change the Haiku model:          edit config.MODEL_MAPPING.
─────────────────────────────────────────────────────────────────────────────
"""

import re
import json
import anthropic
import tenacity
import pandas as pd

from config import MODEL_MAPPING, REGIONAL_MAPPINGS


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

MAPPING_PROMPT_TEMPLATE = """
Map this fiscal measure's target population to the correct Eurostat survey
variables, with exact code values from the dictionaries provided above.

TARGET POPULATION DETAILS
─────────────────────────
Summary      : {target_summary}
Age          : {target_age_min} – {target_age_max}
Income       : {target_income_min} – {target_income_max}  ({target_income_type})
Employment   : {target_employment_status}
Household    : {target_household_type}
Sector       : {target_sector}
Geographic   : {target_geographic}
Other        : {target_other_criteria}

MANDATORY MAPPING RULES
───────────────────────
Apply these rules whenever a criterion is present (not NA / null):

  AGE present       → include RB081  (Age, EU-SILC)
  EMPLOYMENT present → include PL031  (Self-defined economic status, EU-SILC)
  INCOME present    → include HY020  (Total disposable household income, EU-SILC)
                       value format: "<=X" where X is the income threshold
  SECTOR present    → include PL111  (NACE industry code, EU-SILC or LFS)
  GEOGRAPHIC present → include DB040  (Region NUTS-2, EU-SILC)
  HOUSEHOLD TYPE    → include HB110  (Household type, EU-SILC)

For every variable, scan ALL rows in the dictionary CSV above and return the
EXACT Codes value from the matching row.

Set eurostat_mapping_found = false ONLY if target_population_found was false
(i.e. the measure is genuinely universal with no eligibility criteria at all).

Return ONLY valid JSON:
{{
  "eurostat_mapping_found": true,
  "primary_survey": "<EU_SILC|LFS|HBS|SES|EHIS|ECHP|null>",
  "eurostat_variables": [
    {{
      "survey": "<EU_SILC|LFS|HBS|SES|EHIS|ECHP>",
      "variable_code": "<e.g. PL031>",
      "variable_label": "<e.g. Self-defined current economic status>",
      "variable_value": "<EXACT CODE FROM DICTIONARY>",
      "value_label": "<what this value means>",
      "confidence": "<high|medium|low>",
      "rationale": "<one sentence>"
    }}
  ],
  "secondary_surveys": [],
  "filter_conditions_combined": "<e.g. RB081 < 30 AND PL031 = 1 or null>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# RETRY WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def _is_retryable(exc) -> bool:
    if isinstance(exc, anthropic.RateLimitError) or (isinstance(exc, anthropic.APIStatusError) and exc.status_code == 529):
        return True
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return True
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    return False


_retry = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_exponential(multiplier=2, min=10, max=300),
    stop=tenacity.stop_after_attempt(8),
    before_sleep=lambda rs: print(
        f"    ⚠  Haiku retry {rs.attempt_number}/4 — "
        f"{rs.outcome.exception().__class__.__name__}"
    ),
)


@_retry
def _call_api(client: anthropic.Anthropic, messages: list) -> anthropic.types.Message:
    return client.messages.create(
        model=MODEL_MAPPING,
        max_tokens=1_000,
        messages=messages,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HAIKU CALL
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC HBS EXPENDITURE MAPPING
# When extraction.py sets target_expenditure_category, we skip Haiku entirely.
# HBS uses COICOP codes as variable identifiers (02.2=Tobacco, 04.5=Energy etc.)
# ─────────────────────────────────────────────────────────────────────────────

_HBS_COICOP_MAP = {
    # 01 – Food and non-alcoholic beverages
    "01":     ("Total food and non-alcoholic beverages expenditure",   "Food and non-alcoholic beverages"),
    "01.1":   ("Food expenditure",                                     "Food"),
    "01.1.1": ("Bread and cereals expenditure",                        "Bread, rice, pasta, cereals"),
    "01.1.2": ("Meat expenditure",                                     "All types of meat"),
    "01.1.3": ("Fish and seafood expenditure",                         "Fresh, frozen, preserved fish"),
    "01.1.4": ("Milk, cheese and eggs expenditure",                    "Dairy products and eggs"),
    "01.1.5": ("Oils and fats expenditure",                            "Butter, margarine, oils"),
    "01.1.6": ("Fruit expenditure",                                    "Fresh, frozen, dried fruit"),
    "01.1.7": ("Vegetables expenditure",                               "Fresh, frozen, preserved vegetables"),
    "01.1.8": ("Sugar, jam, honey, chocolate and confectionery expenditure", "Sugar, sweets, chocolate"),
    "01.1.9": ("Food products n.e.c. expenditure",                     "Other food products"),
    "01.2":   ("Non-alcoholic beverages expenditure",                  "Coffee, tea, soft drinks, mineral water"),
    # 02 – Alcoholic beverages, tobacco and narcotics
    "02":     ("Total alcoholic beverages and tobacco expenditure",    "Alcoholic beverages and tobacco"),
    "02.1":   ("Alcoholic beverages expenditure",                      "Spirits, wine, beer"),
    "02.2":   ("Tobacco expenditure",                                  "Cigarettes, cigars, tobacco"),
    # 03 – Clothing and footwear
    "03":     ("Total clothing and footwear expenditure",              "Clothing and footwear"),
    "03.1":   ("Clothing expenditure",                                 "Garments, clothing materials, services"),
    "03.2":   ("Footwear expenditure",                                 "Shoes and repair services"),
    # 04 – Housing, water, electricity, gas and other fuels
    "04":     ("Total housing, water, electricity, gas expenditure",   "Housing, water, electricity, gas"),
    "04.1":   ("Actual rentals for housing",                           "Rent payments"),
    "04.2":   ("Imputed rentals for housing",                          "Imputed rent for owner-occupiers"),
    "04.3":   ("Maintenance and repair of the dwelling expenditure",   "Maintenance, repairs, materials"),
    "04.4":   ("Water supply and miscellaneous services expenditure",  "Water, refuse, sewerage"),
    "04.5":   ("Electricity, gas and other fuels expenditure",         "Electricity, gas, heating fuels"),
    # 05 – Furnishings, household equipment and routine maintenance
    "05":     ("Total furnishings and household equipment expenditure", "Furnishings and household equipment"),
    "05.1":   ("Furniture and furnishings, carpets expenditure",       "Furniture, carpets, repair"),
    "05.2":   ("Household textiles expenditure",                       "Curtains, bedding, linen"),
    "05.3":   ("Household appliances expenditure",                     "Fridges, washing machines, repair"),
    "05.4":   ("Glassware, tableware and household utensils expenditure", "Glassware, china, cutlery, kitchenware"),
    "05.5":   ("Tools and equipment for house and garden expenditure",  "Tools, equipment, repairs"),
    "05.6":   ("Goods and services for routine household maintenance",  "Cleaning products, household services"),
    # 06 – Health
    "06":     ("Total health expenditure",                             "Health"),
    "06.1":   ("Medical products, appliances and equipment expenditure","Pharmaceuticals, medical appliances"),
    "06.2":   ("Outpatient services expenditure",                      "Medical, dental, paramedical services"),
    "06.3":   ("Hospital services expenditure",                        "Hospital, nursing home services"),
    # 07 – Transport
    "07":     ("Total transport expenditure",                          "Transport"),
    "07.1":   ("Purchase of vehicles expenditure",                     "Cars, motorcycles, bicycles"),
    "07.2":   ("Operation of personal transport equipment expenditure", "Fuel, maintenance, repairs, parts"),
    "07.3":   ("Transport services expenditure",                       "Public transport, taxi, other services"),
    # 08 – Communication
    "08":     ("Total communication expenditure",                      "Communication"),
    "08.1":   ("Postal services expenditure",                          "Postage"),
    "08.2":   ("Telephone and telefax equipment expenditure",          "Phone purchase, repair"),
    "08.3":   ("Telephone and telefax services expenditure",           "Phone bills, subscriptions"),
    # 09 – Recreation and culture
    "09":     ("Total recreation and culture expenditure",             "Recreation and culture"),
    "09.1":   ("Audio-visual, photographic and information processing equipment expenditure", "TVs, computers, cameras"),
    "09.2":   ("Other major durables for recreation and culture expenditure", "Sports equipment, campers, boats"),
    "09.3":   ("Other recreational items, gardens, pets expenditure",  "Games, toys, hobby materials, pets"),
    "09.4":   ("Recreational and cultural services expenditure",       "Cinema, sports, TV subscriptions"),
    "09.5":   ("Newspapers, books and stationery expenditure",         "Books, newspapers, stationery"),
    "09.6":   ("Package holidays expenditure",                         "All-inclusive holidays"),
    # 10 – Education
    "10":     ("Total education expenditure",                          "Education"),
    "10.1":   ("Pre-primary and primary education expenditure",        "Primary school fees, books, materials"),
    "10.2":   ("Secondary education expenditure",                      "Secondary school fees, books, materials"),
    "10.3":   ("Post-secondary non-tertiary education expenditure",    "Post-secondary fees, books, materials"),
    "10.4":   ("Tertiary education expenditure",                       "University fees, books, materials"),
    "10.5":   ("Education not definable by level expenditure",         "Adult education, other courses"),
    # 11 – Restaurants and hotels
    "11":     ("Total restaurants and hotels expenditure",             "Restaurants and hotels"),
    "11.1":   ("Catering services expenditure",                        "Restaurants, cafes, canteens"),
    "11.2":   ("Accommodation services expenditure",                   "Hotels, hostels, campsites"),
    # 12 – Miscellaneous goods and services
    "12":     ("Total miscellaneous goods and services expenditure",   "Miscellaneous goods and services"),
    "12.1":   ("Personal care expenditure",                            "Hairdressing, toiletries, cosmetics"),
    "12.3":   ("Personal effects n.e.c. expenditure",                  "Jewelry, watches, luggage"),
    "12.4":   ("Social protection expenditure",                        "Homes for elderly, disabled, childcare"),
    "12.5":   ("Insurance expenditure",                                "Life, health, home insurance"),
    "12.6":   ("Financial services n.e.c. expenditure",                "Bank charges, fees"),
    "12.7":   ("Other services n.e.c. expenditure",                    "Legal services, funeral, other"),
}

_ZERO_USAGE = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}


def _map_expenditure_hbs(target_details: dict):
    """
    Deterministic HBS mapping for expenditure-based (indirect tax) measures.
    Reads target_expenditure_category set by _fix_expenditure_target().
    Returns (result_dict, usage_dict) or None if field is absent.
    No API call -- pure lookup, zero cost.
    """
    exp_cat = str(target_details.get("target_expenditure_category", "")).strip()
    if not exp_cat or exp_cat in ("NA", "null", ""):
        return None

    parts = exp_cat.split("|", 1)
    code  = parts[0].strip()
    label = parts[1].strip() if len(parts) > 1 else code

    hbs_entry = _HBS_COICOP_MAP.get(code)
    if hbs_entry is None:
        parent    = code.split(".")[0]
        hbs_entry = _HBS_COICOP_MAP.get(parent, ("Miscellaneous goods and services expenditure","Miscellaneous"))

    var_label, section = hbs_entry

    result = {
        "eurostat_mapping_found": True,
        "primary_survey":         "HBS",
        "eurostat_variables": [{
            "survey":         "HBS",
            "variable_code":  code,
            "variable_label": var_label,
            "variable_value": ">0",
            "value_label":    f"Household has positive expenditure on {label}",
            "confidence":     "high",
            "rationale": (
                f"Indirect tax on {label}: target population is all households with "
                f"positive expenditure in HBS COICOP category {code} ({section}). "
                f"Filter: HBS.{code} > 0"
            ),
        }],
        "secondary_surveys":          ["EU-SILC"],
        "filter_conditions_combined": f"HBS.{code} > 0",
    }
    print(f"      -> HBS deterministic: COICOP {code} ({label})")
    return result, dict(_ZERO_USAGE)


def map_to_eurostat(
    target_summary: str,
    target_details: dict,
    cached_dict_prefix: list,
    client: anthropic.Anthropic,
    country: str = "ITA",
) -> tuple[dict, dict]:
    """
    Calls Haiku to map a target population to Eurostat survey variables.

    Returns:
        (mapping_result, usage_dict)
    """
    # Fast path: expenditure-based measures -> deterministic HBS, no Haiku call
    hbs_result = _map_expenditure_hbs(target_details)
    if hbs_result is not None:
        return hbs_result

    def _fmt(key: str) -> str:
        v = target_details.get(key)
        return str(v) if v not in (None, "", "null", "NA") else "NA"

    user_content = MAPPING_PROMPT_TEMPLATE.format(
        target_summary=target_summary,
        target_age_min=_fmt("target_age_min"),
        target_age_max=_fmt("target_age_max"),
        target_income_min=_fmt("target_income_min"),
        target_income_max=_fmt("target_income_max"),
        target_income_type=_fmt("target_income_type"),
        target_employment_status=_fmt("target_employment_status"),
        target_household_type=_fmt("target_household_type"),
        target_sector=_fmt("target_sector"),
        target_geographic=_fmt("target_geographic"),
        target_other_criteria=_fmt("target_other_criteria"),
    )

    messages = [{
        "role": "user",
        "content": cached_dict_prefix + [{"type": "text", "text": user_content}],
    }]

    response = _call_api(client, messages)

    usage = {
        "input":          response.usage.input_tokens,
        "output":         response.usage.output_tokens,
        "cache_creation": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read":     getattr(response.usage, "cache_read_input_tokens",     0),
    }

    raw = re.sub(
        r"^```json\s*|\s*```$", "",
        response.content[0].text.strip(),
        flags=re.MULTILINE,
    )

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    WARNING: JSON parse error (mapping) — {e}")
        result = {
            "eurostat_mapping_found": False,
            "primary_survey": "NA",
            "eurostat_variables": [],
            "secondary_surveys": [],
            "filter_conditions_combined": "NA",
        }

    # Regional override: add DB040 if geographic targeting detected
    result = _apply_regional_mapping(result, target_details, country)

    return result, usage


# ─────────────────────────────────────────────────────────────────────────────
# REGIONAL MAPPING  (mechanical — no Claude needed)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_regional_mapping(
    result: dict, target_details: dict, country: str
) -> dict:
    """
    If geographic targeting is present and matches a known NUTS-2 region,
    adds DB040 to the mapping. Runs after the Haiku call.
    """
    target_geo = str(target_details.get("target_geographic", "")).lower()
    if not target_geo or target_geo in ("na", "null", "none", ""):
        return result

    region_map = REGIONAL_MAPPINGS.get(country.upper(), {})
    for region_name, nuts_code in region_map.items():
        if region_name in target_geo:
            already_mapped = any(
                v.get("variable_code") == "DB040"
                for v in result.get("eurostat_variables", [])
            )
            if not already_mapped:
                result.setdefault("eurostat_variables", []).append({
                    "survey":         "EU_SILC",
                    "variable_code":  "DB040",
                    "variable_label": "Region (NUTS 2)",
                    "variable_value": nuts_code,
                    "value_label":    region_name.title(),
                    "confidence":     "high",
                    "rationale":      f"Geographic targeting: {region_name}",
                })
                result["eurostat_mapping_found"] = True
                result.setdefault("primary_survey", "EU_SILC")
                print(f"      → Regional mapping: {region_name.title()} → DB040={nuts_code}")
            break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FORCE OBVIOUS MAPPINGS  (mechanical overrides / supplements)
# ─────────────────────────────────────────────────────────────────────────────
#
# Haiku is sometimes too conservative and skips mandatory variables even when
# the criterion is unambiguous. Each block below fires independently, checks
# whether the variable is already present, and adds it if not.
#
# To add a new mandatory variable: copy an existing block and adapt.
# ─────────────────────────────────────────────────────────────────────────────

_EMPLOYMENT_CODES = {
    "unemployed": ("2", "Unemployed"),
    "retired":    ("3", "In retirement"),
    "pension":    ("3", "In retirement"),
    "student":    ("5", "In education or training"),
    "education":  ("5", "In education or training"),
}


def _already_has(result: dict, *variable_codes: str) -> bool:
    return any(
        v.get("variable_code") in variable_codes
        for v in result.get("eurostat_variables", [])
    )


def force_obvious_mappings(target_details: dict, result: dict) -> dict:
    """
    Deterministically adds mandatory Eurostat variables that Haiku may have
    missed. Modifies `result` in place and returns it.

    Mandatory variable rules:
      ─ Age criterion      → RB081
      ─ Employment status  → RB211
      ─ Income threshold   → HY020
    """
    forced: list[dict] = []

    # ── Age → RB081 ──────────────────────────────────────────────────────────
    age_min = target_details.get("target_age_min")
    age_max = target_details.get("target_age_max")
    has_age = (
        pd.notna(age_min) and str(age_min) not in ("NA", "null", "None", "")
        or
        pd.notna(age_max) and str(age_max) not in ("NA", "null", "None", "")
    )
    if has_age and not _already_has(result, "RB081", "AGE"):
        try:
            value = f"<={int(float(age_max))}" if pd.notna(age_max) else f">={int(float(age_min))}"
        except (TypeError, ValueError):
            value = "see notes"
        forced.append({
            "survey":         "EU_SILC",
            "variable_code":  "RB081",
            "variable_label": "Age",
            "variable_value": value,
            "value_label":    f"Age {value}",
            "confidence":     "high",
            "rationale":      "Forced: age criterion present in target population",
        })

    # ── Employment → RB211 ───────────────────────────────────────────────────
    emp = str(target_details.get("target_employment_status", "")).lower()
    if emp and emp not in ("na", "null", "none", "all", ""):
        if not _already_has(result, "RB211", "PL031", "ILOSTAT", "MAINSTAT", "EMPSTAT"):
            code, label = ("1", "Employed")
            for key, (c, lbl) in _EMPLOYMENT_CODES.items():
                if key in emp:
                    code, label = c, lbl
                    break
            forced.append({
                "survey":         "EU_SILC",
                "variable_code":  "RB211",
                "variable_label": "Main activity status (self-defined)",
                "variable_value": code,
                "value_label":    label,
                "confidence":     "high",
                "rationale":      f"Forced: employment status '{emp[:40]}'",
            })

    # ── Income → HY020 ───────────────────────────────────────────────────────
    inc_max = target_details.get("target_income_max")
    has_income = (
        inc_max not in (None, "NA", "", "null")
        and str(inc_max) not in ("NA", "null", "None", "")
    )
    if has_income and not _already_has(result, "HY020", "HY010"):
        try:
            threshold = int(float(inc_max))
            forced.append({
                "survey":         "EU_SILC",
                "variable_code":  "HY020",
                "variable_label": "Total disposable household income",
                "variable_value": f"<={threshold}",
                "value_label":    f"Income ≤ {threshold:,}",
                "confidence":     "high",
                "rationale":      "Forced: income threshold present in target population",
            })
        except (TypeError, ValueError):
            pass

    # ── Apply forced variables ────────────────────────────────────────────────
    if forced:
        if not result.get("eurostat_mapping_found"):
            # Haiku returned nothing useful — use forced vars as the full mapping
            result["eurostat_mapping_found"] = True
            result["eurostat_variables"]     = forced
            result["primary_survey"]         = "EU_SILC"
            result["filter_conditions_combined"] = " AND ".join(
                f"{v['variable_code']} {v['variable_value']}" for v in forced[:3]
            )
            print(f"      → Forced mapping: {len(forced)} variable(s) added (Haiku returned nothing)")
        else:
            result["eurostat_variables"].extend(forced)
            print(f"      → Enhanced mapping: {len(forced)} additional variable(s) added")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FLAT EUROSTAT VALUE COLUMNS  (for Excel output)
# ─────────────────────────────────────────────────────────────────────────────

def build_eurostat_value_columns(eurostat_variables: list[dict]) -> dict[str, str]:
    """
    Converts the eurostat_variables list into flat column-name → value pairs
    for direct insertion into the output DataFrame.

    Column name format: {SURVEY}_{VARIABLE_CODE}
    e.g. EUSILC_HY020, LFS_PL111
    """
    cols: dict[str, str] = {}
    for var in eurostat_variables[:10]:
        survey = var.get("survey", "UNKNOWN").replace("_", "")
        code   = var.get("variable_code", "VAR")
        value  = var.get("variable_value", "NA")
        cols[f"{survey}_{code}"] = value
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY MAPPING RESULT  (used when target_population_found is False)
# ─────────────────────────────────────────────────────────────────────────────

EMPTY_MAPPING: dict = {
    "eurostat_mapping_found":     False,
    "primary_survey":             "NA",
    "eurostat_variables":         [],
    "secondary_surveys":          [],
    "filter_conditions_combined": "NA",
}