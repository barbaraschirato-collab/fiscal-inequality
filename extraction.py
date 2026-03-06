"""
extraction.py
─────────────────────────────────────────────────────────────────────────────
Sonnet-based document extraction: prompt template, API call, and
post-processing of raw JSON into a clean result dict.

To update the extraction prompt or post-processing logic: edit only this file.
The model name lives in config.MODEL_MAIN.
─────────────────────────────────────────────────────────────────────────────
"""

import re
import json
import anthropic
import tenacity

from config import (
    MODEL_MAIN,
    MOTIVATION_CATEGORIES_TEXT,
    MOTIVATION_TAXONOMY,
    CATEGORY_DEFAULTS,
)
from chunking import (
    extract_law_identifiers,
    find_measure_law_in_table_pages,
)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# [FIX-2] NOTE removed from inside JSON block — moved into TASK section above.
# Editing the prompt: change only the string below; nothing else needs updating.
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT_TEMPLATE = """
══════════════════════════════════════════
REFERENCE EXAMPLE
══════════════════════════════════════════
{example_text}

══════════════════════════════════════════
MEASURE TO ANALYSE
══════════════════════════════════════════
Country: {country} | Year: {year}
Category: {category1} / {category2}
Fiscal measure (exact AFG col E text): "{measure}"

══════════════════════════════════════════
RELEVANT SOURCE EXCERPT
══════════════════════════════════════════
IMPORTANT: Information about this measure may be split across multiple documents.
- Document A (e.g. DEF) might announce the measure exists
- Document B (e.g. NADEF) might contain implementation details (income thresholds!)
- Document C (e.g. a law decree) might have the full legal text
READ ALL DOCUMENT EXCERPTS BELOW. The best information might be in the 2nd or 3rd document.

{source_chunk}

══════════════════════════════════════════
MOTIVATION TAXONOMY
══════════════════════════════════════════
{motivation_categories}

══════════════════════════════════════════
TASK
══════════════════════════════════════════
Italian fiscal terminology:
- IRPEF=income tax, IVA=VAT, IMU=property tax
- pensioni=pensions, lavoratori dipendenti=employees, reddito=income
- famiglie numerose=large families, autonomi=self-employed
Legal references: L=Legge, DL=Decreto Legge, D.Lgs.=Decreto Legislativo, Art.=Articolo

1. FIND the measure in the excerpt and EXTRACT legal references.
   Combine information across multiple documents.

2. TARGET POPULATION — extract ALL eligibility criteria, especially income thresholds.

   INCOME THRESHOLDS (critical):
   Italian documents often present tiered benefits, for example:
     "detrazione pari a 900 euro (per redditi non superiori a 15.493 euro)
      e 450 euro (per redditi non superiori a 30.987 euro)"
   This means TWO thresholds: 15,493 and 30,987.
   - Use the HIGHEST threshold as target_income_max (30987).
   - Use 0 as target_income_min when only upper bounds appear.
   - Threshold phrases: "per redditi non superiori a", "fino a", "inferiori a", "oltre"
   - Set target_population_found = true ONLY if specific eligibility criteria exist.
   - Universal measures (no specific eligibility) → target_population_found = false,
     all target fields = null.

3. MOTIVATION — the government's STATED reason.
   - motivation_specific: 2-3 KEY CONCEPTS separated by slashes.
     E.g. "Social support / Housing emergency" or "Fiscal consolidation / EU compliance"
   - motivation_category: SINGLE integer 1-7 from the taxonomy above.
   - NEITHER field is ever null — infer from category/context if not explicit.

Return ONLY valid JSON with no markdown, no comments inside the JSON block:
{{
  "source_document_found": true,
  "source_document_name": "<filename with most details or null>",
  "source_page_reference": "<Page X, Section Y or null>",
  "source_excerpt_original": "<verbatim text with most details, max 2-3 sentences, or null>",
  "source_law_identifier": "<e.g. L 147/2013, DL 16/2014, Art. 5 or null>",

  "target_population_found": true,
  "target_population_summary_en": "<English description or null>",
  "target_age_min": null,
  "target_age_max": null,
  "target_income_min": null,
  "target_income_max": null,
  "target_income_type": "<gross|net|disposable|null>",
  "target_employment_status": "<employed|unemployed|self-employed|retired|all|null>",
  "target_household_type": "<description or null>",
  "target_sector": "<description or null>",
  "target_geographic": "<description or null>",
  "target_other_criteria": "<benefit tiers or additional criteria or null>",

  "motivation_specific": "<2-3 concepts separated by slashes>",
  "motivation_category": 1,

  "extraction_confidence_raw": "<high|medium|low>",
  "extraction_notes": "<max 15 words, empty string if extraction was clean>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# RETRY WRAPPER  [FIX-3]
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
        f"    ⚠  Sonnet retry {rs.attempt_number}/4 — "
        f"{rs.outcome.exception().__class__.__name__}"
    ),
)


@_retry
def _call_api(client: anthropic.Anthropic, messages: list) -> anthropic.types.Message:
    return client.messages.create(
        model=MODEL_MAIN,
        max_tokens=1500,
        messages=messages,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: BRUTE-FORCE INCOME/AGE EXTRACTION
# Runs before Claude if possible (inverted priority from v1).
# Also runs after Claude as a safety net.
# ─────────────────────────────────────────────────────────────────────────────

_INCOME_STOPWORDS = {
    'della', 'sulla', 'nelle', 'dalle', 'degli', 'delle',
    'questa', 'quello', 'quale', 'quando', 'dove', 'fatto', 'tutti',
}


def _regex_extract_income(measure: str, source_docs: dict[str, str]) -> dict | None:
    """
    Mechanical regex pass over all source documents.
    Returns {target_income_min, target_income_max} or None if nothing found.
    """
    terms = [
        w for w in re.findall(r'\b\w{5,}\b', measure)
        if w.lower() not in _INCOME_STOPWORDS
    ][:5]
    if not terms:
        return None

    for doc_text in source_docs.values():
        for term in terms:
            for m in list(re.finditer(re.escape(term), doc_text, re.IGNORECASE))[:3]:
                start = max(0, m.start() - 1_500)
                end   = min(len(doc_text), m.end() + 1_500)
                window = doc_text[start:end]

                if not re.search(r'redditi|reddito|income', window, re.IGNORECASE):
                    continue

                amounts = re.findall(
                    r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*euro',
                    window, re.IGNORECASE,
                )
                numbers = []
                for amt in amounts:
                    try:
                        numbers.append(float(amt.replace('.', '').replace(',', '.')))
                    except ValueError:
                        pass

                if numbers:
                    mx = max((n for n in numbers if 1_000 <= n <= 500_000), default=None)
                    if mx:
                        return {"target_income_min": 0, "target_income_max": int(mx)}
    return None


def enrich_with_fallback_numbers(
    measure: str, source_docs: dict[str, str], extracted: dict
) -> dict:
    """
    If Claude found income-based targeting but did not extract the threshold
    number, attempt a mechanical regex pass to recover it.

    Only fires when BOTH conditions are true:
      1. target_population_found is True  (Claude confirmed income targeting exists)
      2. target_income_max is missing     (Claude missed the specific number)

    Without condition 1, the fallback fires on measures with no income criterion
    at all -- picking up spurious amounts like the IRPEF bracket ceiling
    (150,000 euro) from nearby fiscal tables, producing false positives.
    """
    # Condition 1: Claude must have confirmed income-based targeting exists
    income_targeting = str(extracted.get("target_income_type", "")).strip().lower()
    target_found     = extracted.get("target_population_found") is True
    has_income_field = any(
        str(extracted.get(k, "")).strip().lower() not in ("", "na", "null", "none")
        for k in ("target_income_type", "target_income_min", "target_income_max")
    )
    if not (target_found and has_income_field):
        return extracted

    # Condition 2: The specific number is missing
    has_income = extracted.get("target_income_max") not in (None, "NA", "", "null")
    if has_income:
        return extracted

    result = _regex_extract_income(measure, source_docs)
    if result:
        extracted.update(result)
        extracted["extraction_notes"] = (
            (str(extracted.get("extraction_notes", "")) +
             f" | Fallback income: €{result['target_income_max']}").strip(" | ")
        )
        print(f"      → Fallback income: €{result['target_income_max']}")

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fix_motivation(data: dict, category1: str, category2: str) -> dict:
    """Guarantees motivation_specific and motivation_category are never null."""
    ms = str(data.get("motivation_specific", "")).lower()
    mc = data.get("motivation_category")

    if not ms or ms in ("na", "null", "none", ""):
        cat_key = (category2 or category1).strip()
        label, code = CATEGORY_DEFAULTS.get(
            cat_key, ("Fiscal adjustment / Budget measures", 1)
        )
        data["motivation_specific"] = label
        data["motivation_category"] = code
        data["extraction_notes"] = (
            (str(data.get("extraction_notes", "")) + " | Motivation inferred").strip(" | ")
        )
    else:
        # Validate the category integer
        try:
            mc_int = int(mc)
            data["motivation_category"] = mc_int if 1 <= mc_int <= 7 else 7
        except (TypeError, ValueError):
            data["motivation_category"] = 7

    return data


def _fix_law_identifier(
    data: dict, measure: str,
    source_docs: dict[str, str], afg_source: str,
    country: str,
) -> dict:
    """Fills source_law_identifier using table-page search, then regex fallback."""
    law = find_measure_law_in_table_pages(measure, source_docs, afg_source, country)
    if law and law != "NA":
        data["source_law_identifier"] = law
        return data

    excerpt = data.get("source_excerpt_original") or ""
    if excerpt not in ("NA", None, "", "null"):
        auto = extract_law_identifiers(str(excerpt), country)
        if auto != "NA":
            data["source_law_identifier"] = auto

    return data


def _truncate_notes(data: dict, max_chars: int = 100) -> dict:
    notes = str(data.get("extraction_notes", ""))
    if len(notes) > max_chars:
        data["extraction_notes"] = notes[:max_chars - 3] + "..."
    return data


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


# COICOP keyword matcher for indirect tax measures.
# Order matters: more specific entries must come before broader ones
# (e.g. fuel/operation 07.2 before vehicle purchase 07.1, bread before food).
# Each tuple: (keyword_list, COICOP_code, HBS_label)
_EXPENDITURE_KEYWORDS = [

    # ── PRIORITY GUARDS: must come first to prevent substring matches ─────────
    # 'gasolio' and 'benzina' contain 'olio' and 'enz' which could match food;
    # 'televisore' must hit 09.1 (equipment) not fall through to generic tv/09.4
    (['gasolio','benzina','diesel','carburant','accise sui carburant',
      'accise energet'],                                   '07.2',   'Operation of personal transport equipment'),
    (['televisore','schermo tv','monitor','display'],       '09.1',   'Audio-visual equipment'),

    # ── 01 Food and non-alcoholic beverages ──────────────────────────────────
    # Sub-categories first, then the division
    (['pane','bread','cereali','cereals','pasta','riso','rice','farina'],
                                                        '01.1.1', 'Bread and cereals'),
    (['carne','meat','bovino','suino','pollame','poultry','salumi','macellar'],
                                                        '01.1.2', 'Meat'),
    (['pesce','fish','seafood','prodotti ittici'],       '01.1.3', 'Fish and seafood'),
    (['latte','milk','formaggio','cheese','uova','eggs','latticin','dairy'],
                                                        '01.1.4', 'Milk, cheese and eggs'),
    (['olio','oli','fats','grassi','burro','butter','margarina'],
                                                        '01.1.5', 'Oils and fats'),
    (['frutta','fruit','agrumi'],                        '01.1.6', 'Fruit'),
    (['verdura','vegetabl','ortaggi','legumi'],          '01.1.7', 'Vegetables'),
    (['zucchero','sugar','dolciumi','cioccolat','confection','marmellat',
      'jam','honey','miele'],                           '01.1.8', 'Sugar, jam, honey, chocolate'),
    (['bevande analcolich','soft drink','sugar tax','succhi','acque mineral',
      'the','caffe','coffee','tea'],                    '01.2',   'Non-alcoholic beverages'),
    (['aliment','food','generi alimentari','prodotti alimentari',
      'iva aliment','beni alimentari'],                 '01.1',   'Food'),
    (['iva ridotta','aliquota ridotta','beni di prima necessita',
      'first necessity','paniere'],                     '01',     'Food and non-alcoholic beverages'),

    # ── 02 Alcoholic beverages and tobacco ───────────────────────────────────
    (['tabacch','tobacco','sigarett','cigarett','sigaro','cigar',
      'prodotti del tabacco'],                          '02.2',   'Tobacco'),
    (['alcol','alcohol','birra','beer','vino','wine','spirit','grappa',
      'whisky','liquore','alcolici','distillati','bevande alcolich'],
                                                        '02.1',   'Alcoholic beverages'),

    # ── 03 Clothing and footwear ─────────────────────────────────────────────
    (['scarpe','footwear','calzatur'],                   '03.2',   'Footwear'),
    (['abbigliament','vestiario','clothing','tessil','tessuto','garment'],
                                                        '03.1',   'Clothing'),
    (['moda','fashion'],                                 '03',     'Clothing and footwear'),

    # ── 04 Housing, water, electricity, gas ──────────────────────────────────
    (['affitt','locazion','canone locazion','canone affitto','rent','rental','imu','tasi',
      'property tax','tassa rifiut','abitazion','immobil'],
                                                        '04.1',   'Actual rentals for housing'),
    (['manutenzione','riparazioni','repair','ristrutturazion','renovation'],
                                                        '04.3',   'Maintenance and repair of dwelling'),
    (['acqua','water','acquedott','idric','fognatur','rifiuti','refuse','tari',
      'sewerage','tassa rifiut','raccolta rifiut'],                                      '04.4',   'Water supply and miscellaneous services'),
    (['elettricit','electric','gas natural','energia termica','teleriscald',
      'riscaldamento','heating','luce','metano','gas domestico'],
                                                        '04.5',   'Electricity, gas and other fuels'),
    (['energia','energy','combustibil','fuel'],          '04.5',   'Electricity, gas and other fuels'),
    (['casa','housing','abitativ','immobil'],             '04',     'Housing, water, electricity, gas'),

    # ── 05 Furnishings and household equipment ────────────────────────────────
    (['mobili','mobilio','furniture','arredament','tappeti','carpet'],
                                                        '05.1',   'Furniture and furnishings, carpets'),
    (['biancheria','lenzuola','curtains','tende','household textil'],
                                                        '05.2',   'Household textiles'),
    (['elettrodomestici','appliance','lavatrice','frigorifero','fridge',
      'lavastoviglie'],                                 '05.3',   'Household appliances'),
    (['stoviglie','glassware','pentol','tableware','utensil','posate'],
                                                        '05.4',   'Glassware, tableware and household utensils'),
    (['attrezzi','tools','garden','giardino'],            '05.5',   'Tools and equipment for house and garden'),
    (['pulizia','cleaning','detergent','household maintenance'],
                                                        '05.6',   'Goods and services for routine household maintenance'),

    # ── 06 Health ─────────────────────────────────────────────────────────────
    (['ospedale','hospital','ricovero','degenza','nursing home'],
                                                        '06.3',   'Hospital services'),
    (['visita medica','outpatient','ambulatorio','dentist','dentista',
      'paramedic','fisioterapia'],                      '06.2',   'Outpatient services'),
    (['farmac','medicinali','medicine','pharmaceutical','ticket sanitario',
      'copayment','dispositivi medici','medical device','protesi'],
                                                        '06.1',   'Medical products, appliances and equipment'),
    (['sanit','health','salute','spesa sanitaria'],      '06',     'Health'),

    # ── 07 Transport ─────────────────────────────────────────────────────────
    # 07.2 must come before 07.1: fuel/operation is more specific than vehicle purchase
    (['benzina','gasolio','diesel','petrol','carburant','accise sui carburant',
      'accise energet','bollo auto','assicurazione auto','rc auto',
      'manutenzione veicol','vehicle operation','fuel'],
                                                        '07.2',   'Operation of personal transport equipment'),
    (['acquisto auto','acquisto veicol','auto nuova','autovettur',
      'purchase of vehicle','immatricolazion','tassa immatricolazion',
      'imposta veicol'],                                '07.1',   'Purchase of vehicles'),
    (['trasporto pubblic','biglietti','treno','autobus','metro','ferrovi',
      'aere','airline','taxi','volo','navigazion','public transport'],
                                                        '07.3',   'Transport services'),
    (['autoveicol','veicol','moto','vehicle','trasport'],
                                                        '07',     'Transport'),

    # ── 08 Communication ─────────────────────────────────────────────────────
    (['posta','postale','postal','francobollo'],          '08.1',   'Postal services'),
    (['acquisto telefono','phone purchase','telefax equipment'],
                                                        '08.2',   'Telephone and telefax equipment'),
    (['telefon','telecom','internet','mobile','bolletta telefon',
      'abbonamento telefonico','comunicazion'],         '08.3',   'Telephone and telefax services'),

    # ── 09 Recreation and culture ────────────────────────────────────────────
    (['canone rai','canone tv','abbonamento rai','abbonamento televisiv',
      'licenza televisiv'],                             '09.4',   'Recreational and cultural services'),
    (['tv','television','televisione','computer','fotografi','audio-visual',
      'hi-fi','videocamera'],                           '09.1',   'Audio-visual equipment'),
    (['barca','boat','camper','sports equipment','attrezzatura sportiva'],
                                                        '09.2',   'Major durables for recreation'),
    (['giocattol','toys','hobby','animali domestici','pets','giardino'],
                                                        '09.3',   'Recreational items, gardens, pets'),
    (['libro','libri','books','giornali','newspaper','riviste','magazine',
      'stationery','cartoleria'],                       '09.5',   'Newspapers, books and stationery'),
    (['vacanze','holiday','pacchetto vacanze','package holiday','turismo',
      'viaggio'],                                       '09.6',   'Package holidays'),
    (['cinema','teatro','concert','sport','abbonamento tv','canone rai','canone tv',
      'abbonamento rai','streaming','gioco','gambling','betting','video game',
      'videogame','gaming','scommess','lotteria','gratta','slot','lotto',
      'giochi','cultural services','servizi ricreativi'], '09.4',   'Recreational and cultural services'),
    (['svago','recreation','cultura','cultura','intrattenimento'],
                                                        '09',     'Recreation and culture'),

    # ── 10 Education ─────────────────────────────────────────────────────────
    (['universita','university','tertiar','laurea','tasse universitarie'],
                                                        '10.4',   'Tertiary education'),
    (['scuola media','scuola superior','secondary education','liceo'],
                                                        '10.2',   'Secondary education'),
    (['scuola elementare','scuola primaria','asilo','primary education',
      'pre-primary'],                                   '10.1',   'Pre-primary and primary education'),
    (['formazione','adult education','corso','education'],
                                                        '10',     'Education'),

    # ── 11 Restaurants and hotels ─────────────────────────────────────────────
    (['hotel','albergo','alberghi','hostel','campeggio','accommodation'],
                                                        '11.2',   'Accommodation services'),
    (['ristorante','restaurant','bar','caffe','mensa','canteen','catering',
      'ristorazion','fast food'],                       '11.1',   'Catering services'),
    (['turism','tourism','ospitalita'],                  '11',     'Restaurants and hotels'),

    # ── 12 Miscellaneous goods and services ──────────────────────────────────
    (['assicurazion','insurance','polizza','rc','vita assicurativa'],
                                                        '12.5',   'Insurance'),
    (['banca','bank','servizi bancari','commissioni','financial services',
      'finanziari'],                                    '12.6',   'Financial services n.e.c.'),
    (['cura personal','personal care','parrucchier','cosmetici','toiletries',
      'igiene personal'],                               '12.1',   'Personal care'),
    (['protezione sociale','social protection','anziani','disabili',
      'childcare','asilo nido','elderly','disabled'],   '12.4',   'Social protection'),
    (['gioielli','jewelry','orologio','watch','luggage','bagagli',
      'personal effects'],                              '12.3',   'Personal effects n.e.c.'),
    (['servizi legali','legal','funerale','funeral','servizi vari',
      'other services'],                                '12.7',   'Other services n.e.c.'),

    # ── Generic VAT / fallback ────────────────────────────────────────────────
    # Only reached if no specific good was detected above
    (['iva','vat','imposta sul valore aggiunto','aliquota iva',
      'aliquota base','aliquota ordinaria'],            '12',     'Miscellaneous goods and services'),
]

# AFG category1 codes that signal an indirect/expenditure tax.
# ONLY these codes trigger HBS COICOP mapping.
# Do NOT match on category2 keywords -- "Goods and Services" also appears
# as a government expenditure sub-category (capital/current spending),
# which must not receive HBS mapping.
_INDIRECT_CATEGORY_CODES = {'INDT', 'INDT_VAT', 'EXCD', 'EXCISE'}


def _is_indirect_tax(category1: str, category2: str) -> bool:
    """
    True only when AFG category1 explicitly codes an indirect/expenditure tax.
    Matching on category2 keywords like "Goods and Services" is intentionally
    avoided: that label also appears for government spending measures (capital
    expenditure, current expenditure, ministry spending cuts, etc.) which have
    no household expenditure target and must not be mapped to HBS.
    """
    return category1.strip().upper() in _INDIRECT_CATEGORY_CODES


def _detect_coicop(measure, summary):
    """
    Returns (COICOP_code, label) for the best matching expenditure category.
    Uses word-boundary regex so 'oli' does not match inside 'gasolio',
    and 'posta' does not match inside 'imposta'.
    """
    text = (measure + ' ' + summary).lower()
    for keywords, code, label in _EXPENDITURE_KEYWORDS:
        for kw in keywords:
            # word boundary on both sides prevents substring collisions
            pattern = r'(?<![a-z])' + re.escape(kw)
            if re.search(pattern, text):
                return code, label
    return '12', 'Miscellaneous goods and services'


def _fix_expenditure_target(data, category1, category2, measure):
    if not _is_indirect_tax(category1, category2):
        return data
    summary = str(data.get('target_population_summary_en',''))
    code, label = _detect_coicop(measure, summary)
    if not data.get('target_population_found'):
        data['target_population_found'] = True
        data['target_other_criteria'] = f'Expenditure-based: all consumers of {label} (HBS COICOP {code})'
        if not summary.strip():
            data['target_population_summary_en'] = (
                f'All households purchasing {label} -- universal measure applying to all consumers of this good/service'
            )
    data['target_expenditure_category'] = f'{code}|{label}'
    return data


def extract_from_sources(
    measure: str,
    category1: str,
    category2: str,
    source_chunk: str,
    example_text: str,
    cached_prefix: list,
    client: anthropic.Anthropic,
    country: str = "ITA",
    year: str = "2014",
    afg_source: str = "NA",
    source_docs: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """
    Calls Sonnet to extract target population and motivation for one measure.

    Returns:
        (result_dict, usage_dict)

    result_dict contains all extracted fields plus post-processed law identifier
    and guaranteed non-null motivation fields.
    usage_dict contains token counts for cost tracking.
    """
    source_docs = source_docs or {}

    user_content = EXTRACTION_PROMPT_TEMPLATE.format(
        example_text=example_text[:2_500],
        country=country,
        year=year,
        category1=category1,
        category2=category2,
        measure=measure,
        source_chunk=source_chunk,
        motivation_categories=MOTIVATION_CATEGORIES_TEXT,
    )

    messages = [{
        "role": "user",
        "content": cached_prefix + [{"type": "text", "text": user_content}],
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
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    WARNING: JSON parse error (extraction) — {e}")
        data = {
            "source_document_found":    False,
            "target_population_found":  False,
            "source_law_identifier":    "NA",
            "motivation_specific":      "Fiscal adjustment / Budget measures",
            "motivation_category":      7,
            "extraction_confidence_raw": "low",
            "extraction_notes":         f"JSON error: {str(e)[:60]}",
        }
        return data, usage

    # Post-processing (order matters)
    data = _fix_motivation(data, category1, category2)
    data = _fix_expenditure_target(data, category1, category2, measure)
    data = _fix_law_identifier(data, measure, source_docs, afg_source, country)
    data = _truncate_notes(data)

    return data, usage