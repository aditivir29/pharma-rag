"""
PharmaRAG - Step 1: FDA Drug Label Ingestion

Downloads official drug labels from the openFDA API,
extracts safety-relevant sections, cleans the text,
and stores one JSON record per drug/section.
"""

import json
import re
import time
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://api.fda.gov/drug/label.json"
OUT_PATH = Path("data/labels.jsonl")


# ============================================================
# FDA LABEL SECTIONS
# ============================================================

SECTIONS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "overdosage": "Overdosage",
}


# ============================================================
# DRUG LIST
# ============================================================

DRUGS = [
    "metformin",
    "atorvastatin",
    "amlodipine",
    "lisinopril",
    "losartan",
    "omeprazole",
    "pantoprazole",
    "levothyroxine",
    "warfarin",
    "clopidogrel",
    "aspirin",
    "ibuprofen",
    "naproxen",
    "acetaminophen",
    "diclofenac",
    "amoxicillin",
    "azithromycin",
    "ciprofloxacin",
    "doxycycline",
    "cephalexin",
    "sertraline",
    "escitalopram",
    "fluoxetine",
    "alprazolam",
    "zolpidem",
    "gabapentin",
    "tramadol",
    "prednisone",
    "montelukast",
    "cetirizine",
    "metoprolol",
    "atenolol",
    "furosemide",
    "hydrochlorothiazide",
    "simvastatin",
    "glimepiride",
    "insulin glargine",
    "albuterol",
    "loratadine",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean FDA label text without altering valid Unicode
    characters such as ≥, ≤, µ, or °.
    """

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # Remove obvious leading section headers such as:
    #
    # 1 INDICATIONS AND USAGE
    # 5.1 LACTIC ACIDOSIS
    #
    text = re.sub(
        r"^(\d+(\.\d+)*\s*)?[A-Z][A-Z &,/\-]{6,}\s+"
        r"(?=[A-Z0-9a-z])",
        "",
        text,
    )

    return text


# ============================================================
# FETCH BEST LABEL
# ============================================================

def fetch_best_label(drug: str):
    """
    Fetch up to 5 FDA labels matching the generic drug name.

    Select the label containing the largest number of
    safety-relevant sections.
    """

    params = {
        "search": f'openfda.generic_name:"{drug}"',
        "limit": 5,
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    if not results:
        return None

    # Pick the label with the most useful sections.
    best_label = max(
        results,
        key=lambda label: sum(
            1
            for field in SECTIONS
            if field in label
        ),
    )

    return best_label


# ============================================================
# EXTRACT SECTION RECORDS
# ============================================================

def extract_records(drug: str, label: dict):
    """
    Convert one FDA label into section-level records.
    """

    openfda = label.get(
        "openfda",
        {},
    )

    # Brand name from the actual FDA label.
    brand_names = openfda.get(
        "brand_name",
        [drug],
    )

    brand = brand_names[0]

    set_id = label.get(
        "set_id",
        "",
    )

    source_url = (
        "https://dailymed.nlm.nih.gov/dailymed/"
        f"lookup.cfm?setid={set_id}"
    )

    records = []

    for field, section_title in SECTIONS.items():

        raw = label.get(field)

        if not raw:
            continue

        # FDA may provide multiple strings for one section.
        raw_text = " ".join(raw)

        text = clean_text(raw_text)

        # Skip empty/very short sections.
        if len(text) < 50:
            continue

        record = {
            "drug": drug,
            "brand": brand,
            "section": section_title,
            "text": text,
            "set_id": set_id,
            "source_url": source_url,
        }

        records.append(record)

    return records


# ============================================================
# MAIN INGESTION
# ============================================================

def main():

    # Create data/ if necessary.
    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    failed = []

    # Write UTF-8 JSONL.
    with OUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:

        for drug in DRUGS:

            print(
                f"[fetch] {drug}"
            )

            try:

                label = fetch_best_label(drug)

            except requests.RequestException as error:

                print(
                    f"[error] {drug}: {error}"
                )

                failed.append(drug)

                continue

            if label is None:

                print(
                    f"[skip]  {drug}: no label found"
                )

                failed.append(drug)

                continue

            records = extract_records(
                drug,
                label,
            )

            for record in records:

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            total += len(records)

            print(
                f"[ok]    {drug}: "
                f"{len(records)} sections"
            )

            # Be polite to the API.
            time.sleep(0.5)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total section records: {total}")
    print(f"Output file: {OUT_PATH}")

    if failed:
        print(
            f"Failed/skipped drugs: {failed}"
        )
    else:
        print(
            "Failed/skipped drugs: None"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()