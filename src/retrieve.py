from pathlib import Path
import json
import re

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"
INDEX_PATH = BASE_DIR / "data" / "index.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K_PER_DRUG = 5
FINAL_TOP_K = 5


# ============================================================
# KNOWN DRUG NAMES
# ============================================================

DRUG_NAMES = [
    "acetaminophen",
    "albuterol",
    "amoxicillin",
    "amlodipine",
    "atorvastatin",
    "azithromycin",
    "cetirizine",
    "ciprofloxacin",
    "clopidogrel",
    "dexamethasone",
    "diazepam",
    "diclofenac",
    "diphenhydramine",
    "doxycycline",
    "escitalopram",
    "fluoxetine",
    "furosemide",
    "gabapentin",
    "glimepiride",
    "hydrochlorothiazide",
    "ibuprofen",
    "insulin glargine",
    "lisinopril",
    "loratadine",
    "losartan",
    "meloxicam",
    "metformin",
    "metoprolol",
    "montelukast",
    "naproxen",
    "omeprazole",
    "paracetamol",
    "prednisone",
    "rosuvastatin",
    "sertraline",
    "simvastatin",
    "tramadol",
    "warfarin",
    "pantoprazole",
    "aspirin",
]


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(str(INDEX_PATH))


# ============================================================
# LOAD CHUNKS
# ============================================================

chunks = []

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))


print(f"Loaded {len(chunks)} chunks.")
print(f"FAISS contains {index.ntotal} vectors.")


# ============================================================
# DRUG DETECTION
# ============================================================

def detect_drugs(query):
    """
    Detect known drug names mentioned in the query.
    """

    query_lower = query.lower()

    found = []

    # Check longer names first
    sorted_drugs = sorted(
        DRUG_NAMES,
        key=len,
        reverse=True
    )

    for drug in sorted_drugs:

        pattern = r"\b" + re.escape(drug.lower()) + r"\b"

        if re.search(pattern, query_lower):
            found.append(drug)

    return found


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(query):
    """
    Detect the user's question intent.

    Possible intents:
        interaction
        contraindication
        warning
        adverse_reaction
        general
    """

    q = query.lower().strip()

    # --------------------------------------------------------
    # DRUG INTERACTION
    # --------------------------------------------------------

    interaction_phrases = [
        "interaction",
        "interactions",
        "interact",
        "interacts",
        "taken with",
        "use with",
        "used with",
        "using with",
        "together",
        "combine",
        "combined",
        "combination",
        "concomitant",
        "mix with",
        "mix",
        "take together",
        "taken together",
        "use together",
        "used together",
        "at the same time",
        "along with",
        "alongside",
    ]

    if any(
        phrase in q
        for phrase in interaction_phrases
    ):
        return "interaction"


    # --------------------------------------------------------
    # NATURAL-LANGUAGE INTERACTION PATTERNS
    # --------------------------------------------------------

    interaction_patterns = [
        r"\bcan i take .+ with .+",
        r"\bcan i use .+ with .+",
        r"\bcan i take .+ together with .+",
        r"\bcan i use .+ together with .+",
        r"\bcan .+ be taken with .+",
        r"\bcan .+ be used with .+",
        r"\bshould i take .+ with .+",
        r"\bshould i use .+ with .+",
        r"\bis it safe to take .+ with .+",
        r"\bis it safe to use .+ with .+",
        r"\bcan .+ be taken together",
        r"\bcan .+ be used together",
        r"\bcan .+ be taken at the same time",
        r"\bcan .+ be used at the same time",
    ]

    for pattern in interaction_patterns:

        if re.search(pattern, q):
            return "interaction"


    # --------------------------------------------------------
    # CONTRAINDICATION
    # --------------------------------------------------------

    contraindication_phrases = [
        "contraindication",
        "contraindications",
        "contraindicated",
        "cannot take",
        "can't take",
        "can not take",
        "should not take",
        "shouldn't take",
        "not take",
        "do not take",
        "don't take",
    ]

    if any(
        phrase in q
        for phrase in contraindication_phrases
    ):
        return "contraindication"


    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    warning_phrases = [
        "warning",
        "warnings",
        "precaution",
        "precautions",
        "risk",
        "risks",
        "danger",
    ]

    if any(
        phrase in q
        for phrase in warning_phrases
    ):
        return "warning"


    # --------------------------------------------------------
    # ADVERSE REACTIONS / SIDE EFFECTS
    # --------------------------------------------------------

    adverse_phrases = [
        "side effect",
        "side effects",
        "adverse reaction",
        "adverse reactions",
        "adverse effect",
        "adverse effects",
    ]

    if any(
        phrase in q
        for phrase in adverse_phrases
    ):
        return "adverse_reaction"


    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    return "general"


# ============================================================
# SECTION RELEVANCE
# ============================================================

def section_bonus(section, intent):
    """
    Add a score bonus based on the relevance of the FDA
    label section to the detected intent.
    """

    section_lower = section.lower()


    # --------------------------------------------------------
    # INTERACTION
    # --------------------------------------------------------

    if intent == "interaction":

        if "drug interactions" in section_lower:
            return 0.25

        if "warnings and precautions" in section_lower:
            return 0.12

        if "contraindications" in section_lower:
            return 0.05

        if section_lower == "warnings":
            return 0.03

        return 0.0


    # --------------------------------------------------------
    # CONTRAINDICATION
    # --------------------------------------------------------

    if intent == "contraindication":

        if "contraindications" in section_lower:
            return 0.25

        if "warnings and precautions" in section_lower:
            return 0.12

        if section_lower == "warnings":
            return 0.05

        return 0.0


    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    if intent == "warning":

        if "boxed warning" in section_lower:
            return 0.15

        if "warnings and precautions" in section_lower:
            return 0.12

        if section_lower == "warnings":
            return 0.10

        return 0.0


    # --------------------------------------------------------
    # ADVERSE REACTION
    # --------------------------------------------------------

    if intent == "adverse_reaction":

        if "adverse reactions" in section_lower:
            return 0.20

        return 0.0


    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    return 0.0


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(query, top_k=TOP_K_PER_DRUG):
    """
    Search the FAISS vector index using semantic similarity.
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        chunk = chunks[idx]

        results.append(
            {
                "faiss_score": float(score),
                "chunk": chunk
            }
        )

    return results


# ============================================================
# CROSS-DRUG KEYWORD SEARCH
# ============================================================

def find_cross_drug_chunks(drugs):
    """
    Find chunks where multiple detected drugs occur together.

    This is useful for questions such as:

        Can I take ibuprofen with warfarin?

    A semantic search may retrieve separate chunks for
    ibuprofen and warfarin. This search looks for direct
    textual evidence where both drug names appear in the
    same chunk.
    """

    if len(drugs) < 2:
        return []

    results = []

    for chunk in chunks:

        text = chunk.get(
            "text",
            ""
        ).lower()

        matched = []

        for drug in drugs:

            pattern = (
                r"\b"
                + re.escape(drug.lower())
                + r"\b"
            )

            if re.search(pattern, text):
                matched.append(drug)

        # Keep chunks containing at least two detected drugs
        if len(matched) >= 2:

            results.append(
                {
                    "faiss_score": 0.0,
                    "chunk": chunk,
                    "matched_drugs": matched,
                    "cross_drug_match": True
                }
            )

    return results


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    final_top_k=FINAL_TOP_K
):
    """
    Main retrieval pipeline.

    Steps:
        1. Detect drugs.
        2. Detect intent.
        3. Perform global semantic search.
        4. Search each detected drug.
        5. Search combined drug query.
        6. Search for direct cross-drug evidence.
        7. Rerank using drug + section + evidence bonuses.
        8. Ensure multi-drug coverage.
    """

    drugs = detect_drugs(query)

    intent = detect_intent(query)


    # ========================================================
    # DISPLAY DETECTION
    # ========================================================

    print()
    print("=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    print(
        "Detected drugs:",
        drugs if drugs else "None"
    )

    print(
        "Detected intent:",
        intent
    )


    # ========================================================
    # NO DRUG DETECTED
    # ========================================================

    if not drugs:

        results = semantic_search(
            query,
            top_k=final_top_k
        )

        for result in results:

            section = result["chunk"].get(
                "section",
                ""
            )

            result["final_score"] = (
                result["faiss_score"]
                + section_bonus(
                    section,
                    intent
                )
            )

            result["matched_drugs"] = []

            result["cross_drug_matches"] = []


        results.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return results[:final_top_k]


    # ========================================================
    # DRUGS DETECTED
    # ========================================================

    candidate_results = {}


    # ========================================================
    # 1. GLOBAL SEMANTIC SEARCH
    # ========================================================

    global_results = semantic_search(
        query,
        top_k=30
    )

    for result in global_results:

        chunk = result["chunk"]

        candidate_results[
            chunk["chunk_id"]
        ] = result


    # ========================================================
    # 2. SEARCH EACH DRUG SEPARATELY
    # ========================================================

    for drug in drugs:

        drug_results = semantic_search(
            drug,
            top_k=TOP_K_PER_DRUG
        )

        for result in drug_results:

            chunk = result["chunk"]

            chunk_id = chunk["chunk_id"]

            # Keep the stronger semantic result
            if (
                chunk_id not in candidate_results
                or result["faiss_score"]
                > candidate_results[
                    chunk_id
                ]["faiss_score"]
            ):

                candidate_results[
                    chunk_id
                ] = result


    # ========================================================
    # 3. COMBINED DRUG SEARCH
    # ========================================================

    if len(drugs) >= 2:

        combined_query = (
            " ".join(drugs)
            + " drug interaction"
        )

        combined_results = semantic_search(
            combined_query,
            top_k=15
        )

        for result in combined_results:

            chunk = result["chunk"]

            chunk_id = chunk["chunk_id"]

            if (
                chunk_id not in candidate_results
                or result["faiss_score"]
                > candidate_results[
                    chunk_id
                ]["faiss_score"]
            ):

                candidate_results[
                    chunk_id
                ] = result


    # ========================================================
    # 4. DIRECT CROSS-DRUG SEARCH
    # ========================================================

    cross_drug_results = find_cross_drug_chunks(
        drugs
    )

    for result in cross_drug_results:

        chunk = result["chunk"]

        chunk_id = chunk["chunk_id"]

        if chunk_id not in candidate_results:

            candidate_results[
                chunk_id
            ] = result

        else:

            candidate_results[
                chunk_id
            ]["cross_drug_match"] = True


    # ========================================================
    # 5. RERANK
    # ========================================================

    reranked = []


    for result in candidate_results.values():

        chunk = result["chunk"]

        drug_name = chunk.get(
            "drug",
            ""
        ).lower()

        text = chunk.get(
            "text",
            ""
        ).lower()

        semantic_score = result.get(
            "faiss_score",
            0.0
        )

        score = semantic_score

        matching_drugs = []


        # ----------------------------------------------------
        # DRUG MATCH BONUS
        # ----------------------------------------------------

        for drug in drugs:

            if drug.lower() == drug_name:

                matching_drugs.append(drug)

        if matching_drugs:

            score += 0.20


        # ----------------------------------------------------
        # SECTION BONUS
        # ----------------------------------------------------

        score += section_bonus(
            chunk.get(
                "section",
                ""
            ),
            intent
        )


        # ----------------------------------------------------
        # MULTI-DRUG COVERAGE BONUS
        # ----------------------------------------------------

        if (
            len(drugs) > 1
            and matching_drugs
        ):

            score += 0.05


        # ----------------------------------------------------
        # CHECK DRUG NAMES INSIDE CHUNK TEXT
        # ----------------------------------------------------

        cross_drug_matches = []

        for drug in drugs:

            pattern = (
                r"\b"
                + re.escape(drug.lower())
                + r"\b"
            )

            if re.search(
                pattern,
                text
            ):

                cross_drug_matches.append(drug)


        # ----------------------------------------------------
        # DIRECT CROSS-DRUG TEXT BONUS
        # ----------------------------------------------------
        #
        # Strong bonus when BOTH requested drugs are explicitly
        # present in the SAME FDA chunk.
        #
        # This makes direct evidence rank above generic
        # semantic matches.
        # ----------------------------------------------------

        if len(cross_drug_matches) >= 2:

            score += 0.80


        # ----------------------------------------------------
        # CROSS-DRUG SEARCH BONUS
        # ----------------------------------------------------

        if result.get(
            "cross_drug_match",
            False
        ):

            score += 0.25


        # ----------------------------------------------------
        # SAVE SCORES
        # ----------------------------------------------------

        result["final_score"] = score

        result["matched_drugs"] = matching_drugs

        result["cross_drug_matches"] = (
            cross_drug_matches
        )

        reranked.append(result)


    # ========================================================
    # SORT BY FINAL SCORE
    # ========================================================

    reranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )


    # ========================================================
    # GUARANTEE EACH DETECTED DRUG IS REPRESENTED
    # ========================================================

    selected = []

    selected_ids = set()


    for drug in drugs:

        drug_candidates = [
            result
            for result in reranked
            if (
                drug.lower()
                in result.get(
                    "matched_drugs",
                    []
                )
                or drug.lower()
                in [
                    d.lower()
                    for d in result.get(
                        "cross_drug_matches",
                        []
                    )
                ]
            )
        ]

        if drug_candidates:

            best = drug_candidates[0]

            chunk_id = best["chunk"]["chunk_id"]

            if chunk_id not in selected_ids:

                selected.append(best)

                selected_ids.add(chunk_id)


    # ========================================================
    # FILL REMAINING SLOTS
    # ========================================================

    for result in reranked:

        chunk_id = result["chunk"]["chunk_id"]

        if chunk_id not in selected_ids:

            selected.append(result)

            selected_ids.add(chunk_id)


        if len(selected) >= final_top_k:

            break


    return selected[:final_top_k]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):
    """
    Display retrieved FDA label chunks.
    """

    print()
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)


    if not results:

        print("No results found.")

        return


    for i, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]


        print()
        print(
            f"RESULT #{i}"
        )

        print(
            "-" * 70
        )


        print(
            f"Final Score: "
            f"{result['final_score']:.4f}"
        )


        print(
            f"FAISS Score: "
            f"{result['faiss_score']:.4f}"
        )


        print(
            f"Drug: "
            f"{chunk.get('drug', 'N/A')}"
        )


        print(
            f"Brand: "
            f"{chunk.get('brand', 'N/A')}"
        )


        print(
            f"Section: "
            f"{chunk.get('section', 'N/A')}"
        )


        print(
            f"Chunk ID: "
            f"{chunk.get('chunk_id', 'N/A')}"
        )


        print(
            f"Matched Drugs: "
            f"{result.get('matched_drugs', [])}"
        )


        print(
            f"Cross-Drug Matches: "
            f"{result.get('cross_drug_matches', [])}"
        )


        print(
            f"FDA Source: "
            f"{chunk.get('source_url', 'N/A')}"
        )


        print()

        print(
            chunk.get(
                "text",
                ""
            )
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    while True:

        print()

        query = input(
            "Enter your question "
            "(or type 'exit'): "
        ).strip()


        if query.lower() == "exit":

            print(
                "Exiting..."
            )

            break


        if not query:

            print(
                "Please enter a question."
            )

            continue


        results = retrieve(
            query,
            final_top_k=FINAL_TOP_K
        )


        display_results(
            query,
            results
        )