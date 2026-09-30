import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def get_best_evidence(validation_result):
    """
    Return the strongest validated evidence.
    Direct evidence is preferred over supporting evidence.
    """

    evidence_items = validation_result.get("evidence", [])

    if not evidence_items:
        return None

    direct = [
        item
        for item in evidence_items
        if item.get("strength") == "direct"
    ]

    if direct:
        return direct[0]

    return evidence_items[0]


def extract_interaction_evidence(evidence_item):
    """
    Extract the most relevant portion of an interaction evidence chunk.

    For example, from a large warfarin label chunk we want:

    "Because bleeding risk is increased when these drugs are used
    concomitantly with warfarin..."

    and the table entry containing ibuprofen.
    """

    chunk = evidence_item["result"]["chunk"]

    text = chunk.get("text", "")
    drug = chunk.get("drug", "")
    section = chunk.get("section", "")

    lower_text = text.lower()

    # Find the important bleeding-risk sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    relevant_sentences = []

    for sentence in sentences:
        lower_sentence = sentence.lower()

        if (
            "bleeding risk" in lower_sentence
            or "risk of bleeding" in lower_sentence
            or "concomitantly" in lower_sentence
            or "interact" in lower_sentence
        ):
            relevant_sentences.append(sentence.strip())

    # Find the sentence/table containing the queried drug.
    if drug:
        for sentence in sentences:
            if drug.lower() in sentence.lower():
                if sentence.strip() not in relevant_sentences:
                    relevant_sentences.append(sentence.strip())

    # If we found useful sentences, keep them.
    if relevant_sentences:
        compact_text = " ".join(relevant_sentences)

        # Limit extremely large chunks.
        if len(compact_text) > 1500:
            compact_text = compact_text[:1500]

        return {
            "drug": drug,
            "section": section,
            "text": compact_text,
            "full_text": text
        }

    return {
        "drug": drug,
        "section": section,
        "text": text[:1500],
        "full_text": text
    }


def format_compact_evidence(evidence_data):
    return f"""
FDA LABEL SECTION:
{evidence_data["section"]}

FDA LABEL EVIDENCE:
{evidence_data["text"]}
"""


def deterministic_interaction_answer(
    query,
    evidence_data,
    detected_drugs=None
):
    """
    Safe fallback answer generated directly from validated evidence.

    This does NOT use the LLM.
    """

    text = evidence_data["full_text"]
    section = evidence_data["section"]

    lower_text = text.lower()

    drugs = detected_drugs or []

    # Try to identify the two drugs in the question.
    if len(drugs) >= 2:
        drug1 = drugs[0]
        drug2 = drugs[1]

        # Determine which drug owns the FDA label.
        label_drug = evidence_data["drug"]

        if label_drug.lower() == drug1.lower():
            other_drug = drug2
        elif label_drug.lower() == drug2.lower():
            other_drug = drug1
        else:
            other_drug = drug1

        # Specific FDA wording found in the current evidence.
        if (
            "risk of bleeding" in lower_text
            and other_drug.lower() in lower_text
        ):
            return (
                f"According to the FDA {label_drug} label, "
                f"{other_drug} is listed among drugs that can increase "
                f"the risk of bleeding when used concomitantly with "
                f"{label_drug}. This information appears in the "
                f"{section} section."
            )

    # Generic fallback.
    return (
        f"According to the FDA label, the retrieved evidence in the "
        f"{section} section describes an interaction or risk relevant "
        f"to the question."
    )


def output_is_grounded(answer, evidence_text, drugs):
    """
    Basic post-generation verification.

    Reject answers that contradict the evidence or introduce
    unsupported medical advice.
    """

    answer_lower = answer.lower()
    evidence_lower = evidence_text.lower()

    # Dangerous/unwanted generic responses.
    forbidden_phrases = [
        "i cannot provide medical advice",
        "i can't provide medical advice",
        "consult a healthcare professional",
        "consult your doctor",
        "talk to your doctor",
        "you should consult",
        "i recommend consulting",
        "there is no specific mention",
        "not mentioned in the evidence",
        "no evidence",
    ]

    for phrase in forbidden_phrases:
        if phrase in answer_lower:
            return False

    # If a drug is explicitly present in evidence,
    # the answer must not claim it is absent.
    for drug in drugs:
        if drug.lower() in evidence_lower:
            absence_patterns = [
                f"no mention of {drug.lower()}",
                f"{drug.lower()} is not mentioned",
                f"{drug.lower()} was not mentioned",
                f"{drug.lower()} is absent",
            ]

            for pattern in absence_patterns:
                if pattern in answer_lower:
                    return False

    # The answer should contain at least one relevant drug.
    if drugs:
        contains_drug = any(
            drug.lower() in answer_lower
            for drug in drugs
        )

        if not contains_drug:
            return False

    return True


def generate_with_ollama(query, evidence_data):
    """
    Ask the local LLM to summarize only the compact evidence.
    """

    evidence = format_compact_evidence(evidence_data)

    prompt = f"""
You are a factual FDA drug-label summarizer.

USER QUESTION:
{query}

{evidence}

Write a 1-2 sentence factual summary of ONLY what the FDA label
evidence says.

IMPORTANT:

- The FDA evidence above is authoritative for this answer.
- Read the actual evidence carefully.
- Do not claim that a drug is absent if it appears in the evidence.
- Do not add medical knowledge.
- Do not give medical advice.
- Do not recommend consulting a doctor.
- Do not say "I cannot provide medical advice".
- Do not say "there is no specific mention".
- Do not change "increased risk" into "contraindicated".
- Do not change "increased risk" into "must not take".
- Preserve the meaning of the FDA label.
- Mention the relevant FDA label section.

Return ONLY the factual answer.

Answer:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 150
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()


def generate_answer(
    query,
    validation_result,
    drugs=None,
    intent=None
):
    """
    Main answer-generation pipeline.

    Architecture:

        Validator
            ↓
        Best Evidence
            ↓
        Compact Evidence
            ↓
        Ollama
            ↓
        Output Verification
            ↓
        Valid → return LLM answer
        Invalid → deterministic fallback
    """

    # ---------------------------------------------------------
    # 1. Evidence gate
    # ---------------------------------------------------------

    if not validation_result.get("sufficient", False):
        return (
            "I don't have sufficient evidence in the retrieved "
            "FDA labels to answer this question."
        )

    # ---------------------------------------------------------
    # 2. Select strongest evidence
    # ---------------------------------------------------------

    best_evidence = get_best_evidence(validation_result)

    if best_evidence is None:
        return (
            "I don't have sufficient evidence in the retrieved "
            "FDA labels to answer this question."
        )

    # ---------------------------------------------------------
    # 3. Extract compact evidence
    # ---------------------------------------------------------

    evidence_data = extract_interaction_evidence(
        best_evidence
    )

    # ---------------------------------------------------------
    # 4. Generate answer with Ollama
    # ---------------------------------------------------------

    try:
        answer = generate_with_ollama(
            query,
            evidence_data
        )

        # -----------------------------------------------------
        # 5. Verify generated answer
        # -----------------------------------------------------

        grounded = output_is_grounded(
            answer,
            evidence_data["full_text"],
            drugs or []
        )

        if grounded:
            return answer

    except Exception:
        pass

    # ---------------------------------------------------------
    # 6. Deterministic fallback
    # ---------------------------------------------------------

    return deterministic_interaction_answer(
        query=query,
        evidence_data=evidence_data,
        detected_drugs=drugs
    )