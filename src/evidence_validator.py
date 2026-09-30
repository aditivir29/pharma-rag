import re


# ============================================================
# EVIDENCE VALIDATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for easier keyword matching.
    """

    if not text:
        return ""

    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_drug(text, drug):
    """
    Check whether a drug name occurs as a complete word.
    """

    text = normalize_text(text)
    drug = normalize_text(drug)

    pattern = r"\b" + re.escape(drug) + r"\b"

    return re.search(pattern, text) is not None


# ============================================================
# INTERACTION EVIDENCE
# ============================================================

def validate_interaction_evidence(
    query,
    drugs,
    results
):
    """
    Validate evidence for a drug-interaction question.

    Strong evidence requires a retrieved chunk that:

    1. Comes from a relevant interaction section.
    2. Mentions the relevant drugs directly, OR
       explicitly discusses the relevant drug/class relationship.
    """

    if len(drugs) < 2:
        return {
            "sufficient": False,
            "reason": (
                "An interaction question should identify "
                "at least two drugs."
            ),
            "evidence": []
        }


    evidence = []


    for result in results:

        chunk = result.get(
            "chunk",
            {}
        )

        text = normalize_text(
            chunk.get(
                "text",
                ""
            )
        )

        section = normalize_text(
            chunk.get(
                "section",
                ""
            )
        )

        drug = normalize_text(
            chunk.get(
                "drug",
                ""
            )
        )


        # ----------------------------------------------------
        # CHECK WHETHER THIS IS AN INTERACTION-RELEVANT
        # SECTION
        # ----------------------------------------------------

        relevant_section = (
            "drug interactions" in section
            or "warnings and precautions" in section
            or section == "warnings"
            or "contraindications" in section
        )


        if not relevant_section:
            continue


        # ----------------------------------------------------
        # CHECK DRUG MENTIONS
        # ----------------------------------------------------

        mentioned_drugs = []

        for detected_drug in drugs:

            if contains_drug(
                text,
                detected_drug
            ):
                mentioned_drugs.append(
                    detected_drug
                )


        # ----------------------------------------------------
        # DIRECT TWO-DRUG EVIDENCE
        # ----------------------------------------------------

        if len(mentioned_drugs) >= 2:

            evidence.append(
                {
                    "result": result,
                    "strength": "direct",
                    "mentioned_drugs": mentioned_drugs
                }
            )

            continue


        # ----------------------------------------------------
        # LABEL-LEVEL EVIDENCE
        #
        # A warfarin interaction section may mention warfarin
        # throughout the text while listing ibuprofen in a
        # table/classification.
        # ----------------------------------------------------

        if (
            len(mentioned_drugs) >= 1
            and drug in [
                normalize_text(d)
                for d in drugs
            ]
        ):

            # Check for interaction language
            interaction_terms = [
                "interact",
                "interaction",
                "concomitant",
                "bleeding risk",
                "increase the risk of bleeding",
                "taken together",
                "used concomitantly",
                "coadministration",
                "concurrent",
                "concurrently",
            ]

            has_interaction_language = any(
                term in text
                for term in interaction_terms
            )


            if has_interaction_language:

                evidence.append(
                    {
                        "result": result,
                        "strength": "supporting",
                        "mentioned_drugs": mentioned_drugs
                    }
                )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    direct_evidence = [
        item
        for item in evidence
        if item["strength"] == "direct"
    ]

    supporting_evidence = [
        item
        for item in evidence
        if item["strength"] == "supporting"
    ]


    if direct_evidence:

        return {
            "sufficient": True,
            "confidence": "high",
            "reason": (
                "Retrieved FDA label evidence directly "
                "mentions the relevant drugs together "
                "in an interaction-relevant section."
            ),
            "evidence": evidence
        }


    if len(supporting_evidence) >= 2:

        return {
            "sufficient": True,
            "confidence": "medium",
            "reason": (
                "Multiple interaction-relevant FDA "
                "label passages support the question."
            ),
            "evidence": evidence
        }


    return {
        "sufficient": False,
        "confidence": "low",
        "reason": (
            "The retrieved FDA label evidence does not "
            "directly establish the requested drug interaction."
        ),
        "evidence": evidence
    }


# ============================================================
# GENERAL EVIDENCE VALIDATION
# ============================================================

def validate_evidence(
    query,
    drugs,
    intent,
    results
):
    """
    Main evidence-validation function.
    """

    if not results:

        return {
            "sufficient": False,
            "confidence": "none",
            "reason": "No evidence was retrieved.",
            "evidence": []
        }


    # --------------------------------------------------------
    # INTERACTION QUESTIONS
    # --------------------------------------------------------

    if intent == "interaction":

        return validate_interaction_evidence(
            query,
            drugs,
            results
        )


    # --------------------------------------------------------
    # OTHER QUESTION TYPES
    #
    # For now, require at least one relevant section.
    # We will make these validators more sophisticated later.
    # --------------------------------------------------------

    relevant_results = []

    for result in results:

        chunk = result.get(
            "chunk",
            {}
        )

        section = normalize_text(
            chunk.get(
                "section",
                ""
            )
        )


        if intent == "warning":

            if (
                "boxed warning" in section
                or "warnings and precautions" in section
                or section == "warnings"
            ):

                relevant_results.append(result)


        elif intent == "contraindication":

            if "contraindications" in section:

                relevant_results.append(result)


        elif intent == "adverse_reaction":

            if "adverse reactions" in section:

                relevant_results.append(result)


        else:

            relevant_results.append(result)


    if relevant_results:

        return {
            "sufficient": True,
            "confidence": "medium",
            "reason": (
                "Retrieved evidence comes from "
                "a section relevant to the question."
            ),
            "evidence": [
                {
                    "result": result,
                    "strength": "supporting",
                    "mentioned_drugs": []
                }
                for result in relevant_results
            ]
        }


    return {
        "sufficient": False,
        "confidence": "low",
        "reason": (
            "The retrieved evidence does not contain "
            "a section relevant to the question."
        ),
        "evidence": []
    }