from src.retrieve import retrieve
from src.evidence_validator import validate_evidence
from src.answer_generator import generate_answer


FINAL_TOP_K = 5


def run_pharma_rag(query):
    print()
    print("=" * 70)
    print("PHARMARAG")
    print("=" * 70)

    # ------------------------------------------------------------
    # STEP 1: DETECT DRUGS AND INTENT
    # ------------------------------------------------------------

    from src.retrieve import detect_drugs, detect_intent

    drugs = detect_drugs(query)
    intent = detect_intent(query)

    print()
    print("Detected drugs:", drugs if drugs else "None")
    print("Detected intent:", intent)

    # ------------------------------------------------------------
    # STEP 2: RETRIEVE FDA EVIDENCE
    # ------------------------------------------------------------

    print()
    print("Retrieving FDA evidence...")

    results = retrieve(
        query,
        final_top_k=FINAL_TOP_K
    )

    print()
    print(f"Retrieved {len(results)} evidence chunks.")

    # ------------------------------------------------------------
    # STEP 3: VALIDATE EVIDENCE
    # ------------------------------------------------------------

    print()
    print("Validating evidence...")

    validation_result = validate_evidence(
        query=query,
        drugs=drugs,
        intent=intent,
        results=results
    )

    print(
        "Evidence sufficient:",
        validation_result.get("sufficient", False)
    )

    print(
        "Confidence:",
        validation_result.get("confidence", "unknown")
    )

    print(
        "Reason:",
        validation_result.get("reason", "")
    )

    # ------------------------------------------------------------
    # STEP 4: GENERATE ANSWER
    # ------------------------------------------------------------

    print()
    print("Generating answer...")

    answer = generate_answer(
    query=query,
    validation_result=validation_result,
    drugs=drugs,
    intent=intent
)

    # ------------------------------------------------------------
    # STEP 5: DISPLAY FINAL ANSWER
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("PHARMARAG ANSWER")
    print("=" * 70)

    print(answer)

    # ------------------------------------------------------------
    # STEP 6: DISPLAY EVIDENCE USED
    # ------------------------------------------------------------

    if validation_result.get("sufficient", False):

        print()
        print("=" * 70)
        print("FDA EVIDENCE USED")
        print("=" * 70)

        for i, item in enumerate(
            validation_result.get("evidence", []),
            start=1
        ):
            result = item["result"]
            chunk = result["chunk"]

            print()
            print(f"EVIDENCE #{i}")
            print("-" * 70)

            print(
                "Drug:",
                chunk.get("drug", "N/A")
            )

            print(
                "Brand:",
                chunk.get("brand", "N/A")
            )

            print(
                "FDA Section:",
                chunk.get("section", "N/A")
            )

            print(
                "Evidence Strength:",
                item.get("strength", "N/A")
            )

            print(
                "Chunk ID:",
                chunk.get("chunk_id", "N/A")
            )

            print()
            print(chunk.get("text", ""))


def main():

    print("=" * 70)
    print("PHARMARAG - FDA DRUG LABEL Q&A")
    print("=" * 70)

    print()
    print("Ask a question about the available FDA drug labels.")
    print("Type 'exit' to quit.")

    while True:

        print()

        query = input("Question: ").strip()

        if query.lower() == "exit":
            print("Exiting PharmaRAG...")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            run_pharma_rag(query)

        except Exception as e:

            print()
            print("=" * 70)
            print("ERROR")
            print("=" * 70)
            print(str(e))


if __name__ == "__main__":
    main()