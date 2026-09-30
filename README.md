# 💊 PharmaRAG

### Evidence-Grounded RAG for FDA Drug-Label Question Answering

PharmaRAG is a Retrieval-Augmented Generation (RAG) system that answers natural-language questions about medications using retrieved FDA drug-label evidence.

Instead of relying entirely on an LLM's pretrained knowledge, PharmaRAG retrieves relevant drug-label passages, validates whether the evidence actually supports the question, and only then generates an answer.

When sufficient evidence cannot be established, the system abstains rather than fabricating an answer.

---

## Demo

![PharmaRAG Demo](assets/pharmarag-demo.png)

### Example

**Question**

> Can I take ibuprofen with warfarin?

**PharmaRAG**

> According to the FDA warfarin label, ibuprofen is listed among drugs that can increase the risk of bleeding when used concomitantly with warfarin. This information appears in the Drug Interactions section.

The application also exposes the retrieved FDA evidence and its source.

---

## Why PharmaRAG?

A conventional LLM can generate a fluent answer without actually having evidence for the claim.

For medication-related questions, that creates an important problem:

```text
Fluent answer ≠ Evidence-supported answer

PharmaRAG therefore separates:

Retrieval
    ↓
Evidence Validation
    ↓
Answer Generation

The language model is not treated as the source of truth.

The retrieved FDA label evidence is.

Key Features
FDA drug-label based question answering
Retrieval-Augmented Generation
Sentence Transformer embeddings
FAISS vector similarity search
Drug-name detection
Query intent detection
Cross-drug retrieval for interaction questions
Evidence-strength classification
Evidence sufficiency validation
Automatic abstention when evidence is insufficient
Local LLM inference using Ollama
Post-generation grounding verification
Deterministic fallback when generated output is not grounded
FDA source attribution
Streamlit web interface
System Architecture
                         USER QUESTION
                              │
                              ▼
                    ┌───────────────────┐
                    │   Drug Detection  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Intent Detection │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Semantic Retrieval│
                    │      + FAISS      │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Evidence          │
                    │ Validation        │
                    └─────────┬─────────┘
                              │
                     ┌────────┴────────┐
                     │                 │
              Insufficient         Sufficient
                     │                 │
                     ▼                 ▼
                  ABSTAIN       Strongest Evidence
                                       │
                                       ▼
                                ┌──────────────┐
                                │ Local LLM    │
                                │ Ollama       │
                                │ Llama 3.2 3B │
                                └──────┬───────┘
                                       │
                                       ▼
                              Output Verification
                                       │
                              ┌────────┴────────┐
                              │                 │
                           Grounded        Ungrounded
                              │                 │
                              ▼                 ▼
                         LLM Answer      Deterministic
                                         Evidence Answer
                              │
                              ▼
                     Answer + FDA Evidence
How It Works
1. Data Ingestion

Drug-label information is retrieved from the openFDA drug-label API.

Relevant sections include:

Boxed Warning
Indications and Usage
Dosage and Administration
Contraindications
Warnings and Precautions
Warnings
Adverse Reactions
Drug Interactions
Use in Specific Populations
Overdosage

The retrieved records are stored as JSONL data.

2. Text Chunking

Long FDA label sections are divided into smaller overlapping chunks.

Current configuration:

Chunk size: 180 words
Overlap:    40 words

This allows the retrieval system to work with focused passages rather than entire drug labels.

3. Embeddings

Each chunk is converted into a vector representation using:

sentence-transformers
all-MiniLM-L6-v2

The resulting embeddings have:

Dimension: 384
4. FAISS Retrieval

The embeddings are stored in a FAISS index.

Current index:

Vectors:   1,562
Dimension: 384
Index:     FAISS IndexFlatIP

Queries are embedded using the same model and compared against the indexed FDA passages.

5. Drug Detection

PharmaRAG detects supported drug names directly from the user's question.

For example:

Can I take ibuprofen with warfarin?

becomes:

Detected drugs:
- ibuprofen
- warfarin
6. Intent Detection

The system identifies the type of question.

Current intents include:

interaction
warning
contraindication
adverse_reaction
general

For example:

Can I take ibuprofen with warfarin?

is classified as:

interaction
7. Evidence Validation

Retrieval alone is not considered sufficient.

The retrieved passages are passed through a deterministic evidence-validation layer.

For an interaction question, the validator checks whether the relevant drugs occur together in an interaction-relevant FDA label section.

Evidence can be classified as:

direct
supporting

The validator also determines whether the available evidence is sufficient to answer the question.

8. Evidence-Gated Generation

Only when the validator determines that sufficient evidence exists does the answer-generation stage run.

The strongest evidence is prioritized before being passed to the local LLM.

The LLM is instructed to summarize the evidence without introducing unsupported claims.

9. Output Verification

The generated answer is checked after generation.

Unsupported responses such as:

There is no specific mention of ibuprofen...

when ibuprofen is explicitly present in the evidence are rejected.

If the generated answer fails validation, PharmaRAG uses a deterministic evidence-based fallback.

This creates an additional protection layer:

Retrieved Evidence
       ↓
LLM
       ↓
Generated Answer
       ↓
Grounding Check
       ↓
Valid Answer / Deterministic Fallback
Evidence-Gated Abstention

One of the main design goals is that PharmaRAG should not manufacture an interaction simply because two drugs appear in the same general dataset.

For example:

Question:
Can I take ibuprofen with acetaminophen?

If the retrieved FDA evidence does not establish the requested interaction, PharmaRAG returns:

I don't have sufficient evidence in the retrieved FDA labels
to answer this question.

This behavior is intentional.

Example Queries
Drug Interaction
Can I take ibuprofen with warfarin?

The system identifies:

Drugs: ibuprofen, warfarin
Intent: interaction
Confidence: high

and retrieves direct FDA evidence identifying ibuprofen among drugs that can increase bleeding risk when used concomitantly with warfarin.

Warnings
What are the warnings for warfarin?

The system retrieves relevant FDA sections including:

Warnings and Precautions
Boxed Warning
Adverse Reactions
What are the adverse reactions of ibuprofen?

The system retrieves evidence from the:

Adverse Reactions

section of the ibuprofen label.

Insufficient Evidence
Can I take ibuprofen with acetaminophen?

If the indexed FDA evidence does not directly establish the requested interaction, the system abstains.

Technology Stack
Component	Technology
Language	Python
RAG	Custom retrieval pipeline
Embeddings	Sentence Transformers
Embedding Model	all-MiniLM-L6-v2
Vector Database	FAISS
LLM Runtime	Ollama
Local Model	Llama 3.2 3B
UI	Streamlit
Data Source	openFDA drug-label API
API / Backend Support	Requests / FastAPI
Data Format	JSONL
Project Structure
pharma-rag/
│
├── data/
│   ├── labels.jsonl
│   ├── chunks.jsonl
│   └── index.faiss
│
├── src/
│   ├── build_index.py
│   ├── retrieve.py
│   ├── evidence_validator.py
│   └── answer_generator.py
│
├── app.py
├── main.py
├── ingest.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env

.env is intentionally excluded from version control.

Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/pharma-rag.git
cd pharma-rag
2. Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Install Ollama

Install Ollama and download the local model:

ollama pull llama3.2:3b

Verify:

ollama list
Running the Application

Start the Streamlit application:

streamlit run app.py

Then open:

http://localhost:8501
Command-Line Interface

PharmaRAG also includes a terminal interface:

python main.py
Rebuilding the Dataset

If the dataset needs to be regenerated:

python ingest.py

Then rebuild the vector index:

python src/build_index.py
Evaluation

The current prototype was tested against several question types.

Query	Drug Detection	Intent	Evidence	Behavior
Ibuprofen + warfarin	Correct	Interaction	High / Direct	Answer
Ibuprofen + acetaminophen	Correct	Interaction	Insufficient	Abstain
Warfarin warnings	Correct	Warning	Medium	Answer
Ibuprofen adverse reactions	Correct	Adverse Reaction	Medium	Answer

The most important evaluation criterion for the prototype is not simply whether the model produces an answer.

It is whether the system:

1. Retrieves relevant evidence
2. Validates the evidence
3. Answers when evidence is sufficient
4. Abstains when evidence is insufficient
Design Decisions
Why FAISS?

FAISS provides fast vector similarity search and is lightweight enough for a local RAG application.

Why a local LLM?

Ollama allows the project to perform generation locally without requiring a paid hosted LLM API.

Why evidence validation?

A retrieval system can retrieve text that is semantically similar but does not actually establish the user's claim.

The validation layer therefore acts as a gate between retrieval and generation.

Why abstention?

An unsupported answer is more problematic than explicitly stating that the indexed evidence is insufficient.

Why output verification?

Even when the retrieved evidence is correct, a generative model can misinterpret or contradict it. The output-verification layer provides another grounding check.

Limitations

This is a research/portfolio prototype and should not be treated as a medical decision-making system.

Current limitations include:

Limited drug-label coverage
Rule-based drug-name detection
Rule-based intent detection
Retrieval quality depends on the indexed labels
Evidence validation is based on deterministic heuristics
The local 3B model has limited reasoning capability
The system does not represent a patient's complete medical history
The system should not replace professional medical advice
Future Work

Potential improvements include:

Expand the FDA label collection
Improve brand-name and generic-name recognition
Add synonym normalization
Improve interaction-specific retrieval
Add automated retrieval evaluation
Measure Recall@K and Precision@K
Add answer faithfulness evaluation
Improve citation granularity
Add structured evidence extraction
Add a larger local or hosted model
Add automated regression tests
Deploy the application
