# PharmaRAG

PharmaRAG is an evidence-grounded Retrieval-Augmented Generation (RAG) system for answering questions about prescription and medication information using official FDA drug-label data.

The system retrieves relevant drug-label passages, validates whether the retrieved evidence actually supports the user's question, and generates an answer grounded in the retrieved evidence.

If sufficient evidence is not available, PharmaRAG abstains instead of generating an unsupported answer.

## Features

- FDA drug-label based question answering
- Semantic search using Sentence Transformers
- FAISS vector database
- Drug and intent detection
- Cross-drug interaction retrieval
- Evidence validation before generation
- Evidence-grounded local LLM generation
- Post-generation grounding verification
- Deterministic fallback when the LLM output is not grounded
- Source attribution to the underlying drug label
- Streamlit web interface
- Abstention when evidence is insufficient

## Architecture

```text
                    User Question
                          |
                          v
                  Drug Detection
                          |
                          v
                  Intent Detection
                          |
                          v
                  Semantic Retrieval
                          |
                          v
                     FAISS Index
                          |
                          v
                 Evidence Validation
                          |
                +---------+---------+
                |                   |
          Insufficient          Sufficient
                |                   |
                v                   v
             Abstain          Best Evidence
                                    |
                                    v
                              Local LLM
                                    |
                                    v
                         Output Verification
                                    |
                         +----------+----------+
                         |                     |
                      Valid                 Invalid
                         |                     |
                         v                     v
                    LLM Answer        Deterministic Fallback
                         |
                         v
                  FDA Evidence + Source
Technology Stack
Python
Sentence Transformers
FAISS
Ollama
Llama 3.2 3B
FastAPI
Streamlit
Pandas
Requests
Data Source

The system uses official FDA drug-label information retrieved through the openFDA drug-label API.

Retrieved label records are processed into smaller text chunks before embedding and indexing.

Retrieval Pipeline
User submits a natural-language question.
Supported drug names are detected.
The question intent is identified.
Relevant FDA label chunks are retrieved using semantic similarity.
Additional cross-drug evidence is searched for interaction questions.
Retrieved evidence is validated using deterministic Python rules.
If sufficient evidence exists, the strongest evidence is passed to the local LLM.
The generated answer is checked against the evidence.
Unsupported generations are rejected and replaced with a deterministic evidence-based response.
The UI displays the answer and supporting FDA evidence.
Example
Question
Can I take ibuprofen with warfarin?
System output
According to the FDA warfarin label, ibuprofen is listed among
drugs that can increase the risk of bleeding when used
concomitantly with warfarin.

The interface also displays the relevant FDA label section and retrieved evidence.

Abstention Example
Question
Can I take ibuprofen with acetaminophen?

If the indexed FDA evidence does not establish the requested interaction, PharmaRAG does not fabricate an answer.

I don't have sufficient evidence in the retrieved FDA labels
to answer this question.
Installation

Clone the repository:

git clone <YOUR_REPOSITORY_URL>
cd pharma-rag

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Install and run Ollama, then download the local model:

ollama pull llama3.2:3b
Running PharmaRAG

Start the Streamlit application:

streamlit run app.py

The application will be available at:

http://localhost:8501
Command-Line Version

The project also includes a terminal interface:

python main.py
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
└── .gitignore
Key Design Principle

PharmaRAG does not treat the language model as the source of truth.

The retrieval and evidence-validation layers determine whether sufficient evidence exists. The LLM is used primarily to summarize validated evidence.

This reduces the risk of unsupported answers and allows the system to abstain when the retrieved evidence does not establish the requested claim.

Future Improvements
Expand the drug-label dataset
Improve drug-name and brand-name recognition
Add automated retrieval evaluation
Add answer faithfulness evaluation
Add retrieval precision/recall metrics
Improve citation granularity
Add conversation history
Deploy the Streamlit application