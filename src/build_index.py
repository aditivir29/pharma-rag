"""
PharmaRAG - Step 2: Chunking, Embeddings and FAISS Index

Reads FDA label sections from data/labels.jsonl,
splits long sections into overlapping chunks,
creates embeddings using Sentence Transformers,
and stores the vectors in a FAISS index.

Output:
    data/chunks.jsonl
    data/index.faiss
"""


import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/labels.jsonl")

CHUNKS_PATH = Path("data/chunks.jsonl")

INDEX_PATH = Path("data/index.faiss")


# Embedding model.
#
# This is a lightweight general-purpose sentence embedding
# model and is suitable for our first version.
MODEL_NAME = "all-MiniLM-L6-v2"


# Maximum approximate number of words per chunk.
CHUNK_SIZE = 180

# Number of words repeated between neighboring chunks.
CHUNK_OVERLAP = 40


# ============================================================
# LOAD LABEL RECORDS
# ============================================================

def load_records():
    """
    Load section-level records from JSONL.
    """

    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Split text into overlapping word-based chunks.

    Example:

        Chunk 1:
        words 0-179

        Chunk 2:
        words 140-319

        Chunk 3:
        words 280-459

    The overlap helps preserve context across boundaries.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# CREATE CHUNKS
# ============================================================

def build_chunks(records):
    """
    Convert section-level records into chunk-level records.
    """

    chunks = []

    chunk_id = 0

    for record in records:

        text_chunks = chunk_text(
            record["text"]
        )

        for chunk_number, text in enumerate(
            text_chunks
        ):

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "drug": record["drug"],
                    "brand": record["brand"],
                    "section": record["section"],
                    "chunk_number": chunk_number,
                    "text": text,
                    "set_id": record["set_id"],
                    "source_url": record["source_url"],
                }
            )

            chunk_id += 1

    return chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(chunks):
    """
    Save chunk metadata as JSONL.
    """

    with CHUNKS_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        for chunk in chunks:

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(chunks):
    """
    Generate embeddings and build a FAISS cosine-similarity
    index.

    FAISS IndexFlatIP is used with normalized embeddings,
    making inner product equivalent to cosine similarity.
    """

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Ensure float32 for FAISS.
    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    # Inner product on normalized vectors
    # is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    print(
        f"FAISS index saved to: {INDEX_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PharmaRAG - Step 2")
    print("Chunking + Embeddings + FAISS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("Loading FDA records...")

    records = load_records()

    print(
        f"Loaded {len(records)} section records."
    )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    print()
    print("Creating chunks...")

    chunks = build_chunks(
        records
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # Save chunk metadata
    # --------------------------------------------------------

    save_chunks(
        chunks
    )

    print(
        f"Chunk metadata saved to: "
        f"{CHUNKS_PATH}"
    )

    # --------------------------------------------------------
    # Embeddings + FAISS
    # --------------------------------------------------------

    build_faiss_index(
        chunks
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)

    print(
        f"Section records : {len(records)}"
    )

    print(
        f"Chunks          : {len(chunks)}"
    )

    print(
        f"FAISS index     : {INDEX_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()