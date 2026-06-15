"""
rag.py — The core retrieval-augmented generation pipeline.

Read this top to bottom. It is intentionally written without a heavyweight
RAG framework (no LangChain here) so you can see every step:

    chunk  ->  embed  ->  store  ->  retrieve  ->  generate

Each function is one stage. The vector store is a simple in-memory list of
(chunk_text, embedding) pairs with cosine similarity search. That is all a
"vector database" fundamentally is — the production versions (pgvector,
Chroma, Pinecone) just add persistence, scale, and indexing on top.

EMBEDDINGS NOTE:
Anthropic does not currently serve an embeddings endpoint, so this file
uses a small, dependency-free hashing embedding so the project runs with
ONLY an Anthropic key and numpy. It is good enough to demonstrate the
retrieval mechanics and to make the evals pass on this small corpus. The
embed_text() function is isolated specifically so you can later swap in a
real embedding model (Voyage AI, OpenAI text-embedding-3, or a local
sentence-transformers model) without touching anything else. That swap is
a great follow-on exercise.
"""

import os
import re
import glob
import math
import hashlib
import numpy as np
from anthropic import Anthropic

from .prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT

client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

GEN_MODEL = "claude-sonnet-4-6"
EMBED_DIM = 512


# ---------------------------------------------------------------------------
# STAGE 1: CHUNKING
# ---------------------------------------------------------------------------
def chunk_document(text: str, source: str) -> list[dict]:
    """
    Split a markdown spec into retrievable chunks.

    Strategy: split on markdown headers (## and ###), keeping each section
    together. Sections are the natural unit of a spec — "Filters", "Metric",
    "Reference Date", "Aggregation" each become their own chunk. This is
    "semantic chunking" by document structure, which beats naive
    fixed-character chunking for structured documents like specs.

    Each chunk carries metadata (source file, header) so answers can cite it.
    """
    # Split on headers but keep the header with its section.
    parts = re.split(r"\n(?=#{1,6}\s)", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Pull the header line (if any) for metadata/citation.
        header_match = re.match(r"#{1,6}\s+(.*)", part)
        header = header_match.group(1).strip() if header_match else "(intro)"
        # If a section is very large, sub-split it so chunks stay focused.
        if len(part) > 1200:
            sub = _split_long(part, 1200)
            for i, s in enumerate(sub):
                chunks.append({"text": s, "source": source,
                               "header": f"{header} (part {i+1})"})
        else:
            chunks.append({"text": part, "source": source, "header": header})
    return chunks


def _split_long(text: str, size: int) -> list[str]:
    """Split an over-long section on blank lines, packing up to `size` chars."""
    paras = text.split("\n\n")
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > size and buf:
            out.append(buf.strip())
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf.strip():
        out.append(buf.strip())
    return out


# ---------------------------------------------------------------------------
# STAGE 2: EMBEDDING
# ---------------------------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    """
    Turn text into a vector. SWAP THIS FUNCTION to use a real embedding model.

    This implementation is a deterministic bag-of-words hashing embedding:
    each token is hashed into one of EMBED_DIM buckets and counted, then the
    vector is L2-normalized. It captures lexical overlap (shared words ->
    similar vectors), which is enough for retrieval on this small,
    keyword-heavy spec corpus.

    A real embedding model would also capture *semantic* similarity
    (e.g. "churn" near "cancellation"). That is the upgrade path.
    """
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity. Vectors are pre-normalized, so this is a dot."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# STAGE 3 + 4: VECTOR STORE + RETRIEVAL
# ---------------------------------------------------------------------------
class VectorStore:
    """
    The simplest possible vector store: a list of chunks plus their vectors.
    `search` does brute-force cosine similarity. For thousands of chunks this
    is instant; for millions you'd reach for a real indexed DB. The interface
    (add / search) matches what pgvector or Chroma expose, so swapping is easy.
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self.vectors: list[np.ndarray] = []

    def add(self, chunk: dict):
        self.chunks.append(chunk)
        self.vectors.append(embed_text(chunk["text"]))

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-k most similar chunks to the query, with scores."""
        if not self.chunks:
            return []
        qv = embed_text(query)
        scored = [
            ({**chunk, "score": cosine_similarity(qv, vec)})
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:k]


def build_store(specs_dir: str) -> VectorStore:
    """Load every .md file in specs_dir, chunk, embed, and index it."""
    store = VectorStore()
    paths = glob.glob(os.path.join(specs_dir, "*.md"))
    if not paths:
        raise FileNotFoundError(
            f"No .md spec files found in {specs_dir}. "
            "Add at least one spec document."
        )
    for path in paths:
        with open(path) as f:
            text = f.read()
        source = os.path.basename(path)
        for chunk in chunk_document(text, source):
            store.add(chunk)
    return store


# ---------------------------------------------------------------------------
# STAGE 5: GROUNDED GENERATION
# ---------------------------------------------------------------------------
def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered, citable block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[chunk {i}] (from {c['source']} - {c['header']})\n{c['text']}"
        )
    return "\n\n".join(blocks)


def answer_question(store: VectorStore, question: str, k: int = 4) -> dict:
    """
    The full RAG call: retrieve -> ground -> generate.
    Returns the answer plus the chunks used (for inspection and eval).
    """
    chunks = store.search(question, k=k)
    context = format_context(chunks)
    system = RAG_SYSTEM_PROMPT.format(context=context)
    user = RAG_USER_PROMPT.format(question=question)

    resp = client.messages.create(
        model=GEN_MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text")
    return {"answer": answer, "chunks": chunks}


# ---------------------------------------------------------------------------
# Manual smoke test:  python -m src.rag
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(__file__)
    specs = os.path.join(here, "..", "specs")
    store = build_store(specs)
    print(f"Indexed {len(store.chunks)} chunks.\n")
    q = "What defines churn in this model?"
    result = answer_question(store, q)
    print("Q:", q)
    print("A:", result["answer"])
