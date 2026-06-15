# Predict Spec Assistant

A retrieval-augmented (RAG) question-answering tool for Predict model
specification documents, with a tool-using agent layer and an evaluation
suite. Built to ingest churn-model specs and answer grounded questions
about filters, metrics, reference dates, and aggregations. Ships with a
synthetic example spec (fictional customer "Northwind Software").

## Why this project exists

This is a learning + portfolio project that demonstrates the four
LLM-native engineering skills most commonly screened for in Applied AI
Engineer / Forward Deployed Engineer roles:

1. **RAG architecture** — document chunking, embeddings, vector search,
   grounded generation (`src/rag.py`)
2. **Agent framework** — a tool-using loop that decides when to search
   the spec vs. answer directly (`src/agent.py`)
3. **Evals** — a golden-dataset test harness with an LLM-as-judge scorer
   that measures answer quality and catches regressions (`evals/`)
4. **Prompt engineering at production scale** — grounding, citation,
   refusal-when-unsure prompts (`src/prompts.py`)

The domain (Predict churn-model specs) is deliberately chosen to overlap
with real Customer Engineer work, so building it reinforces product ramp.

## Architecture

```
spec.md ──► chunk ──► embed ──► vector store (local, in-memory)
                                      │
user question ──► embed ──► similarity search ──► top-k chunks
                                      │
                          [AGENT decides: search or answer]
                                      │
                    grounded prompt + chunks ──► Claude ──► cited answer
                                      │
                          eval harness scores answer vs. golden set
```

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."   # Windows: set ANTHROPIC_API_KEY=...

# 4. Run the assistant interactively
python -m src.agent

# 5. Run the eval suite
python -m evals.run_evals
```

## What to read first (learning order)

1. `src/prompts.py` — see how grounded prompts are written
2. `src/rag.py` — the core retrieval pipeline, top to bottom
3. `src/agent.py` — how the tool-using loop wraps the RAG layer
4. `evals/golden_set.json` — the test questions and expected answers
5. `evals/run_evals.py` — how answers get scored

## Extending it

- Drop more `.md` spec files into `specs/` — they get indexed automatically
- Add questions to `evals/golden_set.json` to expand coverage
- Swap the in-memory store for a real vector DB (pgvector / Chroma) as a
  follow-on exercise — the interface in `rag.py` is designed for this
