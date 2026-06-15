"""
agent.py — A tool-using agent that wraps the RAG pipeline.

This is the "agent framework" piece. It is deliberately built on the raw
Anthropic tool-use API rather than LangChain/LangGraph, so you can see what
an agent actually IS underneath the frameworks:

    a loop where the model is given tools, decides to call one, you run it,
    feed the result back, and let the model continue until it produces a
    final answer.

That loop — model -> tool call -> tool result -> model -> ... -> answer — is
the entire concept. LangGraph adds state machines and branching on top, but
this is the core. Once you understand this file, LangGraph will read as
"oh, it's this with more structure."

The single tool here is `search_spec`, which calls into rag.py. The agent
decides whether a question needs a spec search at all (see AGENT_SYSTEM_PROMPT
in prompts.py).

Run interactively:  python -m src.agent
"""

import os
import json
from anthropic import Anthropic

from .rag import build_store, answer_question
from .prompts import AGENT_SYSTEM_PROMPT

client = Anthropic()
AGENT_MODEL = "claude-sonnet-4-6"

# The tool definition. This JSON schema is how the model knows the tool
# exists and what arguments it takes. This is the same shape you'll use
# for any production tool-use system.
TOOLS = [
    {
        "name": "search_spec",
        "description": (
            "Search the indexed Predict specification documents and return "
            "a grounded, cited answer to a spec-content question. Use this "
            "for any question about filters, metrics, churn/renewal "
            "definitions, reference dates, aggregations, or data sources."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The spec question to look up.",
                }
            },
            "required": ["query"],
        },
    }
]


class SpecAgent:
    """Holds the vector store and runs the tool-use loop for each question."""

    def __init__(self, specs_dir: str):
        self.store = build_store(specs_dir)

    def _run_tool(self, name: str, args: dict) -> str:
        """Execute a tool call. Currently only search_spec exists."""
        if name == "search_spec":
            result = answer_question(self.store, args["query"])
            return result["answer"]
        return f"Unknown tool: {name}"

    def ask(self, question: str, verbose: bool = True) -> str:
        """
        Run the full agent loop for one user question.
        Returns the agent's final text answer.
        """
        messages = [{"role": "user", "content": question}]

        # The loop. We cap iterations so a misbehaving model can't spin forever.
        for _ in range(5):
            resp = client.messages.create(
                model=AGENT_MODEL,
                max_tokens=1000,
                system=AGENT_SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # If the model wants to use a tool, run it and feed the result back.
            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        if verbose:
                            print(f"  [agent] searching spec: "
                                  f"\"{block.input.get('query', '')}\"")
                        output = self._run_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        })
                messages.append({"role": "user", "content": tool_results})
                continue  # let the model continue with the tool output

            # Otherwise the model produced a final answer.
            return "".join(b.text for b in resp.content if b.type == "text")

        return "(agent stopped after max iterations)"


def main():
    here = os.path.dirname(__file__)
    specs = os.path.join(here, "..", "specs")
    agent = SpecAgent(specs)
    print(f"Predict Spec Assistant ready. Indexed "
          f"{len(agent.store.chunks)} chunks.")
    print("Ask a question about the spec (or 'quit'):\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        answer = agent.ask(q)
        print(f"\nassistant > {answer}\n")


if __name__ == "__main__":
    main()
