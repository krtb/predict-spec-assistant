"""
run_evals.py — The evaluation harness.

This is the piece that separates a serious engineer from someone who built a
demo. Anyone can make a RAG system answer one question impressively. The
question that matters in production is: how often is it right, and how do you
KNOW when a change made it worse?

This harness:
  1. Runs every question in the golden set through the agent.
  2. Uses an LLM-as-judge (a second model call) to score each answer for
     CORRECTNESS and GROUNDING against the reference answer.
  3. Handles the refusal cases specially — for out-of-scope questions, the
     correct behavior is to refuse, and we reward that.
  4. Prints per-question scores and an aggregate, with a pass/fail threshold.

Run it before and after any change to prompts.py or rag.py. If the aggregate
score drops, your "improvement" was a regression. That is the entire point
of an eval suite — it turns "seems better" into a number.

Run:  python -m evals.run_evals
"""

import os
import json
from anthropic import Anthropic

from src.agent import SpecAgent
from src.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT

client = Anthropic()
JUDGE_MODEL = "claude-sonnet-4-6"

# Aggregate score below this fraction of the max = suite fails.
PASS_THRESHOLD = 0.80


def load_golden_set() -> list[dict]:
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "golden_set.json")) as f:
        return json.load(f)


def judge_answer(question: str, reference: str, answer: str) -> dict:
    """
    LLM-as-judge: a second model call scores the answer against the reference.
    Returns {correctness: 0-2, grounding: 0-2, reasoning: str}.
    """
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": JUDGE_USER_PROMPT.format(
                question=question, reference=reference, answer=answer
            ),
        }],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    # The judge is told to return only JSON, but strip fences just in case.
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"correctness": 0, "grounding": 0,
                "reasoning": f"Could not parse judge output: {raw[:80]}"}


def detect_refusal(answer: str) -> bool:
    """Heuristic: did the assistant correctly decline to answer?"""
    markers = [
        "not specified in the provided spec",
        "not in the provided spec",
        "not contained in",
        "cannot find",
        "does not specify",
        "isn't specified",
        "is not specified",
    ]
    low = answer.lower()
    return any(m in low for m in markers)


def main():
    here = os.path.dirname(__file__)
    specs = os.path.join(here, "..", "specs")
    agent = SpecAgent(specs)
    golden = load_golden_set()

    print(f"Running {len(golden)} eval cases against "
          f"{len(agent.store.chunks)} indexed chunks...\n")
    print("=" * 72)

    total_correctness = 0
    total_grounding = 0
    refusal_results = []
    max_per_case = 4  # 2 correctness + 2 grounding

    for case in golden:
        answer = agent.ask(case["question"], verbose=False)

        if case.get("expect_refusal"):
            # For refusal cases, correct behavior is declining to answer.
            refused = detect_refusal(answer)
            refusal_results.append(refused)
            status = "PASS" if refused else "FAIL"
            print(f"\n[{case['id']}]  (refusal case)  {status}")
            print(f"  Q: {case['question']}")
            print(f"  A: {answer[:120].strip()}...")
            # A correct refusal scores full marks; a hallucination scores 0.
            score = max_per_case if refused else 0
            total_correctness += 2 if refused else 0
            total_grounding += 2 if refused else 0
            continue

        verdict = judge_answer(case["question"], case["reference"], answer)
        c = verdict.get("correctness", 0)
        g = verdict.get("grounding", 0)
        total_correctness += c
        total_grounding += g

        print(f"\n[{case['id']}]  correctness={c}/2  grounding={g}/2")
        print(f"  Q: {case['question']}")
        print(f"  A: {answer[:120].strip()}...")
        print(f"  judge: {verdict.get('reasoning', '')}")

    print("\n" + "=" * 72)

    n = len(golden)
    max_total = n * max_per_case
    achieved = total_correctness + total_grounding
    pct = achieved / max_total if max_total else 0

    print("AGGREGATE RESULTS")
    print(f"  Correctness: {total_correctness}/{n * 2}")
    print(f"  Grounding:   {total_grounding}/{n * 2}")
    if refusal_results:
        passed = sum(refusal_results)
        print(f"  Refusal handling: {passed}/{len(refusal_results)} correct")
    print(f"  Overall score: {achieved}/{max_total} = {pct:.0%}")
    print(f"  Threshold: {PASS_THRESHOLD:.0%}")
    print(f"  SUITE: {'PASS' if pct >= PASS_THRESHOLD else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
