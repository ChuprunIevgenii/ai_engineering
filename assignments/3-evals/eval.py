# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "langfuse>=5.0.1,<6",
#   "litellm==1.104.3",
# ]
# ///
"""
Quick structural validation of the homework (no API calls).

Shows progress per TODO: seed cases, keyword_grader, failure taxonomy, judge questions,
labels in outputs_B.json, adversarial cases + safety rubric. Also reports optional Prompt C.

Run:
  uv run eval.py
"""

import sys

try:
    import starter  # type: ignore
except ImportError as e:
    print("Could not import starter.py:", e)
    print("Run this script from the homework directory: uv run eval.py")
    sys.exit(1)


def main() -> None:
    print("=" * 64)
    print("  Homework structure check (no API calls)")
    print("=" * 64)

    all_ok = True
    for todo, check in starter.TODO_CHECKS.items():
        errors = check()
        print(f"\n  [{'x' if not errors else ' '}] {todo}")
        for e in errors:
            print(f"      - {e}")
        all_ok = all_ok and not errors

    if not starter.check_taxonomy() and not starter.check_judge_questions():
        taxonomy_names = {ft.get("name") for ft in starter.FAILURE_TAXONOMY}
        missing = sorted(n for n in taxonomy_names - set(starter.JUDGE_QUESTIONS) if n)
        if missing:
            print(f"\n  Note: failure types without a judge question: {', '.join(missing)}")

    print(f"\n  [{'x' if starter.prompt_c_ready() else ' '}] Optional: SYSTEM_PROMPT_C")

    print("\n" + "=" * 64)
    if all_ok:
        print("  All 6 TODOs pass. Run the full pipeline: uv run starter.py")
        print("=" * 64)
        sys.exit(0)
    print("  Some TODOs are not done yet. Typical order:")
    print("  1-2 -> uv run starter.py --dump -> read + label -> 3-5 (--kappa) -> 6 -> full run")
    print("=" * 64)
    sys.exit(1)


if __name__ == "__main__":
    main()
