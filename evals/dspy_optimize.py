"""Optimise the classifier prompt with DSPy instead of hand-tuning it.

The current prompts were written by hand and patched whenever something broke
("You should never omit any of the 5 points below"). DSPy treats the prompt as
parameters to be *compiled* against a metric: you declare the signature, supply
labelled examples, and an optimiser searches instructions and few-shot
demonstrations for you.

This is the highest-leverage framework claim available here, because it is
uncommon and it maps onto a real defect in the project -- the few-shot
exemplars in ``utils/few_shot_gpt4o.py`` were three images chosen by hand, with
no evidence they are the best three.

    python evals/dspy_optimize.py --train data/train.jsonl --dev data/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["motion corrupted", "undersampled", "noisy", "no corruption"]


def build_program():
    """A two-stage module: reason about artefacts, then commit to a label."""
    import dspy

    class ClassifyCorruption(dspy.Signature):
        """Identify the dominant corruption in an MRI scan."""

        image: dspy.Image = dspy.InputField(desc="Axial MRI slice.")
        reasoning: str = dspy.OutputField(desc="Artefact-specific evidence.")
        classification: str = dspy.OutputField(desc=f"One of: {', '.join(LABELS)}")
        confidence: float = dspy.OutputField(desc="Probability the label is correct, 0-1.")

    class CorruptionClassifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.classify = dspy.ChainOfThought(ClassifyCorruption)

        def forward(self, image):
            return self.classify(image=image)

    return CorruptionClassifier()


def accuracy_metric(example, prediction, trace=None) -> float:
    """What the optimiser maximises. Ties are broken toward calibrated answers."""
    predicted = (getattr(prediction, "classification", "") or "").strip().lower()
    correct = float(predicted == example.classification.strip().lower())
    if trace is not None:  # bootstrapping mode wants a hard pass/fail
        return correct > 0.5
    try:
        confidence = float(getattr(prediction, "confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return 0.9 * correct + 0.1 * (1.0 - (confidence - correct) ** 2)


def load_examples(path: Path) -> list:
    import dspy

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        dspy.Example(
            image=dspy.Image.from_file(r["source_path"]),
            classification=r["ground_truth"],
        ).with_inputs("image")
        for r in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-4o-2024-11-20")
    parser.add_argument("--out", type=Path, default=Path("evals/compiled_classifier.json"))
    args = parser.parse_args()

    import dspy

    dspy.configure(lm=dspy.LM(args.model, temperature=0.0))

    trainset, devset = load_examples(args.train), load_examples(args.dev)
    program = build_program()

    baseline = sum(
        accuracy_metric(ex, program(image=ex.image)) for ex in devset
    ) / len(devset)
    print(f"baseline dev score: {baseline:.3f}")

    # MIPROv2 searches both the instruction text and which demonstrations to
    # include -- i.e. it replaces the hand-picked few-shot exemplars.
    optimiser = dspy.MIPROv2(metric=accuracy_metric, auto="light")
    compiled = optimiser.compile(program, trainset=trainset, valset=devset)

    optimised = sum(
        accuracy_metric(ex, compiled(image=ex.image)) for ex in devset
    ) / len(devset)
    print(f"optimised dev score: {optimised:.3f}  (delta {optimised - baseline:+.3f})")

    compiled.save(str(args.out))
    print(f"saved compiled program to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
