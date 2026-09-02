"""LangSmith evaluation harness.

This is the Phase-1 evaluation work from the interview handbook, expressed in
the framework rather than hand-rolled: a versioned dataset, custom evaluators,
and a comparison view across pipeline variants.

    export LANGSMITH_API_KEY=...
    export LANGSMITH_TRACING=true
    python evals/langsmith_eval.py --upload data/labelled.jsonl
    python evals/langsmith_eval.py --run

The evaluators below are the metrics that actually matter for this task
(per-class correctness and calibration), not generic text similarity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASET_NAME = "mri-agentnet-corruption"


# --------------------------------------------------------------------------
# Evaluators
# --------------------------------------------------------------------------


def exact_label(outputs: dict, reference_outputs: dict) -> dict:
    """The headline metric: did the arbitrated classification match ground truth."""
    predicted = (outputs or {}).get("classification")
    expected = (reference_outputs or {}).get("classification")
    return {"key": "label_correct", "score": float(predicted == expected)}


def brier(outputs: dict, reference_outputs: dict) -> dict:
    """Calibration as a proper scoring rule.

    Rewards honest confidence rather than confident-sounding output, which is
    the failure mode of verbalised LLM confidence. Lower is better, so the
    score is inverted to keep 'higher is better' in the LangSmith UI.
    """
    predicted = (outputs or {}).get("classification")
    expected = (reference_outputs or {}).get("classification")
    confidence = float((outputs or {}).get("confidence") or 0.5)
    correct = float(predicted == expected)
    return {"key": "calibration", "score": 1.0 - (confidence - correct) ** 2}


def routed_correctly(outputs: dict, reference_outputs: dict) -> dict:
    """Task-level metric: did the case reach the right restoration backend."""
    expected_route = (reference_outputs or {}).get("route")
    return {
        "key": "route_correct",
        "score": float((outputs or {}).get("route") == expected_route),
    }


def panel_unanimous(outputs: dict) -> dict:
    """Diagnostic, not a quality metric.

    Cross-tabulating this against label_correct answers the question the paper
    needs: does agreement predict correctness, or do the agents fail together?
    """
    labels = [o.get("classification") for o in (outputs or {}).get("opinions", [])]
    labels = [x for x in labels if x]
    return {"key": "unanimous", "score": float(len(set(labels)) == 1 if labels else 0.0)}


EVALUATORS = [exact_label, brier, routed_correctly, panel_unanimous]


# --------------------------------------------------------------------------
# Dataset upload
# --------------------------------------------------------------------------


def upload(path: Path) -> None:
    """Push a labelled JSONL file to LangSmith as a versioned dataset."""
    from langsmith import Client

    client = Client()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="MRI scans with known synthetic corruption and best-SSIM route.",
        )

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"source_path": r["source_path"]} for r in rows],
        outputs=[
            {"classification": r["ground_truth"], "route": r.get("best_route")}
            for r in rows
        ],
    )
    print(f"uploaded {len(rows)} examples to '{DATASET_NAME}'")


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------


def run_experiment(variant: str = "full_pipeline") -> None:
    """Evaluate one pipeline variant over the dataset.

    Running this for several variants gives the ablation table the handbook
    asks for -- single model, majority vote, full panel -- side by side in one
    LangSmith comparison view instead of scattered console output.
    """
    from langsmith import Client

    from agentnet_lc.graph import build_graph
    from agentnet_lc.llms import build_models
    from agentnet_lc.prompts import image_data_uri
    from agentnet_lc.state import initial_state

    graph = build_graph(build_models())

    def target(inputs: dict) -> dict:
        path = inputs["source_path"]
        final = graph.invoke(initial_state("eval", path, image_data_uri(path)))
        decision = final.get("pi_final")
        route = final.get("route")
        return {
            "classification": decision.classification.value if decision else None,
            "confidence": decision.confidence if decision else 0.0,
            "route": route.model.value if route else None,
            "opinions": [
                {
                    "agent": o.agent,
                    "classification": o.classification.value if o.classification else None,
                }
                for o in final.get("opinions", [])
            ],
        }

    Client().evaluate(
        target,
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix=variant,
        max_concurrency=4,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", type=Path, help="Labelled JSONL to push as a dataset.")
    parser.add_argument("--run", action="store_true", help="Evaluate the pipeline.")
    parser.add_argument("--variant", default="full_pipeline")
    args = parser.parse_args()

    if args.upload:
        upload(args.upload)
    if args.run:
        run_experiment(args.variant)
    if not args.upload and not args.run:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
