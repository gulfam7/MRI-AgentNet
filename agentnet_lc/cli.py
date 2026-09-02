"""Command-line entry point.

Replaces the Tkinter file dialog and the ``input()`` prompt in
``agent_multi_meta_learning.py``, which made the pipeline impossible to run
headless, in CI, or over a batch of scans.

    python -m agentnet_lc.cli --input scan.png --ground-truth "motion corrupted"
    python -m agentnet_lc.cli --input data/ --glob "*.png" --jsonl traces/cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from agentnet_lc.graph import build_graph, draw_ascii
from agentnet_lc.llms import build_models
from agentnet_lc.prompts import image_data_uri
from agentnet_lc.schemas import CorruptionType
from agentnet_lc.state import initial_state


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MRI-AgentNet pipeline.")
    parser.add_argument("--input", required=False, help="Image file or directory.")
    parser.add_argument("--glob", default="*.png", help="Pattern when --input is a directory.")
    parser.add_argument("--ground-truth", default=None, help="Known corruption, if synthetic.")
    parser.add_argument("--jsonl", default=None, help="Append one trace row per case.")
    parser.add_argument("--router-checkpoint", default=None)
    parser.add_argument("--print-graph", action="store_true", help="Show topology and exit.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _cases(args: argparse.Namespace) -> list[Path]:
    target = Path(args.input)
    if target.is_dir():
        return sorted(target.glob(args.glob))
    return [target]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    models = build_models()
    graph = build_graph(
        models,
        router_checkpoint=Path(args.router_checkpoint) if args.router_checkpoint else None,
    )

    if args.print_graph:
        print(draw_ascii(graph))
        return 0

    if not args.input:
        print("--input is required unless --print-graph is given", file=sys.stderr)
        return 2

    truth = CorruptionType.coerce(args.ground_truth) if args.ground_truth else None
    sink = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None

    try:
        for path in _cases(args):
            state = initial_state(
                case_id=uuid.uuid4().hex,
                source_path=str(path),
                image_url=image_data_uri(path),
                ground_truth=truth,
            )
            final = graph.invoke(state)

            decision = final.get("pi_final")
            route = final.get("route")
            print(f"\n=== {path.name} ===")
            for opinion in final.get("opinions", []):
                label = opinion.classification.value if opinion.classification else "FAILED"
                print(f"  {opinion.agent:16} {label:18} conf={opinion.confidence:.2f}")
            if decision:
                print(f"  final classification: {decision.classification.value}")
            if route:
                print(f"  route: {route.model.value}  ({route.source})")
            for err in final.get("errors", []):
                print(f"  ! {err}")

            if sink:
                sink.write(
                    json.dumps(
                        {
                            "case_id": final["case_id"],
                            "source_path": final["source_path"],
                            "ground_truth": truth.value if truth else None,
                            "opinions": [o.model_dump(mode="json") for o in final.get("opinions", [])],
                            "pi_final": decision.model_dump(mode="json") if decision else None,
                            "route": route.model_dump(mode="json") if route else None,
                            "timings": final.get("timings", {}),
                            "errors": final.get("errors", []),
                        }
                    )
                    + "\n"
                )
                sink.flush()
    finally:
        if sink:
            sink.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
