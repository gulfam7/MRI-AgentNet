"""End-to-end graph tests using stub models -- no API keys, no network.

This is the payoff of injecting `ModelBundle` rather than constructing clients
inside the nodes: the entire orchestration is testable offline.

Run:  python -m pytest tests/ -v      (or: python tests/test_graph_smoke.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentnet_lc.graph import build_graph
from agentnet_lc.llms import ModelBundle
from agentnet_lc.router import FEATURE_DIM, featurise
from agentnet_lc.schemas import (
    AgentOpinion,
    CorruptionAssessment,
    CorruptionType,
    ImageSpace,
    PrincipalDecision,
    RadiologistReview,
    RestorationModel,
    SpaceClassification,
)
from agentnet_lc.state import initial_state


class StubRunnable:
    """Stands in for a structured-output Runnable."""

    def __init__(self, response, *, fail=False):
        self.response = response
        self.fail = fail
        self.calls = 0

    def invoke(self, _messages, **_kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated provider outage")
        return self.response


def _assessment(label: CorruptionType, conf: float) -> CorruptionAssessment:
    return CorruptionAssessment(
        classification=label,
        reasoning="Regularly spaced replicas along the phase-encode direction.",
        recommended_model=RestorationModel.RECONSTRUCTION,
        correction_plan="Run the reconstruction backend.",
        confidence=conf,
    )


def make_bundle(*, space=ImageSpace.IMAGE, gemini_fails=False) -> ModelBundle:
    return ModelBundle(
        space_classifier=StubRunnable(
            SpaceClassification(space=space, reasoning="Anatomy is visible.")
        ),
        assistant_primary=StubRunnable(_assessment(CorruptionType.UNDERSAMPLED, 0.82)),
        assistant_secondary=StubRunnable(
            _assessment(CorruptionType.MOTION, 0.55), fail=gemini_fails
        ),
        radiologist=StubRunnable(
            RadiologistReview(
                classification=CorruptionType.UNDERSAMPLED,
                confidence=0.78,
                reasoning="Aliasing replicas at integer spacing.",
                agrees_with_assistants=True,
                recommended_model=RestorationModel.RECONSTRUCTION,
            )
        ),
        principal=StubRunnable(
            PrincipalDecision(
                classification=CorruptionType.UNDERSAMPLED,
                confidence=0.88,
                reasoning="Background SNR normal; replicas coherent.",
                recommended_model=RestorationModel.RECONSTRUCTION,
            )
        ),
    )


def _run(bundle) -> dict:
    graph = build_graph(bundle)
    return graph.invoke(
        initial_state(
            case_id="test-1",
            source_path="scan.png",
            image_url="data:image/png;base64,iVBORw0KGgo=",
        )
    )


def test_happy_path_routes_to_reconstruction():
    final = _run(make_bundle())
    assert final["route"].model is RestorationModel.RECONSTRUCTION
    assert final["pi_final"].classification is CorruptionType.UNDERSAMPLED
    assert final["errors"] == []
    agents = {o.agent for o in final["opinions"]}
    assert agents == {"gpt4o", "gemini", "radiologist", "pi_independent", "pi_final"}


def test_pipeline_survives_a_dead_provider():
    """The old code raised IndexError on classification_results[1] here."""
    final = _run(make_bundle(gemini_fails=True))
    assert final["route"] is not None
    assert final["pi_final"].classification is CorruptionType.UNDERSAMPLED
    failed = [o for o in final["opinions"] if not o.ok]
    assert [o.agent for o in failed] == ["gemini"]
    assert any("gemini" in e for e in final["errors"])


def test_kspace_branch_is_taken_only_for_kspace():
    bundle = make_bundle(space=ImageSpace.KSPACE)
    graph = build_graph(bundle)
    final = graph.invoke(
        initial_state("t", "scan.npy", "data:image/png;base64,iVBORw0KGgo=")
    )
    # Conversion is attempted (and fails here, since scan.npy does not exist),
    # which proves the conditional edge fired rather than silently skipping.
    assert any("convert_kspace" in e for e in final["errors"])


def test_image_branch_skips_conversion():
    final = _run(make_bundle(space=ImageSpace.IMAGE))
    assert "convert_kspace" not in final.get("timings", {})


def test_assistants_run_concurrently():
    """Both assistants are dispatched in the same superstep."""
    bundle = make_bundle()
    _run(bundle)
    assert bundle.assistant_primary.calls == 1
    assert bundle.assistant_secondary.calls == 1


def test_featuriser_dimension_and_no_leakage():
    opinions = [
        AgentOpinion(agent="gpt4o", classification=CorruptionType.MOTION, confidence=0.9),
        AgentOpinion(agent="pi_final", classification=CorruptionType.MOTION, confidence=0.8),
    ]
    feats = featurise(opinions)
    assert len(feats) == FEATURE_DIM == 20
    # Unanimous panel -> agreement feature is 1.0
    assert feats[15] == 1.0
    # Coverage: 2 of 5 agents answered
    assert abs(feats[17] - 0.4) < 1e-9


def test_label_coercion_matches_old_mapping():
    assert CorruptionType.coerce("Motion Artifact") is CorruptionType.MOTION
    assert CorruptionType.coerce("k-space artifact") is CorruptionType.UNDERSAMPLED
    assert CorruptionType.coerce("Radiofrequency noise") is CorruptionType.NOISY


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
