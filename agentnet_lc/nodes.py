"""Graph nodes.

Each node is a plain function of ``(state, models) -> partial state``. Two
consequences worth knowing for an interview: the functions are unit-testable
without a graph, and LangGraph can checkpoint between them, so a failure in
arbitration does not force you to re-pay for the four calls that preceded it.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from agentnet_lc import prompts
from agentnet_lc.llms import ModelBundle
from agentnet_lc.schemas import AgentOpinion, ImageSpace
from agentnet_lc.state import PipelineState, successful_opinions

log = logging.getLogger(__name__)


def _timed(name: str, fn: Callable[[], dict]) -> dict:
    """Run a node body, recording latency and converting exceptions to state.

    A node that raises kills the graph run. For a panel of independent
    evaluators that is the wrong default: losing Gemini should degrade the
    ensemble, not abort the case.
    """
    start = time.perf_counter()
    try:
        update = fn()
    except Exception as exc:  # noqa: BLE001 - deliberately broad at the node boundary
        log.exception("node %s failed", name)
        return {"errors": [f"{name}: {exc}"], "timings": {name: time.perf_counter() - start}}
    update.setdefault("timings", {})[name] = time.perf_counter() - start
    return update


def classify_space(state: PipelineState, models: ModelBundle) -> dict:
    def body() -> dict:
        result = models.space_classifier.invoke(
            prompts.space_messages(state["image_url"])
        )
        return {"space": result.space}

    return _timed("classify_space", body)


def convert_kspace(state: PipelineState, models: ModelBundle) -> dict:
    """Inverse-FFT k-space to image space, then re-point the state at the result.

    Delegates to the existing utility so the numerical path is unchanged.
    """

    def body() -> dict:
        from utils import data_processing_confidence as dp

        volume = dp.convert_kspace_to_image_space(state["source_path"])
        if volume is None:
            raise RuntimeError("k-space to image-space conversion returned None")
        png_path = dp.save_image_as_png(volume)
        if not png_path:
            raise RuntimeError("failed to write converted image")
        return {
            "image_url": prompts.image_data_uri(png_path),
            "source_path": png_path,
            "space": ImageSpace.IMAGE,
        }

    return _timed("convert_kspace", body)


def _assistant(state: PipelineState, runnable, agent_name: str) -> dict:
    def body() -> dict:
        assessment = runnable.invoke(
            prompts.assistant_messages(
                state["image_url"],
                "Classify the dominant corruption in this MRI scan.",
            )
        )
        return {
            "opinions": [
                AgentOpinion(
                    agent=agent_name,
                    classification=assessment.classification,
                    confidence=assessment.confidence,
                    reasoning=assessment.reasoning,
                )
            ]
        }

    update = _timed(agent_name, body)
    # Preserve the failure as a first-class opinion so downstream stages can see
    # which evaluator was missing rather than silently working with a short list.
    if "opinions" not in update and "errors" in update:
        update["opinions"] = [AgentOpinion.failed(agent_name, update["errors"][0])]
    return update


def assistant_primary(state: PipelineState, models: ModelBundle) -> dict:
    return _assistant(state, models.assistant_primary, "gpt4o")


def assistant_secondary(state: PipelineState, models: ModelBundle) -> dict:
    return _assistant(state, models.assistant_secondary, "gemini")


def radiologist(state: PipelineState, models: ModelBundle) -> dict:
    def body() -> dict:
        review = models.radiologist.invoke(
            prompts.radiologist_messages(
                state["image_url"], successful_opinions(state)
            )
        )
        return {
            "radiologist": review,
            "opinions": [
                AgentOpinion(
                    agent="radiologist",
                    classification=review.classification,
                    confidence=review.confidence,
                    reasoning=review.reasoning,
                )
            ],
        }

    return _timed("radiologist", body)


def pi_independent(state: PipelineState, models: ModelBundle) -> dict:
    def body() -> dict:
        decision = models.principal.invoke(
            prompts.pi_independent_messages(state["image_url"])
        )
        return {
            "pi_independent": decision,
            "opinions": [
                AgentOpinion(
                    agent="pi_independent",
                    classification=decision.classification,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                )
            ],
        }

    return _timed("pi_independent", body)


def pi_arbitration(state: PipelineState, models: ModelBundle) -> dict:
    def body() -> dict:
        peers = [o for o in successful_opinions(state) if o.agent != "pi_independent"]
        decision = models.principal.invoke(
            prompts.pi_arbitration_messages(
                state["image_url"],
                state.get("pi_independent"),
                peers,
                state.get("radiologist"),
            )
        )
        return {
            "pi_final": decision,
            "opinions": [
                AgentOpinion(
                    agent="pi_final",
                    classification=decision.classification,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                )
            ],
        }

    return _timed("pi_arbitration", body)


# --------------------------------------------------------------------------
# Conditional edge
# --------------------------------------------------------------------------


def needs_kspace_conversion(state: PipelineState) -> str:
    """Branch key for the k-space detour."""
    return "convert" if state.get("space") == ImageSpace.KSPACE else "assess"
