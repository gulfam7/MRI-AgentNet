"""The graph's shared state.

In ``agent_multi_meta_learning.py`` this state lives as ~15 local variables
threaded down one 270-line ``process()`` method. Making it an explicit,
typed object is what allows LangGraph to run branches in parallel, checkpoint
between steps, and resume a failed run without re-paying for earlier LLM calls.

The ``Annotated[..., operator.add]`` reducers matter: the two assistant nodes
execute concurrently and both append to ``opinions``. Without a reducer,
LangGraph raises ``InvalidUpdateError`` on the concurrent write.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from agentnet_lc.schemas import (
    AgentOpinion,
    CorruptionType,
    ImageSpace,
    PrincipalDecision,
    RadiologistReview,
    RouteDecision,
)


class PipelineState(TypedDict, total=False):
    """Everything the pipeline knows about one MRI case."""

    # --- inputs ---
    case_id: str
    source_path: str
    image_url: str
    """Either an https URL or a `data:image/png;base64,...` URI."""
    ground_truth: CorruptionType | None
    """Known when the corruption was applied synthetically. Enables the
    verifiable reward described in the RL handbook; None in real use."""

    # --- stage outputs ---
    space: ImageSpace | None
    opinions: Annotated[list[AgentOpinion], operator.add]
    radiologist: RadiologistReview | None
    pi_independent: PrincipalDecision | None
    pi_final: PrincipalDecision | None
    route: RouteDecision | None

    # --- diagnostics ---
    errors: Annotated[list[str], operator.add]
    timings: Annotated[dict[str, float], operator.or_]


def initial_state(
    case_id: str,
    source_path: str,
    image_url: str,
    ground_truth: CorruptionType | None = None,
) -> PipelineState:
    return PipelineState(
        case_id=case_id,
        source_path=source_path,
        image_url=image_url,
        ground_truth=ground_truth,
        space=None,
        opinions=[],
        radiologist=None,
        pi_independent=None,
        pi_final=None,
        route=None,
        errors=[],
        timings={},
    )


def successful_opinions(state: PipelineState) -> list[AgentOpinion]:
    """Assistants that actually returned a usable label.

    The original code indexed ``classification_results[0]`` and ``[1]``
    unconditionally, so a single provider outage raised IndexError. Every
    consumer goes through this helper instead.
    """
    return [o for o in state.get("opinions", []) if o.ok and o.classification is not None]
