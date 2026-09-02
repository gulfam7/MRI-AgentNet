"""The routing node: learned meta-model, with a principled fallback.

Two defects in the original router are fixed here.

1. **No ground-truth feature.** ``model_selection/data_generation.py`` put
   ``true_corruption`` into the feature vector and derived the label from a
   variable that copied it 90% of the time, so the network learned an identity
   map. The featuriser below sees only what is available at inference.

2. **Same features at train and inference.** The old inference vector filled
   slot 0 (trained as ground truth) with the PI's *independent* guess and never
   used the arbitrated decision at all. Here the featuriser is a single function
   imported by both the trainer and this node, so the two cannot drift apart.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from agentnet_lc.schemas import CorruptionType, RestorationModel, RouteDecision
from agentnet_lc.state import PipelineState, successful_opinions

log = logging.getLogger(__name__)

# Order is part of the model contract. Changing it invalidates the checkpoint.
AGENT_ORDER = ["gpt4o", "gemini", "radiologist", "pi_independent", "pi_final"]
LABEL_ORDER = [
    CorruptionType.UNDERSAMPLED,
    CorruptionType.MOTION,
    CorruptionType.NOISY,
]
ROUTE_ORDER = [
    RestorationModel.MOTION_CORRECTION,
    RestorationModel.DENOISING,
    RestorationModel.RECONSTRUCTION,
]

LABEL_TO_ROUTE = {
    CorruptionType.MOTION: RestorationModel.MOTION_CORRECTION,
    CorruptionType.NOISY: RestorationModel.DENOISING,
    CorruptionType.UNDERSAMPLED: RestorationModel.RECONSTRUCTION,
    CorruptionType.NONE: RestorationModel.NONE,
}

FEATURE_DIM = len(AGENT_ORDER) * len(LABEL_ORDER) + 5  # 15 + 5 = 20


def featurise(opinions: list) -> list[float]:
    """Confidence-weighted one-hots plus explicit disagreement signals.

    Shared by training and inference. The five trailing features give the
    network something a plain argmax could not compute: how split the panel is.
    """
    by_agent = {o.agent: o for o in opinions if o.ok and o.classification is not None}

    feats: list[float] = []
    for agent in AGENT_ORDER:
        opinion = by_agent.get(agent)
        block = [0.0] * len(LABEL_ORDER)
        if opinion is not None and opinion.classification in LABEL_ORDER:
            block[LABEL_ORDER.index(opinion.classification)] = float(opinion.confidence)
        feats.extend(block)

    labels = [o.classification for o in by_agent.values()]
    if labels:
        counts = Counter(labels)
        top_label, top_count = counts.most_common(1)[0]
        agreement = top_count / len(labels)
        distinct = len(counts) / len(LABEL_ORDER)
    else:
        agreement, distinct = 0.0, 1.0

    pi_final = by_agent.get("pi_final") or by_agent.get("pi_independent")
    feats.extend(
        [
            agreement,
            distinct,
            len(by_agent) / len(AGENT_ORDER),  # coverage: how many agents answered
            float(pi_final.confidence) if pi_final else 0.5,
            1.0,  # bias
        ]
    )
    return feats


def _weighted_vote(opinions: list) -> tuple[RestorationModel, str]:
    """Fallback when no checkpoint is present.

    Replaces ``utils/model_selector.py``, whose fallback branch read
    ``if 'motion' or 'artifacts' in classification`` -- always true, so it
    returned motion_correction unconditionally.
    """
    scores: dict[CorruptionType, float] = {}
    weights = {"pi_final": 2.0, "pi_independent": 1.5, "radiologist": 1.25}
    for o in opinions:
        if o.classification is None:
            continue
        scores[o.classification] = scores.get(o.classification, 0.0) + (
            weights.get(o.agent, 1.0) * float(o.confidence)
        )
    if not scores:
        return RestorationModel.NONE, "no usable classification"
    winner = max(scores, key=scores.__getitem__)
    return LABEL_TO_ROUTE[winner], "confidence-weighted vote (no meta-model checkpoint)"


def _load_meta_model(checkpoint: Path):
    """Load the trained router if its checkpoint matches the current feature dim."""
    try:
        import torch
        from model_selection.router_v2 import RouterV2  # trained with `featurise`
    except Exception as exc:  # noqa: BLE001
        log.debug("meta-model unavailable: %s", exc)
        return None, None
    if not checkpoint.exists():
        return None, None
    try:
        model = RouterV2(d_in=FEATURE_DIM, d_out=len(ROUTE_ORDER))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        return model, torch
    except Exception as exc:  # noqa: BLE001
        log.warning("meta-model checkpoint rejected (%s); using weighted vote", exc)
        return None, None


def route(state: PipelineState, checkpoint: Path | None = None) -> dict:
    opinions = successful_opinions(state)
    checkpoint = checkpoint or Path(__file__).resolve().parents[1] / (
        "model_selection/router_v2.pth"
    )

    model, torch = _load_meta_model(checkpoint)
    if model is None:
        selected, reason = _weighted_vote(opinions)
        return {"route": RouteDecision(model=selected, source=reason)}

    with torch.no_grad():
        logits = model(torch.tensor([featurise(opinions)], dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1)[0].tolist()

    index = max(range(len(ROUTE_ORDER)), key=probs.__getitem__)
    return {
        "route": RouteDecision(
            model=ROUTE_ORDER[index],
            source="meta_model",
            probabilities={r.value: round(p, 4) for r, p in zip(ROUTE_ORDER, probs)},
        )
    }
