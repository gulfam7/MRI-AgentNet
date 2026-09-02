"""Prompt construction.

Two things changed from the original prompt strings:

1. All formatting instructions are gone. The five-section contract, "do not use
   newline characters between numbered sections", and "if a confidence score is
   missing, provide your best estimate" were compensating for a regex parser.
   The schema does that job now, so the prompts only carry domain content.

2. Images are passed as base64 data URIs rather than public Dropbox links, so
   scans never leave the machine. ``upload_to_dropbox`` is no longer on the
   critical path.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agentnet_lc.schemas import AgentOpinion, PrincipalDecision, RadiologistReview


def image_data_uri(path: str | Path) -> str:
    """Inline an image so no third-party file host is involved."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _image_block(image_url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": image_url}}


def _user(image_url: str, text: str) -> HumanMessage:
    return HumanMessage(content=[{"type": "text", "text": text}, _image_block(image_url)])


# --------------------------------------------------------------------------
# Stage 1: image space vs k-space
# --------------------------------------------------------------------------

SPACE_SYSTEM = (
    "You are an MRI physicist. Decide whether the supplied array is displayed in "
    "image space (recognisable anatomy) or k-space (a centred bright DC region with "
    "symmetric high-frequency structure and no anatomy)."
)


def space_messages(image_url: str) -> list[BaseMessage]:
    return [
        SystemMessage(content=SPACE_SYSTEM),
        _user(image_url, "Which domain is this MRI data in?"),
    ]


# --------------------------------------------------------------------------
# Stage 2: the two independent assistants
# --------------------------------------------------------------------------

ASSISTANT_SYSTEM = (
    "You are an MRI quality-assessment agent. Identify the dominant corruption in "
    "the scan and recommend a restoration model.\n\n"
    "Diagnostic cues:\n"
    "- Motion: coherent ghosting replicated along the phase-encode direction, "
    "blurred edges, streaking that follows anatomy.\n"
    "- Undersampling: regularly spaced aliasing replicas at an integer spacing, "
    "wrap-around, preserved background SNR.\n"
    "- Noise: grainy speckle across the whole field including background air, "
    "reduced SNR without structured replicas.\n"
    "- No corruption: sharp edges, clean background, no periodic structure.\n\n"
    "Report the probability that your classification is correct as your confidence. "
    "Do not inflate it; a wrong answer stated at 0.95 is worse than one stated at 0.5."
)


def assistant_messages(image_url: str, user_prompt: str) -> list[BaseMessage]:
    return [
        SystemMessage(content=ASSISTANT_SYSTEM),
        _user(image_url, user_prompt),
    ]


# --------------------------------------------------------------------------
# Stage 3: few-shot radiologist
# --------------------------------------------------------------------------

# Same three exemplars as utils/few_shot_gpt4o.py. Swap these for local paths
# and they stop depending on the Dropbox links surviving.
FEW_SHOT_EXEMPLARS: list[tuple[str, str, str]] = [
    (
        "https://www.dropbox.com/scl/fi/tti26ukmr27y8rtwk8nus/608.png?rlkey=fn46e0ejd19p8fz78f0vctg92&st=tleid5du&dl=1",
        "Blurring and streaking along tissue boundaries from patient movement.",
        "motion corrupted",
    ),
    (
        "https://www.dropbox.com/scl/fi/o2xrwxbxmrqvqiyal49c6/103.png?rlkey=jsoosp7e7dpg90cvclldh4519&st=pj2dhcuo&dl=1",
        "Repetitive aliasing replicas and loss of effective resolution.",
        "undersampled",
    ),
    (
        "https://www.dropbox.com/scl/fi/u45y588tauanlx71iy3lb/602.png?rlkey=wwu9wigpwfl0zu75jqez9il7b&st=75yqztvv&dl=1",
        "High-frequency speckle throughout, including background air.",
        "noisy",
    ),
]

RADIOLOGIST_SYSTEM = (
    "You are a radiologist specialising in MRI artefact identification. First form "
    "your own opinion from the image alone, then state whether the assistants' "
    "classifications are consistent with it. Do not defer to them."
)


def radiologist_messages(
    image_url: str, opinions: list[AgentOpinion]
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=RADIOLOGIST_SYSTEM)]

    for exemplar_url, description, label in FEW_SHOT_EXEMPLARS:
        messages.append(_user(exemplar_url, description))
        messages.append(AIMessage(content=label))

    if opinions:
        summary = "\n".join(
            f"- {o.agent}: {o.classification.value if o.classification else 'no answer'} "
            f"(confidence {o.confidence:.2f}) -- {o.reasoning[:400]}"
            for o in opinions
        )
    else:
        summary = "- No assistant returned a usable classification."

    messages.append(
        _user(
            image_url,
            "Classify the corruption in this scan.\n\n"
            f"Assistant classifications for reference:\n{summary}",
        )
    )
    return messages


# --------------------------------------------------------------------------
# Stage 4: principal investigator, independent then arbitrating
# --------------------------------------------------------------------------

PI_INDEPENDENT_SYSTEM = (
    "You are the principal investigator, a senior MRI expert fine-tuned on this "
    "corruption taxonomy. Classify this scan using only the image. You have not "
    "been shown any other opinion; do not speculate about one."
)

PI_ARBITRATION_SYSTEM = (
    "You are the principal investigator arbitrating a case you have already judged "
    "independently.\n\n"
    "Weighting policy: your own fine-tuned judgement is the highest-precision "
    "single signal, so treat it as the prior. Revise only when the other evaluators "
    "supply specific artefact evidence that contradicts it -- agreement among them "
    "is not itself evidence, since they are largely the same model family and their "
    "errors correlate. If you revise, set changed_from_independent and say what "
    "evidence moved you."
)


def pi_independent_messages(image_url: str) -> list[BaseMessage]:
    return [
        SystemMessage(content=PI_INDEPENDENT_SYSTEM),
        _user(image_url, "Classify the corruption in this MRI scan."),
    ]


def pi_arbitration_messages(
    image_url: str,
    independent: PrincipalDecision | None,
    opinions: list[AgentOpinion],
    radiologist: RadiologistReview | None,
) -> list[BaseMessage]:
    lines: list[str] = []

    if independent is not None:
        lines.append(
            f"Your independent judgement: {independent.classification.value} "
            f"(confidence {independent.confidence:.2f})\nReasoning: {independent.reasoning}"
        )

    for o in opinions:
        label = o.classification.value if o.classification else "no answer"
        lines.append(
            f"{o.agent}: {label} (confidence {o.confidence:.2f})\nReasoning: {o.reasoning}"
        )

    if radiologist is not None:
        lines.append(
            f"Radiologist: {radiologist.classification.value} "
            f"(confidence {radiologist.confidence:.2f})\n"
            f"Agrees with assistants: {radiologist.agrees_with_assistants}\n"
            f"Reasoning: {radiologist.reasoning}"
        )

    return [
        SystemMessage(content=PI_ARBITRATION_SYSTEM),
        _user(image_url, "Evidence on the table:\n\n" + "\n\n".join(lines)),
    ]
