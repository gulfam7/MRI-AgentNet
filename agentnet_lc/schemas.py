"""Typed contracts for every agent in the pipeline.

These Pydantic models replace the hand-written extraction layer in
``utils/data_processing_confidence.py`` (``parse_gpt4o_response``,
``parse_evaluator_response``, ``extract_reasoning`` -- roughly 250 lines of
regex, spaCy token walking and defaulting).

The provider is handed the JSON schema generated from these classes and is
constrained to emit conforming output, so a malformed response becomes a
validation error we can retry on rather than a silently mis-parsed field.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class CorruptionType(str, Enum):
    """The label space. Free-text synonyms are normalised into this by `coerce`."""

    MOTION = "motion corrupted"
    UNDERSAMPLED = "undersampled"
    NOISY = "noisy"
    NONE = "no corruption"

    @classmethod
    def coerce(cls, value: str) -> "CorruptionType":
        """Map the synonyms the old `map_classification` dict handled."""
        text = (value or "").strip().lower()
        synonyms = {
            "motion artifact": cls.MOTION,
            "motion": cls.MOTION,
            "k-space artifact": cls.UNDERSAMPLED,
            "k-space artifact or undersampling": cls.UNDERSAMPLED,
            "undersampling": cls.UNDERSAMPLED,
            "aliasing": cls.UNDERSAMPLED,
            "noise": cls.NOISY,
            "noisy or noise corrupted": cls.NOISY,
            "radiofrequency noise": cls.NOISY,
            "gradient instability": cls.NOISY,
            "mri noise": cls.NOISY,
            "none": cls.NONE,
            "corruption free": cls.NONE,
        }
        for member in cls:
            if text == member.value:
                return member
        if text in synonyms:
            return synonyms[text]
        for key, member in synonyms.items():
            if key in text:
                return member
        raise ValueError(f"unrecognised corruption type: {value!r}")


class RestorationModel(str, Enum):
    MOTION_CORRECTION = "motion_correction"
    DENOISING = "denoising"
    RECONSTRUCTION = "reconstruction"
    NONE = "no_correction"


class ImageSpace(str, Enum):
    IMAGE = "image space"
    KSPACE = "k-space"


class SpaceClassification(BaseModel):
    """Output of the pre-classification step."""

    space: ImageSpace
    reasoning: str = Field(description="One sentence on what indicates this domain.")


class CorruptionAssessment(BaseModel):
    """The five-section contract, as a type instead of a prompt instruction.

    The old prompt said: "You should never omit any of the 5 points below ...
    Do not use newline characters between numbered sections". None of that is
    needed now -- omission is a schema violation the provider cannot emit.
    """

    classification: CorruptionType
    reasoning: str = Field(
        min_length=20,
        description="Artefact-specific evidence: ghosting pattern, SNR, k-space signature.",
    )
    recommended_model: RestorationModel
    correction_plan: str = Field(min_length=10)
    confidence: float = Field(
        ge=0.0, le=1.0, description="Honest probability that the classification is correct."
    )

    @field_validator("classification", mode="before")
    @classmethod
    def _coerce_label(cls, value):
        if isinstance(value, CorruptionType):
            return value
        return CorruptionType.coerce(str(value))


class RadiologistReview(BaseModel):
    """Few-shot expert reviewer: independent call, then a verdict on the assistants."""

    classification: CorruptionType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    agrees_with_assistants: bool
    recommended_model: RestorationModel

    @field_validator("classification", mode="before")
    @classmethod
    def _coerce_label(cls, value):
        if isinstance(value, CorruptionType):
            return value
        return CorruptionType.coerce(str(value))


class PrincipalDecision(BaseModel):
    """Arbitration output. `changed_from_independent` makes the old
    'Are you changing your Decision?' free-text field machine-checkable."""

    classification: CorruptionType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    recommended_model: RestorationModel
    changed_from_independent: bool = False
    change_justification: str | None = None

    @field_validator("classification", mode="before")
    @classmethod
    def _coerce_label(cls, value):
        if isinstance(value, CorruptionType):
            return value
        return CorruptionType.coerce(str(value))


class AgentOpinion(BaseModel):
    """One evaluator's contribution, carried through the graph state."""

    agent: str
    classification: CorruptionType | None = None
    confidence: float = 0.5
    reasoning: str = ""
    ok: bool = True
    error: str | None = None

    @classmethod
    def failed(cls, agent: str, error: str) -> "AgentOpinion":
        return cls(agent=agent, ok=False, error=error, confidence=0.0)


class RouteDecision(BaseModel):
    """Final routing choice, with the evidence that produced it."""

    model: RestorationModel
    source: str = Field(description="'meta_model' or the rule-based fallback reason.")
    probabilities: dict[str, float] = Field(default_factory=dict)
