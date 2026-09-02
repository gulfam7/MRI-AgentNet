"""Model wiring: structured output, retries, cross-provider fallback.

This replaces ``utils/gpt4o_interface.py`` (three copy-pasted methods, each with
a four-level ``try/except`` cascade for digging the text out of the response
object) and ``utils/gemini_interface_confidence.py``.

Everything here returns a ``Runnable``, so the same object composes with
``.with_retry()``, ``.with_fallbacks()``, ``.batch()`` and LangSmith tracing
regardless of which provider is underneath.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable

from agentnet_lc.schemas import (
    CorruptionAssessment,
    PrincipalDecision,
    RadiologistReview,
    SpaceClassification,
)

# Temperature 0 everywhere: every call is a classification with a fixed schema,
# so sampling diversity buys nothing and costs reproducibility. The original
# code never set temperature and therefore ran at the provider default of 1.0.
DETERMINISTIC = 0.0

DEFAULT_GPT4O = "gpt-4o-2024-11-20"
DEFAULT_GEMINI = "gemini-2.0-flash"
DEFAULT_PI = os.getenv(
    "MRI_PI_MODEL", "ft:gpt-4o-2024-08-06:personal:combined-900:Av7pB23x"
)


@dataclass(frozen=True)
class ModelBundle:
    """The four roles, already bound to their output schemas.

    Held as a value object so `build_graph` can be handed stubs in tests
    without any network access or API keys.
    """

    space_classifier: Runnable
    assistant_primary: Runnable
    assistant_secondary: Runnable
    radiologist: Runnable
    principal: Runnable


def _openai(model: str, **kwargs: Any):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=DETERMINISTIC,
        timeout=90,
        max_retries=0,  # retry policy is applied by the Runnable wrapper below
        **kwargs,
    )


def _gemini(model: str, **kwargs: Any):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=DETERMINISTIC,
        timeout=90,
        max_retries=0,
        **kwargs,
    )


def _structured(chat_model, schema, *, attempts: int = 3) -> Runnable:
    """Bind a response schema and an exponential-backoff retry policy.

    ``with_structured_output`` makes the provider emit JSON conforming to the
    Pydantic model and parses it for us. A schema violation raises, and the
    retry wrapper re-asks -- which is the behaviour the old code approximated
    with "If a confidence score is missing, do not omit it" in the prompt.
    """
    return chat_model.with_structured_output(schema).with_retry(
        stop_after_attempt=attempts,
        wait_exponential_jitter=True,
    )


def build_models(
    *,
    gpt4o_model: str = DEFAULT_GPT4O,
    gemini_model: str = DEFAULT_GEMINI,
    principal_model: str = DEFAULT_PI,
    cross_provider_fallback: bool = True,
) -> ModelBundle:
    """Construct the production bundle from environment credentials.

    Requires OPENAI_API_KEY and GOOGLE_API_KEY. Raises at call time, not import
    time, so the module stays importable in tests.
    """
    gpt4o = _openai(gpt4o_model)
    gemini = _gemini(gemini_model)
    principal = _openai(principal_model)

    primary = _structured(gpt4o, CorruptionAssessment)
    secondary = _structured(gemini, CorruptionAssessment)

    if cross_provider_fallback:
        # If OpenAI is rate-limited or down, the same call transparently runs
        # against Gemini. The old pipeline just logged a warning and continued
        # with a shorter list, which then crashed on classification_results[1].
        primary = primary.with_fallbacks([secondary])
        secondary = secondary.with_fallbacks([_structured(gpt4o, CorruptionAssessment)])

    return ModelBundle(
        space_classifier=_structured(gpt4o, SpaceClassification),
        assistant_primary=primary,
        assistant_secondary=secondary,
        radiologist=_structured(gpt4o, RadiologistReview),
        principal=_structured(principal, PrincipalDecision),
    )
