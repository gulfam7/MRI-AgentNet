"""The pipeline as an explicit LangGraph state machine.

    START -> classify_space -> [k-space?] -> convert_kspace -.
                    |                                        |
                    '---------------- assess <---------------'
                                     /      \\
                        assistant_gpt4o    assistant_gemini      (concurrent)
                                     \\      /
                                    radiologist                  (joins both)
                                         |
                                   pi_independent
                                         |
                                   pi_arbitration
                                         |
                                       route -> END

Why this is worth the dependency, when the original was a straight-line method:

* The two assistants genuinely run at the same time. The old ``for model_name,
  model_interface in models:`` loop made them sequential, so wall-clock was the
  sum of two API calls rather than the max.
* ``models`` is injected, so the whole graph runs against stubs in tests.
* With a checkpointer, a crash in arbitration resumes without re-paying for the
  four calls before it.
* Every node boundary is a LangSmith span for free.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agentnet_lc import nodes, router
from agentnet_lc.llms import ModelBundle
from agentnet_lc.state import PipelineState


def build_graph(
    models: ModelBundle,
    *,
    checkpointer=None,
    router_checkpoint: Path | None = None,
):
    """Compile the pipeline. Pass a checkpointer to make runs resumable."""
    builder = StateGraph(PipelineState)

    builder.add_node("classify_space", partial(nodes.classify_space, models=models))
    builder.add_node("convert_kspace", partial(nodes.convert_kspace, models=models))
    builder.add_node("assess", lambda state: {})  # fan-out join point
    builder.add_node("assistant_gpt4o", partial(nodes.assistant_primary, models=models))
    builder.add_node("assistant_gemini", partial(nodes.assistant_secondary, models=models))
    builder.add_node("radiologist", partial(nodes.radiologist, models=models))
    builder.add_node("pi_independent", partial(nodes.pi_independent, models=models))
    builder.add_node("pi_arbitration", partial(nodes.pi_arbitration, models=models))
    builder.add_node("route", partial(router.route, checkpoint=router_checkpoint))

    builder.add_edge(START, "classify_space")
    builder.add_conditional_edges(
        "classify_space",
        nodes.needs_kspace_conversion,
        {"convert": "convert_kspace", "assess": "assess"},
    )
    builder.add_edge("convert_kspace", "assess")

    # Two edges out of one node = one concurrent superstep.
    builder.add_edge("assess", "assistant_gpt4o")
    builder.add_edge("assess", "assistant_gemini")

    # Two edges into one node = a barrier; radiologist waits for both.
    builder.add_edge("assistant_gpt4o", "radiologist")
    builder.add_edge("assistant_gemini", "radiologist")

    builder.add_edge("radiologist", "pi_independent")
    builder.add_edge("pi_independent", "pi_arbitration")
    builder.add_edge("pi_arbitration", "route")
    builder.add_edge("route", END)

    return builder.compile(checkpointer=checkpointer)


def draw_ascii(graph) -> str:
    """Render the compiled topology, handy for a README or a slide.

    Falls back to Mermaid when `grandalf` (the ASCII layout engine) is absent,
    since Mermaid is what you would paste into the paper or GitHub anyway.
    """
    drawable = graph.get_graph()
    try:
        return drawable.draw_ascii()
    except ImportError:
        return drawable.draw_mermaid()
