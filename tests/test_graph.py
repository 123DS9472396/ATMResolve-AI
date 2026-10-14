from __future__ import annotations
import pytest

def test_graph_compiles_with_required_nodes():
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_ollama")
    from langgraph.checkpoint.memory import InMemorySaver
    from src.agents.graph import _compile_graph
    graph = _compile_graph(InMemorySaver())
    nodes = set(graph.get_graph().nodes)
    assert {"validate_input", "classify_intent", "retrieve_policy", "lookup_evidence", "apply_rules", "draft_recommendation", "human_review", "finalize_case"}.issubset(nodes)
