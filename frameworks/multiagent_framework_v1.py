"""
Multi-Agent Collaborative Problem Solving Framework (v1)

Shared components for multi-agent systems using a verifier + reasoners pattern
with a collaborative knowledge graph.
"""

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
from openai import OpenAI


# ============================================================
# Agent wrapper with separate history
# ============================================================
@dataclass
class HistoryItem:
    phase: str
    prompt: str
    output_text: str


class SimpleAgent:
    def __init__(self, client: OpenAI, model: str, name: str, verbose: bool = True):
        self.client = client
        self.model = model
        self.name = name
        self.verbose = verbose
        self.history: List[HistoryItem] = []

    def call_json(
        self,
        *,
        phase: str,
        prompt: str,
        instructions: str,
        reasoning_effort: str = "low",
    ) -> Dict[str, Any]:
        if self.verbose:
            print(f"\n--- [{self.name}] {phase} ---")

        response = self.client.responses.create(
            model=self.model,
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            }],
            instructions=instructions,
            reasoning={"effort": reasoning_effort},
        )

        out_text = response.output_text
        self.history.append(HistoryItem(phase=phase, prompt=prompt, output_text=out_text))

        if self.verbose:
            preview = out_text.strip().replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            print(f"output: {preview}")

        # Clean up markdown if present
        cleaned = out_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[error] JSON parse failed: {e}")
            print(f"[error] Raw output: {out_text[:500]}")
            return {"nodes": [], "edges": []}


# ============================================================
# GraphML-safe attribute sanitizer
# ============================================================
def _graphml_safe_value(v: Any) -> Any:
    """
    GraphML supports scalar attribute values (str/int/float/bool).
    Convert list/dict/etc to JSON strings.
    """
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _graphml_safe_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _graphml_safe_value(v) for k, v in (attrs or {}).items()}


# ============================================================
# Knowledge graph using undirected MultiGraph
# ============================================================
def init_graph(problem_id: str) -> nx.MultiGraph:
    return nx.MultiGraph(problem_id=problem_id)


def add_node(G: nx.MultiGraph, node_id: str, **attrs):
    attrs = _graphml_safe_attrs(attrs)
    if node_id not in G:
        G.add_node(node_id, **attrs)
    else:
        for k, v in attrs.items():
            if k not in G.nodes[node_id]:
                G.nodes[node_id][k] = v


def add_edge(G: nx.MultiGraph, node1: str, node2: str, relation: str, **attrs):
    attrs = _graphml_safe_attrs(attrs)
    G.add_edge(node1, node2, relation=str(relation), **attrs)


def graph_summary_text(G: nx.MultiGraph, max_nodes: int = 40, max_edges: int = 80) -> str:
    nodes = list(G.nodes(data=True))[:max_nodes]
    edges = list(G.edges(data=True, keys=True))[:max_edges]

    lines = []
    lines.append("GRAPH NODES (subset):")
    for nid, a in nodes:
        lines.append(f"- {nid} :: type={a.get('type')} label={a.get('label')}")

    lines.append("\nGRAPH EDGES (subset):")
    for u, v, k, a in edges:
        lines.append(f"- {u} -[{a.get('relation')}]- {v}")

    return "\n".join(lines)


def apply_graph_delta(G: nx.MultiGraph, delta: Dict[str, Any]):
    """
    Expected JSON:
    {
      "nodes": [{"id": "...", "label":"...", "type":"...", "attrs": {...}}, ...],
      "edges": [{"node1":"...", "node2":"...", "relation":"...", "attrs": {...}}, ...]
    }
    """
    for n in delta.get("nodes", []):
        add_node(
            G,
            n["id"],
            label=n.get("label", n["id"]),
            type=n.get("type", "unknown"),
            **_graphml_safe_attrs(n.get("attrs") or {})
        )

    for e in delta.get("edges", []):
        n1 = e["node1"]
        n2 = e["node2"]
        if n1 not in G:
            add_node(G, n1, label=n1, type="unknown")
        if n2 not in G:
            add_node(G, n2, label=n2, type="unknown")

        add_edge(
            G,
            n1,
            n2,
            e.get("relation", "related_to"),
            **_graphml_safe_attrs(e.get("attrs") or {})
        )


def prune_graph(G: nx.MultiGraph, prune: Dict[str, Any], verbose: bool = True):
    """
    prune JSON:
    {
      "remove_nodes": ["..."],
      "remove_edges": [{"node1":"...","node2":"...","relation":"..."}]
    }
    """
    removed_nodes = 0
    removed_edges = 0

    for nid in prune.get("remove_nodes", []):
        if nid in G:
            G.remove_node(nid)
            removed_nodes += 1

    for e in prune.get("remove_edges", []):
        n1, n2, rel = e.get("node1"), e.get("node2"), e.get("relation")
        if n1 in G and n2 in G:
            to_remove = []
            data = G.get_edge_data(n1, n2, default={})
            for key, attrs in data.items():
                if rel is None or attrs.get("relation") == rel:
                    to_remove.append(key)
            for key in to_remove:
                G.remove_edge(n1, n2, key=key)
                removed_edges += 1

    if verbose:
        print(f"[prune] removed_nodes={removed_nodes}, removed_edges={removed_edges}")


# ============================================================
# Shared Instructions (ONLY JSON)
# ============================================================
GRAPH_DELTA_INSTRUCTIONS = (
    'Respond with ONLY valid JSON:\n'
    '{"nodes":[{"id":<string>,"label":<string>,"type":<string>,"attrs":<object>}...],'
    '"edges":[{"node1":<string>,"node2":<string>,"relation":<string>,"attrs":<object>}...]}\n'
    'No extra text.'
)

VERIFY_GRAPH_INSTRUCTIONS = (
    'Respond with ONLY valid JSON:\n'
    '{"accept_agents":[<int 1-5>...],'
    '"reject":[{"agent_id":<int 1-5>,"reason":<string>}...],'
    '"prune":{"remove_nodes":[<string>...],'
    '"remove_edges":[{"node1":<string>,"node2":<string>,"relation":<string>}...]},'
    '"notes":<string>}\n'
    'No extra text.'
)


# ============================================================
# Utility for saving artifacts
# ============================================================
def save_artifacts(
    *,
    G: nx.MultiGraph,
    out_dir: str,
    problem_id: str,
    verifier: SimpleAgent,
    reasoners: List[SimpleAgent],
    verdict: Dict[str, Any],
    answer_proposals: List[Dict[str, Any]],
    final: Dict[str, Any],
    extra_trace_fields: Dict[str, Any] = None,
    model: str = "",
):
    """Save graph, trace, and agent histories."""
    graph_path = os.path.join(out_dir, f"{problem_id}.graphml")
    nx.write_graphml(G, graph_path)

    trace_data = {
        "problem_id": problem_id,
        "model": model,
        "verdict": verdict,
        "answer_proposals": answer_proposals,
        "final": final,
    }
    if extra_trace_fields:
        trace_data.update(extra_trace_fields)

    trace_path = os.path.join(out_dir, f"{problem_id}_trace.json")
    with open(trace_path, "w") as tf:
        json.dump(trace_data, tf, indent=2, ensure_ascii=False)

    histories = {
        "verifier": [{"phase": h.phase, "prompt": h.prompt, "output_text": h.output_text} for h in verifier.history],
        "reasoners": {
            f"reasoner_{i+1}": [{"phase": h.phase, "prompt": h.prompt, "output_text": h.output_text} for h in r.history]
            for i, r in enumerate(reasoners)
        },
    }
    histories_path = os.path.join(out_dir, f"{problem_id}_agent_histories.json")
    with open(histories_path, "w") as hf:
        json.dump(histories, hf, indent=2, ensure_ascii=False)

    print(f"[saved] {graph_path}")
    print(f"[saved] {trace_path}")
    print(f"[saved] {histories_path}")

    return {
        "graph_path": graph_path,
        "trace_path": trace_path,
        "histories_path": histories_path,
    }
