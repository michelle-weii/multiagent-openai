"""
ARC-AGI Multi-Agent Example

Uses the multiagent_framework_v1 to collaboratively solve ARC-AGI visual
reasoning problems using a verifier + 5 reasoners pattern.
"""

import os
import sys
import json
from typing import List

import networkx as nx
from openai import OpenAI

# Add frameworks directory to path for framework import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frameworks"))

from multiagent_framework_v1 import (
    SimpleAgent,
    init_graph,
    add_node,
    add_edge,
    graph_summary_text,
    apply_graph_delta,
    prune_graph,
    GRAPH_DELTA_INSTRUCTIONS,
    VERIFY_GRAPH_INSTRUCTIONS,
)


# ============================================================
# ARC-AGI specific helpers
# ============================================================
def format_grid(grid: List[List[int]], indent: str = "  ") -> str:
    """Format a grid as a readable string."""
    if not grid:
        return f"{indent}(empty)"
    lines = []
    for row in grid:
        lines.append(indent + " ".join(str(cell) for cell in row))
    return "\n".join(lines)


def format_arc_problem(train_pairs: List[dict], test_input: List[List[int]]) -> str:
    """Format ARC problem as text."""
    lines = [
        "ARC-AGI Problem",
        "=" * 40,
        "",
        "Find the transformation rule from the training examples,",
        "then apply it to the test input.",
        "",
        "TRAINING EXAMPLES:",
    ]

    for i, pair in enumerate(train_pairs, 1):
        lines.append(f"\n--- Example {i} ---")
        lines.append("Input:")
        lines.append(format_grid(pair["input"]))
        lines.append("Output:")
        lines.append(format_grid(pair["output"]))

    lines.append("\n" + "=" * 40)
    lines.append("TEST INPUT (apply the transformation):")
    lines.append(format_grid(test_input))

    return "\n".join(lines)


# ============================================================
# ARC-AGI specific instructions
# ============================================================
ANSWER_INSTRUCTIONS = (
    'Respond with ONLY valid JSON:\n'
    '{"answer":[[<int>,...],...],'
    '"reasoning":<string>,'
    '"graph_refs":[<string>...]}\n'
    'The answer must be a 2D grid (list of lists of integers).\n'
    'No extra text.'
)

FINAL_ANSWER_INSTRUCTIONS = (
    'Respond with ONLY valid JSON:\n'
    '{"chosen_agent":<int 1-5>,'
    '"answer":[[<int>,...],...],'
    '"why":<string>}\n'
    'The answer must be a 2D grid (list of lists of integers).\n'
    'No extra text.'
)


# ============================================================
# Solve ONE ARC problem collaboratively
# ============================================================
def solve_arc_problem_collaborative(
    *,
    train_pairs: List[dict],
    test_input: List[List[int]],
    model: str = "gpt-5",
    out_dir: str = "arc_problem_out",
    problem_id: str = "arc_0001",
    verbose: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    client = OpenAI()
    verifier = SimpleAgent(client, model, name="verifier", verbose=verbose)
    reasoners = [SimpleAgent(client, model, name=f"reasoner_{i+1}", verbose=verbose) for i in range(5)]

    problem_text = format_arc_problem(train_pairs, test_input)

    # -------------------------
    # Phase 0: init graph
    # -------------------------
    G = init_graph(problem_id=problem_id)
    add_node(G, f"{problem_id}:problem", type="problem", label="problem")

    print("=" * 60)
    print(f"[solve] {problem_id}")
    print(problem_text[:1000] + ("..." if len(problem_text) > 1000 else ""))

    # -------------------------
    # Phase 1: Graph building
    # -------------------------
    print("[phase 1] verifier seeds graph")
    seed_prompt = (
        "Role: VERIFIER. Phase: GRAPH.\n"
        "Seed a knowledge graph for solving THIS ARC-AGI problem.\n"
        "Include: observed patterns, colors used, grid dimensions, symmetries, transformations detected.\n"
        "Make nodes precise (e.g., 'pattern: diagonal line', 'transform: flip horizontal', 'color_map: 1->2').\n\n"
        f"PROBLEM:\n{problem_text}\n\n"
        "Return JSON nodes/edges."
    )
    seed_delta = verifier.call_json(
        phase="verifier_graph_seed",
        prompt=seed_prompt,
        instructions=GRAPH_DELTA_INSTRUCTIONS,
        reasoning_effort="low",
    )
    apply_graph_delta(G, seed_delta)
    print(f"[phase 1] after seed: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    print("[phase 1] reasoners propose graph additions")
    deltas = []
    for ridx, r in enumerate(reasoners, start=1):
        r_prompt = (
            f"Role: REASONER {ridx}/5. Phase: GRAPH.\n"
            "Propose helpful nodes/edges for understanding the transformation.\n"
            "Focus on: input-output relationships, color mappings, shape transformations, positional changes.\n"
            "Be specific about what changes between input and output in each example.\n\n"
            f"PROBLEM:\n{problem_text}\n\n"
            f"CURRENT GRAPH:\n{graph_summary_text(G)}\n\n"
            "Return JSON nodes/edges."
        )
        delta = r.call_json(
            phase="reasoner_graph_build",
            prompt=r_prompt,
            instructions=GRAPH_DELTA_INSTRUCTIONS,
            reasoning_effort="low",
        )
        deltas.append({"agent_id": ridx, "delta": delta})

    print("[phase 1] verifier accepts/rejects + prunes")
    preview_lines = []
    for d in deltas:
        aid = d["agent_id"]
        dd = d["delta"]
        preview_nodes = [n.get("id") for n in dd.get("nodes", [])[:6]]
        preview_edges = [
            f"{e.get('node1')} -[{e.get('relation')}]- {e.get('node2')}"
            for e in dd.get("edges", [])[:6]
        ]
        preview_lines.append(f"Agent {aid}: nodes={preview_nodes} edges={preview_edges}")
    preview = "\n".join(preview_lines)

    verify_prompt = (
        "Role: VERIFIER. Phase: GRAPH.\n"
        "Decide which reasoner additions to accept, and prune bad/redundant graph parts.\n"
        "Prioritize correctness and usefulness for predicting the test output.\n\n"
        f"PROBLEM:\n{problem_text}\n\n"
        "CURRENT GRAPH:\n"
        f"{graph_summary_text(G)}\n\n"
        "REASONER PROPOSALS (preview):\n"
        f"{preview}\n\n"
        "Return JSON with accept_agents/reject/prune/notes."
    )
    verdict = verifier.call_json(
        phase="verifier_graph_verify",
        prompt=verify_prompt,
        instructions=VERIFY_GRAPH_INSTRUCTIONS,
        reasoning_effort="low",
    )

    accept = set(verdict.get("accept_agents", []))
    print(f"[phase 1] accepted agents: {sorted(list(accept))}")
    for rj in verdict.get("reject", []):
        print(f"[phase 1] rejected agent {rj['agent_id']}: {rj['reason']}")

    for d in deltas:
        if d["agent_id"] in accept:
            apply_graph_delta(G, d["delta"])

    prune_graph(G, verdict.get("prune", {}), verbose=verbose)
    print(f"[phase 1] final graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # Save Phase 1 graph snapshot
    phase1_graph_path = os.path.join(out_dir, f"{problem_id}_phase1.graphml")
    nx.write_graphml(G, phase1_graph_path)
    print(f"[saved] {phase1_graph_path}")

    # -------------------------
    # Phase 2: Answer proposals
    # -------------------------
    print("[phase 2] reasoners propose final answers")
    answer_proposals = []
    for ridx, r in enumerate(reasoners, start=1):
        qa_prompt = (
            f"Role: REASONER {ridx}/5. Phase: ANSWER.\n"
            "Apply the transformation rule to the test input and produce the output grid.\n"
            "Your 'answer' must be a 2D grid (list of lists of integers).\n"
            "Make sure the output dimensions and values are consistent with the pattern.\n\n"
            f"PROBLEM:\n{problem_text}\n\n"
            f"GRAPH SUMMARY:\n{graph_summary_text(G)}\n"
        )
        a = r.call_json(
            phase="reasoner_answer",
            prompt=qa_prompt,
            instructions=ANSWER_INSTRUCTIONS,
            reasoning_effort="high",
        )
        answer_proposals.append({"agent_id": ridx, "answer": a})

        if verbose:
            ans_preview = str(a.get("answer", ""))[:120]
            print(f"[phase 2] agent {ridx} -> answer={ans_preview}")

    # -------------------------
    # Phase 3: Verifier selects final
    # -------------------------
    print("[phase 3] verifier chooses final answer")
    proposals_text = "\n".join(
        f"Agent {ap['agent_id']}: answer={ap['answer'].get('answer')} reasoning={ap['answer'].get('reasoning', '')[:200]}"
        for ap in answer_proposals
    )
    final_prompt = (
        "Role: VERIFIER. Phase: ANSWER.\n"
        "Pick the best answer among the 5 proposals.\n"
        "Verify that the chosen answer correctly applies the transformation rule.\n"
        "The answer must be a 2D grid consistent with the pattern.\n\n"
        f"PROBLEM:\n{problem_text}\n\n"
        f"GRAPH SUMMARY:\n{graph_summary_text(G)}\n\n"
        "ANSWER PROPOSALS:\n"
        f"{proposals_text}\n\n"
        "Return chosen_agent, answer (as 2D grid), why."
    )
    final = verifier.call_json(
        phase="verifier_final_answer",
        prompt=final_prompt,
        instructions=FINAL_ANSWER_INSTRUCTIONS,
        reasoning_effort="high",
    )

    # -------------------------
    # Save artifacts
    # -------------------------
    final_node = f"{problem_id}:final"
    add_node(
        G,
        final_node,
        type="final_answer",
        label="final_answer",
        answer=json.dumps(final.get("answer", [])),
        why=str(final.get("why", "")),
        chosen_agent=int(final.get("chosen_agent", 0)),
    )
    add_edge(G, f"{problem_id}:problem", final_node, "solved_as")

    for ap in answer_proposals:
        aid = ap["agent_id"]
        a = ap["answer"]
        prop_node = f"{problem_id}:proposal:{aid}"
        add_node(
            G,
            prop_node,
            type="proposal",
            label=f"proposal_agent_{aid}",
            answer=json.dumps(a.get("answer", [])),
            reasoning=str(a.get("reasoning", "")),
            graph_refs=json.dumps(a.get("graph_refs", []), ensure_ascii=False),
        )
        add_edge(G, f"{problem_id}:problem", prop_node, "proposal")
        add_edge(G, prop_node, final_node, "considered")

    graph_path = os.path.join(out_dir, f"{problem_id}.graphml")
    nx.write_graphml(G, graph_path)

    trace_path = os.path.join(out_dir, f"{problem_id}_trace.json")
    with open(trace_path, "w") as tf:
        json.dump(
            {
                "problem_id": problem_id,
                "model": model,
                "verdict": verdict,
                "answer_proposals": answer_proposals,
                "final": final,
            },
            tf,
            indent=2,
            ensure_ascii=False,
        )

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

    print(f"\n[final] chosen_agent={final.get('chosen_agent')}")
    print(f"[final] answer=")
    print(format_grid(final.get("answer", []), indent="  "))
    print(f"[saved] {graph_path}")
    print(f"[saved] {trace_path}")
    print(f"[saved] {histories_path}")

    return {
        "answer": final.get("answer"),
        "graph_path": graph_path,
        "trace_path": trace_path,
        "histories_path": histories_path,
    }


if __name__ == "__main__":
    # Puzzle 3aa6fb7a: Find L-shaped patterns of 8s, mark the corner with 1
    # Online visualization: https://arcprize.org/play?task=3aa6fb7a
    train_pairs = [
        {
            "input": [[0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0], [0, 8, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
            "output": [[0, 0, 0, 0, 0, 0, 0], [0, 8, 1, 0, 0, 0, 0], [0, 8, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 1, 8, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
        },
        {
            "input": [[0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 0, 8, 0], [0, 0, 8, 0, 0, 0, 0], [0, 0, 8, 8, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0], [0, 0, 0, 8, 8, 0, 0]],
            "output": [[0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 1, 8, 0], [0, 0, 8, 1, 0, 0, 0], [0, 0, 8, 8, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 8, 0, 0], [0, 0, 0, 8, 8, 0, 0]],
        },
    ]
    test_input = [[0, 0, 0, 0, 0, 8, 8], [8, 8, 0, 0, 0, 0, 8], [8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 8, 0, 0, 0], [0, 0, 0, 8, 8, 0, 0], [0, 8, 0, 0, 0, 0, 0], [8, 8, 0, 0, 0, 0, 0]]

    result = solve_arc_problem_collaborative(
        train_pairs=train_pairs,
        test_input=test_input,
        model="gpt-5-nano",
        out_dir="arc_out",
        problem_id="arc_3aa6fb7a",
        verbose=True,
    )

    print(f"\nExpected:")
    print("  [[0, 0, 0, 0, 0, 8, 8],")
    print("   [8, 8, 0, 0, 0, 1, 8],")
    print("   [8, 1, 0, 0, 0, 0, 0],")
    print("   [0, 0, 0, 8, 1, 0, 0],")
    print("   [0, 0, 0, 8, 8, 0, 0],")
    print("   [1, 8, 0, 0, 0, 0, 0],")
    print("   [8, 8, 0, 0, 0, 0, 0]]")
