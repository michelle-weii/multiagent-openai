"""
Math Multi-Agent Example

Uses the multiagent_framework_v1 to collaboratively solve math problems
using a verifier + 5 reasoners pattern.
"""

import os
import sys
import json

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
# Math-specific instructions
# ============================================================
ANSWER_INSTRUCTIONS = (
    'Respond with ONLY valid JSON:\n'
    '{"answer":<string>,'
    '"reasoning":<string>,'
    '"graph_refs":[<string>...]}\n'
    'No extra text.'
)

FINAL_ANSWER_INSTRUCTIONS = (
    'Respond with ONLY valid JSON:\n'
    '{"chosen_agent":<int 1-5>,'
    '"answer":<string>,'
    '"why":<string>}\n'
    'No extra text.'
)


# ============================================================
# Solve ONE math problem collaboratively
# ============================================================
def solve_math_problem_collaborative(
    *,
    problem_text: str,
    model: str = "gpt-5",
    out_dir: str = "math_problem_out",
    problem_id: str = "problem_0001",
    verbose: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    client = OpenAI()
    verifier = SimpleAgent(client, model, name="verifier", verbose=verbose)
    reasoners = [SimpleAgent(client, model, name=f"reasoner_{i+1}", verbose=verbose) for i in range(5)]

    # -------------------------
    # Phase 0: init graph
    # -------------------------
    G = init_graph(problem_id=problem_id)
    add_node(G, f"{problem_id}:problem", type="problem", label="problem", text=problem_text)

    print("=" * 60)
    print(f"[solve] {problem_id}")
    print(problem_text[:800] + ("..." if len(problem_text) > 800 else ""))

    # -------------------------
    # Phase 1: Graph building
    # -------------------------
    print("[phase 1] verifier seeds graph")
    seed_prompt = (
        "Role: VERIFIER. Phase: GRAPH.\n"
        "Seed a small knowledge graph for solving THIS math problem.\n"
        "Include: definitions, key variables, subgoals, known lemmas/theorems, and a plan outline.\n"
        "Make nodes precise (e.g., 'E=expected perimeter', 'event: diagonal drawn', 'subgoal: compute P(center region)').\n\n"
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
            "Propose helpful nodes/edges to add for solving.\n"
            "Focus on nontrivial steps, pitfalls, and any alternate approaches.\n"
            "Prefer explicit formulas / invariants.\n\n"
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
        "Prioritize correctness and usefulness for reaching the final numeric answer.\n\n"
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

    # -------------------------
    # Save Phase 1 graph snapshot (before Phase 2 proposals)
    # -------------------------
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
            "Solve the problem and propose the final answer.\n"
            "IMPORTANT: If the problem asks for a specific formatted output (e.g., floor, integer), your 'answer'\n"
            "must be exactly that output (e.g., the integer as a string).\n"
            "Keep 'reasoning' concise but sufficient to justify the numeric result.\n"
            "In 'graph_refs', list ids of graph nodes you relied on.\n\n"
            f"PROBLEM:\n{problem_text}\n\n"
            f"GRAPH SUMMARY:\n{graph_summary_text(G)}\n"
        )
        a = r.call_json(
            phase="reasoner_answer",
            prompt=qa_prompt,
            instructions=ANSWER_INSTRUCTIONS,
            reasoning_effort="low",
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
        f"Agent {ap['agent_id']}: answer={ap['answer'].get('answer')} reasoning={ap['answer'].get('reasoning')[:200]}"
        for ap in answer_proposals
    )
    final_prompt = (
        "Role: VERIFIER. Phase: ANSWER.\n"
        "Pick the best answer among the 5 proposals.\n"
        "If proposals disagree, identify the most reliable derivation and choose its final output.\n"
        "Return the final answer EXACTLY in the required format (often a single integer).\n\n"
        f"PROBLEM:\n{problem_text}\n\n"
        f"GRAPH SUMMARY:\n{graph_summary_text(G)}\n\n"
        "ANSWER PROPOSALS:\n"
        f"{proposals_text}\n\n"
        "Return chosen_agent, answer, why."
    )
    final = verifier.call_json(
        phase="verifier_final_answer",
        prompt=final_prompt,
        instructions=FINAL_ANSWER_INSTRUCTIONS,
        reasoning_effort="low",
    )

    # -------------------------
    # Graph bookkeeping + save artifacts
    # -------------------------
    final_node = f"{problem_id}:final"
    add_node(
        G,
        final_node,
        type="final_answer",
        label="final_answer",
        answer=str(final.get("answer", "")),
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
            answer=str(a.get("answer", "")),
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

    print(f"[final] chosen_agent={final.get('chosen_agent')} answer={final.get('answer')}")
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
    problem_text = "Let $\mathcal{P}$ be a regular $101$-gon of circumradius $1$. Draw each diagonal of $\mathcal{P}$ with probability $0.001$. This splits $\mathcal{P}$ into several closed regions. Let $E$ be the expected value of the perimeter of the region containing the center of $\mathcal{P}$. Compute $\left\lfloor 10^9 E \right\rfloor$."

    solve_math_problem_collaborative(
        problem_text=problem_text,
        model="gpt-5",
        out_dir="math_out",
        problem_id="problem_0003",
        verbose=True,
    )
