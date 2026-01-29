"""
Tic-Tac-Toe Multi-Agent Example

Uses the multiagent_framework_v1 to collaboratively determine the best move
in a tic-tac-toe position using a verifier + 5 reasoners pattern.
"""

import os
import sys
import json
from typing import List, Tuple

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
# Tic-Tac-Toe specific helpers
# ============================================================
def get_board_string(state):
    """Convert state array to string representation for display"""
    result = "\n"
    for i, row in enumerate(state):
        display_row = [" " if cell == "." else cell for cell in row]
        result += f" {display_row[0]} | {display_row[1]} | {display_row[2]} \n"
        if i < 2:
            result += "-----------\n"
    return result


def legal_moves(state) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(3) for c in range(3) if state[r][c] == "."]


# ============================================================
# Tic-Tac-Toe specific instructions
# ============================================================
MOVE_INSTRUCTIONS = (
    'Respond with ONLY valid JSON:\n'
    '{"row":<int 0-2>,"col":<int 0-2>,"why":<string>,"graph_refs":[<string>...]}\n'
    'No extra text.'
)

FINAL_MOVE_INSTRUCTIONS = (
    'Respond with ONLY valid JSON:\n'
    '{"chosen_agent":<int 1-5>,"row":<int 0-2>,"col":<int 0-2>,"why":<string>}\n'
    'No extra text.'
)


# ============================================================
# Run only sample 1 (example 1)
# ============================================================
def run_example_1_only(
    dataset_path: str,
    model: str = "gpt-5",
    graphs_dir: str = "ttt_graphs_example1",
    verbose: bool = True,
):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if not dataset:
        raise ValueError("Dataset is empty.")

    os.makedirs(graphs_dir, exist_ok=True)

    example = dataset[0]  # ONLY EXAMPLE 1
    state = example["state"]
    current_player = example["current_player"]
    optimal_moves = example["optimal_moves"]
    board_str = get_board_string(state)

    print("=" * 60)
    print("[example 1] player=", current_player)
    print(board_str)

    client = OpenAI()
    verifier = SimpleAgent(client, model, name="verifier", verbose=verbose)
    reasoners = [SimpleAgent(client, model, name=f"reasoner_{i+1}", verbose=verbose) for i in range(5)]

    # -------------------------
    # Phase 1: Graph building
    # -------------------------
    print("[phase 1] init graph")
    G = init_graph(problem_id="sample_1")

    print("[phase 1] verifier seeds graph")
    seed_prompt = (
        "Role: VERIFIER. Phase: GRAPH.\n"
        "Seed a small knowledge graph for this Tic-Tac-Toe position.\n"
        "Include key concepts + what matters in THIS board (threats, blocks, forks).\n\n"
        f"Player to move: {current_player}\n"
        f"Board:{board_str}\n\n"
        "Return JSON nodes/edges. Use edge fields node1/node2."
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
            "Propose helpful nodes/edges to add.\n"
            "Focus on threats, blocks, forks, and important squares.\n\n"
            f"Player to move: {current_player}\n"
            f"Board:{board_str}\n\n"
            f"{graph_summary_text(G, max_nodes=30, max_edges=60)}\n\n"
            "Return JSON nodes/edges. Use edge fields node1/node2."
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
        preview_nodes = [n.get("id") for n in dd.get("nodes", [])[:5]]
        preview_edges = [
            f"{e.get('node1')} -[{e.get('relation')}]- {e.get('node2')}"
            for e in dd.get("edges", [])[:5]
        ]
        preview_lines.append(f"Agent {aid}: nodes={preview_nodes} edges={preview_edges}")
    preview = "\n".join(preview_lines)

    verify_prompt = (
        "Role: VERIFIER. Phase: GRAPH.\n"
        "Decide which reasoner additions to accept, and prune bad/redundant graph parts.\n\n"
        f"Player to move: {current_player}\n"
        f"Board:{board_str}\n\n"
        "Current graph:\n"
        f"{graph_summary_text(G, max_nodes=30, max_edges=60)}\n\n"
        "Reasoner proposals (preview):\n"
        f"{preview}\n\n"
        "Return JSON with accept_agents/reject/prune."
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
    # Phase 2: Question answering
    # -------------------------
    print("[phase 2] reasoners propose moves")
    move_proposals = []
    for ridx, r in enumerate(reasoners, start=1):
        qa_prompt = (
            f"Role: REASONER {ridx}/5. Phase: ANSWER.\n"
            "Propose the best next move for the current player.\n"
            "Use the graph summary; list graph node ids you used in graph_refs.\n\n"
            f"Player to move: {current_player}\n"
            f"Board:{board_str}\n\n"
            f"{graph_summary_text(G, max_nodes=30, max_edges=60)}\n"
        )
        m = r.call_json(
            phase="reasoner_answer",
            prompt=qa_prompt,
            instructions=MOVE_INSTRUCTIONS,
            reasoning_effort="low",
        )
        move_proposals.append({"agent_id": ridx, "move": m})
        print(f"[phase 2] agent {ridx} -> ({m['row']},{m['col']})")

    print("[phase 2] verifier chooses final move")
    moves_text = "\n".join(
        f"Agent {mp['agent_id']}: ({mp['move']['row']},{mp['move']['col']}) why={mp['move']['why']}"
        for mp in move_proposals
    )
    final_prompt = (
        "Role: VERIFIER. Phase: ANSWER.\n"
        "Choose the best move among the 5 proposals. Must be legal.\n\n"
        f"Player to move: {current_player}\n"
        f"Board:{board_str}\n\n"
        "Graph:\n"
        f"{graph_summary_text(G, max_nodes=30, max_edges=60)}\n\n"
        "Move proposals:\n"
        f"{moves_text}\n\n"
        "Return chosen_agent,row,col,why."
    )
    final = verifier.call_json(
        phase="verifier_final_answer",
        prompt=final_prompt,
        instructions=FINAL_MOVE_INSTRUCTIONS,
        reasoning_effort="low",
    )

    model_move = {"row": int(final["row"]), "col": int(final["col"])}
    if (model_move["row"], model_move["col"]) not in set(legal_moves(state)):
        raise ValueError(f"Illegal final move from verifier: {model_move}")

    is_correct = model_move in optimal_moves
    result = "correct" if is_correct else "incorrect"
    print(f"[result] {result} final=({model_move['row']},{model_move['col']}) chose_agent={final['chosen_agent']}")
    print(f"[result] why: {final['why'][:300]}")
    print(f"[result] optimal_moves={optimal_moves}")

    # -------------------------
    # Save graph + trace + histories (ONLY example 1)
    # -------------------------
    board_node = "sample:1:board"
    add_node(G, board_node, type="board", label="board", player=current_player, state=json.dumps(state, ensure_ascii=False))

    final_node = "sample:1:final"
    add_node(G, final_node, type="final_move", label="final_move",
             row=model_move["row"], col=model_move["col"], why=final["why"])
    add_edge(G, board_node, final_node, "final_move")

    for mp in move_proposals:
        aid = mp["agent_id"]
        mv = mp["move"]
        prop_node = f"sample:1:proposal:{aid}"
        add_node(G, prop_node, type="proposal", label=f"proposal_agent_{aid}",
                 row=mv["row"], col=mv["col"], why=mv["why"])
        add_edge(G, board_node, prop_node, "proposal")
        add_edge(G, prop_node, final_node, "considered")

    graph_path = os.path.join(graphs_dir, "sample_0001.graphml")
    nx.write_graphml(G, graph_path)

    trace_path = os.path.join(graphs_dir, "sample_0001_trace.json")
    with open(trace_path, "w") as tf:
        json.dump({
            "sample_idx": 1,
            "current_player": current_player,
            "verdict": verdict,
            "move_proposals": move_proposals,
            "final": final
        }, tf, indent=2, ensure_ascii=False)

    histories = {
        "verifier": [{"phase": h.phase, "prompt": h.prompt, "output_text": h.output_text} for h in verifier.history],
        "reasoners": {
            f"reasoner_{i+1}": [{"phase": h.phase, "prompt": h.prompt, "output_text": h.output_text} for h in r.history]
            for i, r in enumerate(reasoners)
        }
    }
    histories_path = os.path.join(graphs_dir, "agent_histories.json")
    with open(histories_path, "w") as hf:
        json.dump(histories, hf, indent=2, ensure_ascii=False)

    print(f"[saved] {graph_path}")
    print(f"[saved] {trace_path}")
    print(f"[saved] {histories_path}")

    return {
        "correct": is_correct,
        "model_move": model_move,
        "optimal_moves": optimal_moves,
        "graph_path": graph_path,
        "trace_path": trace_path,
        "histories_path": histories_path,
    }


if __name__ == "__main__":
    dataset_path = "../data/tictactoe_failed_examples.json"
    run_example_1_only(
        dataset_path=dataset_path,
        model="gpt-5",
        graphs_dir="ttt_example1_out",
        verbose=True
    )
