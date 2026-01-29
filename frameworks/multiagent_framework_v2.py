"""
Generic Multi-Agent Collaborative Problem Solving Framework v2
"""

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, field
from typing import Any, Dict, List, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
from datetime import datetime

import networkx as nx
from openai import OpenAI


# ============================================================
# Logging - writes to both console and file
# ============================================================
class TeeLogger:
    """Writes output to both console and a log file."""

    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_path = log_path
        self.log_file = open(log_path, "w")

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.terminal
        self.log_file.close()


def create_log_path(out_dir: str, problem_id: str) -> str:
    """Create log file path."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{problem_id}_{timestamp}.log")


# ============================================================
# Configuration
# ============================================================
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_REASONING_EFFORT = "low"
MAX_TOKENS_GRAPH_DELTA = 16000        # No effective limit
MAX_TOKENS_VERIFICATION = 16000       # No effective limit
MAX_TOKENS_ANSWER = 16000             # No effective limit
MAX_TOKENS_HINT_GEN = 8000            # No effective limit
MAX_PREVIEW_NODES = 8
MAX_PREVIEW_EDGES = 8

# Node/edge guidance (keep small to reduce redundancy)
MIN_NODES_PER_PROPOSAL = 1
MAX_NODES_PER_PROPOSAL = 8
MIN_EDGES_PER_PROPOSAL = 1
MAX_EDGES_PER_PROPOSAL = 12


# ============================================================
# Agent with history tracking
# ============================================================
@dataclass
class HistoryItem:
    phase: str
    prompt: str
    output_text: str
    round_num: int = 0


class SimpleAgent:
    """
    Agent wrapper. Each LLM call is STATELESS - no memory between calls.
    Context must be passed explicitly in prompts.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        name: str,
        verbose: bool = True,
    ):
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
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_tokens: int = MAX_TOKENS_GRAPH_DELTA,
        round_num: int = 0,
    ) -> Dict[str, Any]:
        if self.verbose:
            print(f"\n--- [{self.name}] {phase} (round {round_num}) ---")

        response = self.client.responses.create(
            model=self.model,
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            }],
            instructions=instructions,
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_tokens,
        )

        out_text = response.output_text
        self.history.append(HistoryItem(
            phase=phase,
            prompt=prompt,
            output_text=out_text,
            round_num=round_num,
        ))

        if self.verbose:
            preview = out_text.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"output: {preview}")

        # Clean up common LLM issues before parsing
        cleaned = out_text.strip()
        # Remove markdown code blocks if present
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
            # Return empty structure as fallback
            return {"nodes": [], "edges": []}


# ============================================================
# Graph utilities
# ============================================================
def _graphml_safe_value(v: Any) -> Any:
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


def graph_summary_text(G: nx.MultiGraph, max_nodes: int = 30, max_edges: int = 50) -> str:
    nodes = list(G.nodes(data=True))[:max_nodes]
    edges = list(G.edges(data=True, keys=True))[:max_edges]

    lines = ["GRAPH NODES:"]
    for nid, a in nodes:
        lines.append(f"  - {nid} :: type={a.get('type')} label={a.get('label')}")

    lines.append("\nGRAPH EDGES:")
    for u, v, k, a in edges:
        lines.append(f"  - {u} -[{a.get('relation')}]-> {v}")

    return "\n".join(lines)


def apply_graph_delta(G: nx.MultiGraph, delta: Dict[str, Any]):
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
        add_edge(G, n1, n2, e.get("relation", "related_to"),
                 **_graphml_safe_attrs(e.get("attrs") or {}))


def prune_graph(G: nx.MultiGraph, prune: Dict[str, Any], verbose: bool = True):
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

    if verbose and (removed_nodes or removed_edges):
        print(f"[prune] removed {removed_nodes} nodes, {removed_edges} edges")


def format_proposals_preview(deltas: List[Dict[str, Any]]) -> str:
    """Format proposals with FULL detail for verifier review."""
    lines = []
    for i, d in enumerate(deltas):
        lines.append(f"\n=== Agent {i+1} Proposal ===")

        # Nodes with full detail
        nodes = d.get("nodes", [])[:MAX_PREVIEW_NODES]
        if nodes:
            lines.append("NODES:")
            for n in nodes:
                node_str = f"  - id: {n.get('id')}"
                if n.get('label'):
                    node_str += f", label: {n.get('label')}"
                if n.get('type'):
                    node_str += f", type: {n.get('type')}"
                if n.get('attrs'):
                    attrs_preview = str(n.get('attrs'))[:100]
                    node_str += f", attrs: {attrs_preview}"
                lines.append(node_str)

        # Edges with full detail
        edges = d.get("edges", [])[:MAX_PREVIEW_EDGES]
        if edges:
            lines.append("EDGES:")
            for e in edges:
                edge_str = f"  - {e.get('node1')} -[{e.get('relation', '?')}]-> {e.get('node2')}"
                if e.get('attrs'):
                    edge_str += f" (attrs: {str(e.get('attrs'))[:50]})"
                lines.append(edge_str)

        if not nodes and not edges:
            lines.append("  (empty proposal)")

    return "\n".join(lines)


# ============================================================
# Simplified ProblemConfig
# ============================================================
ProblemData = TypeVar('ProblemData')
AnswerType = TypeVar('AnswerType')


class ProblemConfig(ABC, Generic[ProblemData, AnswerType]):
    """
    Simplified problem configuration - only 3 methods needed!
    """

    def __init__(self, problem_data: ProblemData, problem_type: str):
        self.problem_data = problem_data
        self.problem_type = problem_type

    @abstractmethod
    def format_problem_for_prompt(self) -> str:
        """Convert problem data to text description."""
        pass

    @abstractmethod
    def get_answer_format(self) -> str:
        """Describe expected answer format (e.g., 'A single integer')."""
        pass

    @abstractmethod
    def parse_final_answer(self, final: Dict[str, Any]) -> AnswerType:
        """Extract typed answer from JSON response."""
        pass

    def validate_answer(self, answer: AnswerType) -> bool:
        return True

    def get_problem_metadata(self) -> Dict[str, Any]:
        return {"problem_type": self.problem_type}


# ============================================================
# Hardcoded JSON schemas (optimal format)
# ============================================================
GRAPH_DELTA_INSTRUCTIONS = (
    'CRITICAL: Respond with ONLY valid JSON. No markdown code blocks, no ```json, no extra text.\n'
    'Start your response with { and end with }.\n\n'
    'Required format:\n'
    '{\n'
    '  "nodes": [\n'
    '    {"id": "example:node1", "label": "Node Label", "type": "concept", "attrs": {}},\n'
    '    {"id": "example:node2", "label": "Another Node", "type": "variable", "attrs": {}}\n'
    '  ],\n'
    '  "edges": [\n'
    '    {"node1": "example:node1", "node2": "example:node2", "relation": "relates_to", "attrs": {}}\n'
    '  ]\n'
    '}'
)

VERIFY_INSTRUCTIONS = (
    'CRITICAL: Respond with ONLY valid JSON. No markdown code blocks, no ```json, no extra text.\n'
    'Start your response with { and end with }.\n\n'
    'Required format:\n'
    '{\n'
    '  "accept_agents": [1, 2, 3],\n'
    '  "reject": [{"agent_id": 4, "reason": "explanation"}],\n'
    '  "prune": {\n'
    '    "remove_nodes": ["node_id_to_remove"],\n'
    '    "remove_edges": [{"node1": "a", "node2": "b", "relation": "rel"}]\n'
    '  },\n'
    '  "continue": true,\n'
    '  "notes": "brief notes"\n'
    '}'
)

ANSWER_INSTRUCTIONS = (
    'CRITICAL: Respond with ONLY valid JSON. No markdown code blocks, no ```json, no extra text.\n'
    'Start your response with { and end with }.\n\n'
    'Required format:\n'
    '{\n'
    '  "answer": "your answer here",\n'
    '  "reasoning": "step by step explanation",\n'
    '  "confidence": 0.85,\n'
    '  "graph_refs": ["node_id_1", "node_id_2"]\n'
    '}'
)

FINAL_ANSWER_INSTRUCTIONS = (
    'CRITICAL: Respond with ONLY valid JSON. No markdown code blocks, no ```json, no extra text.\n'
    'Start your response with { and end with }.\n\n'
    'Required format:\n'
    '{\n'
    '  "chosen_agent": 1,\n'
    '  "answer": "the final answer",\n'
    '  "why": "explanation for choosing this answer"\n'
    '}'
)


# ============================================================
# Domain hint generation (only the domain-specific part)
# ============================================================
@dataclass
class DomainHints:
    """LLM-generated domain-specific guidance."""
    graph_seed_hints: str = ""
    graph_proposal_hints: str = ""
    answer_hints: str = ""
    agent_focus_areas: List[str] = field(default_factory=list)


def generate_domain_hints(
    client: OpenAI,
    problem_str: str,
    problem_type: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> DomainHints:
    """
    Generate domain-specific hints AND dynamically decide number of agents (2-8).
    The prompt structure remains hardcoded for optimal performance.
    """
    if verbose:
        print("\n=== Generating domain-specific hints ===")

    hint_prompt = f"""You are helping design prompts for a multi-agent problem-solving system.

PROBLEM TYPE: {problem_type}

PROBLEM:
{problem_str[:1500]}

Generate domain-specific guidance for agents working on this problem.

Return a JSON object with these keys:

1. "num_agents": How many parallel agents (2-8) should work on this problem?
   - Simple problems (basic arithmetic, obvious patterns): 2-3 agents
   - Medium problems (multi-step reasoning, some complexity): 4-5 agents
   - Complex problems (advanced math, many constraints, deep reasoning): 6-8 agents

2. "graph_seed_hints": What concepts/structures should be in the initial graph? (1-2 sentences)

3. "graph_proposal_hints": What should agents focus on when proposing additions? (1-2 sentences)

4. "answer_hints": What should agents consider when proposing final answers? (1-2 sentences)

5. "agent_focus_areas": A list of EXACTLY num_agents DISTINCT focus areas.
   Each agent works on ONE area to avoid overlap. Make them complementary.
   Examples:
   - Math: ["problem setup", "key concepts", "solution strategy"]
   - Games: ["position evaluation", "strategy"]
   - Visual: ["pattern analysis", "transformation rules"]"""

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": hint_prompt}]}],
        instructions="Return ONLY valid JSON with the five keys. No extra text.",
        reasoning={"effort": "low"},
        max_output_tokens=MAX_TOKENS_HINT_GEN,
    )

    # Defaults
    default_num = 4
    default_focus = ["problem setup", "key relationships", "solution steps", "verification"]

    try:
        data = json.loads(response.output_text)
        num_agents = int(data.get("num_agents", default_num))
        num_agents = max(2, min(8, num_agents))  # Clamp to 2-8

        focus_areas = data.get("agent_focus_areas", default_focus)
        # Ensure we have exactly num_agents focus areas
        if len(focus_areas) < num_agents:
            focus_areas.extend([f"aspect {i+1}" for i in range(len(focus_areas), num_agents)])
        focus_areas = focus_areas[:num_agents]

        hints = DomainHints(
            graph_seed_hints=data.get("graph_seed_hints", "key concepts, variables, constraints, and relationships"),
            graph_proposal_hints=data.get("graph_proposal_hints", "nontrivial steps, pitfalls, and alternate approaches"),
            answer_hints=data.get("answer_hints", "verify your answer and ensure correct format"),
            agent_focus_areas=focus_areas,
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        if verbose:
            print("[warning] Failed to parse hints, using defaults")
        hints = DomainHints(
            graph_seed_hints="key concepts, variables, constraints, and relationships",
            graph_proposal_hints="nontrivial steps, pitfalls, and alternate approaches",
            answer_hints="verify your answer and ensure correct format",
            agent_focus_areas=default_focus,
        )

    if verbose:
        print(f"[hints] num_agents: {len(hints.agent_focus_areas)}")
        print(f"[hints] seed: {hints.graph_seed_hints[:80]}...")
        print(f"[hints] proposal: {hints.graph_proposal_hints[:80]}...")
        print(f"[hints] answer: {hints.answer_hints[:80]}...")
        print(f"[hints] focus areas:")
        for i, area in enumerate(hints.agent_focus_areas):
            print(f"  agent {i+1}: {area}")

    return hints


# ============================================================
# Hardcoded prompt templates (with placeholders for hints)
# ============================================================
def build_graph_seed_prompt(problem_str: str, hints: DomainHints) -> str:
    return f"""Role: VERIFIER. Phase: GRAPH SEED.

Create an initial knowledge graph to help solve this problem.
This graph will be the foundation that other agents build upon.

PROBLEM:
{problem_str}

Include: {hints.graph_seed_hints}

REQUIREMENTS:
- Create 8-15 nodes covering the key aspects of the problem
- Create 8-15 edges connecting related concepts
- Use clear, descriptive node IDs (e.g., "variable:E", "constraint:positive", "given:circumradius_1")
- Be SPECIFIC - include actual values, formulas, concrete details
- Cover: problem setup, given values, key variables, relevant formulas, solution approach

Return ONLY valid JSON with nodes and edges. No markdown, no extra text."""


def build_graph_proposal_prompt(
    problem_str: str,
    graph_summary: str,
    agent_id: int,
    total_agents: int,
    hints: DomainHints,
) -> str:
    focus = hints.agent_focus_areas[(agent_id - 1) % len(hints.agent_focus_areas)]

    return f"""Role: REASONER {agent_id}. Focus: {focus}

PROBLEM:
{problem_str}

EXISTING GRAPH (do NOT duplicate these nodes):
{graph_summary}

TASK: Add NOVEL nodes and CONNECT them to existing nodes.

NODES (1-5 new nodes):
- Check each node isn't already in the graph (even with different wording)
- Good: specific calculations, concrete values, proof steps, new insights
- Bad: restatements, rewordings, vague concepts already covered

EDGES (important - connect the graph!):
- Draw edges FROM your new nodes TO existing nodes above
- Draw edges FROM existing nodes TO your new nodes
- Use meaningful relations: "implies", "requires", "computed_from", "equals", "contradicts"
- More edges = better connected graph

If nothing new to add, return {{"nodes": [], "edges": []}}.

Return JSON with nodes and edges."""


def build_verify_prompt(
    problem_str: str,
    graph_summary: str,
    proposals_preview: str,
    hints: DomainHints,
    round_num: int,
    max_rounds: int,
) -> str:
    return f"""Role: VERIFIER. Round {round_num}/{max_rounds}.

PROBLEM:
{problem_str}

EXISTING GRAPH NODES (check EACH proposal against these):
{graph_summary}

PROPOSALS TO REVIEW:
{proposals_preview}

YOUR TASK: For EACH proposed node, check if it duplicates an existing node.

DUPLICATE DETECTION (be strict):
- Same concept, different words = DUPLICATE (reject)
- Same formula, different variable names = DUPLICATE (reject)
- Same insight rephrased = DUPLICATE (reject)
- Examples: "boundary condition" = "condition for boundary" = "criterion for boundary"

For each agent, list which specific nodes to REJECT and why.

WHEN TO CONTINUE (continue: true) - BE AGGRESSIVE:
- Any agent proposed something novel -> keep going
- Problem is complex (multi-step, many variables) -> keep going
- If unsure, default to CONTINUE (continue: true)

ONLY STOP when graph is truly saturated with no new ideas possible.

Return JSON:
{{
  "accept_agents": [agent numbers with ANY novel content],
  "reject": [{{"agent_id": N, "node_id": "...", "reason": "duplicates existing node X"}}],
  "prune": {{"remove_nodes": [...], "remove_edges": [...]}},
  "continue": false,
  "notes": "..."
}}"""


def build_answer_proposal_prompt(
    problem_str: str,
    graph_summary: str,
    agent_id: int,
    total_agents: int,
    answer_format: str,
    hints: DomainHints,
) -> str:
    return f"""Role: REASONER {agent_id}/{total_agents}. Phase: ANSWER.

You are part of a multi-agent system. The knowledge graph has been built.
Your task is to propose a final answer to the problem.

PROBLEM:
{problem_str}

KNOWLEDGE GRAPH:
{graph_summary}

EXPECTED ANSWER FORMAT: {answer_format}

TASK:
Solve the problem and propose your answer.

Guidelines:
- {hints.answer_hints}
- Reference graph nodes that support your reasoning (graph_refs)
- Provide clear reasoning for your answer
- Rate your confidence (0-1)

Return JSON with answer, reasoning, confidence, and graph_refs."""


def build_final_select_prompt(
    problem_str: str,
    graph_summary: str,
    proposals_text: str,
    answer_format: str,
) -> str:
    return f"""Role: VERIFIER. Phase: FINAL ANSWER SELECTION.

You are the verifier in a multi-agent problem-solving system.
Multiple agents have proposed answers. Select the best one.

PROBLEM:
{problem_str}

KNOWLEDGE GRAPH:
{graph_summary}

EXPECTED ANSWER FORMAT: {answer_format}

AGENT PROPOSALS:
{proposals_text}

TASK:
1. Review all proposals
2. Select the best answer (chosen_agent)
3. Return the final answer in the expected format
4. Explain your choice

Consider:
- Correctness of reasoning
- Consistency with the knowledge graph
- Confidence levels
- Agreement between agents

Return JSON with chosen_agent, answer, and why."""


# ============================================================
# Parallel execution
# ============================================================
def run_parallel_calls(
    agents: List[SimpleAgent],
    prompts: List[str],
    instructions: str,
    phase: str,
    round_num: int,
    max_tokens: int = MAX_TOKENS_GRAPH_DELTA,
) -> List[Dict[str, Any]]:
    """Run multiple LLM calls in parallel."""
    results = [None] * len(agents)

    def call_agent(idx: int):
        try:
            result = agents[idx].call_json(
                phase=phase,
                prompt=prompts[idx],
                instructions=instructions,
                round_num=round_num,
                max_tokens=max_tokens,
            )
            results[idx] = result
        except Exception as e:
            print(f"[error] Agent {idx} failed: {e}")
            results[idx] = {"nodes": [], "edges": []}

    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        executor.map(call_agent, range(len(agents)))

    return results


# ============================================================
# Main solver
# ============================================================
def solve_problem_collaborative(
    *,
    problem_config: ProblemConfig,
    problem_id: str,
    model: str = DEFAULT_MODEL,
    out_dir: str = "output",
    num_agents: Optional[int] = None,  # None = auto-determine (2-8 based on complexity)
    max_rounds: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Solve a problem using multi-agent collaboration with parallel execution.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Setup logging to file
    log_path = create_log_path(out_dir, problem_id)
    logger = TeeLogger(log_path)
    sys.stdout = logger

    client = OpenAI()
    problem_str = problem_config.format_problem_for_prompt()
    answer_format = problem_config.get_answer_format()

    print("=" * 60)
    print(f"[solve] {problem_id} (type: {problem_config.problem_type})")
    print(f"[log] Output being saved to: {log_path}")
    print(problem_str[:500] + ("..." if len(problem_str) > 500 else ""))

    # =========================================
    # Step 0: Generate domain-specific hints (includes dynamic agent count)
    # =========================================
    hints = generate_domain_hints(
        client=client,
        problem_str=problem_str,
        problem_type=problem_config.problem_type,
        model=model,
        verbose=verbose,
    )

    # Use provided num_agents or auto-determined from hints
    if num_agents is None:
        num_agents = len(hints.agent_focus_areas)
    else:
        # If user specified, adjust focus areas to match
        if len(hints.agent_focus_areas) < num_agents:
            hints.agent_focus_areas.extend([f"aspect {i+1}" for i in range(len(hints.agent_focus_areas), num_agents)])
        hints.agent_focus_areas = hints.agent_focus_areas[:num_agents]

    print(f"[config] Using {num_agents} parallel agents")

    # =========================================
    # Print all prompt templates BEFORE rounds
    # =========================================
    print("\n" + "=" * 60)
    print("PROMPT TEMPLATES (with generated hints)")
    print("=" * 60)

    sample_graph = "GRAPH NODES:\n  - problem:root :: type=problem label=root\n\nGRAPH EDGES:\n  (none yet)"

    print("\n--- GRAPH SEED PROMPT (verifier) ---")
    print(build_graph_seed_prompt(problem_str[:300] + "...", hints))

    print("\n--- GRAPH PROPOSAL PROMPT ---")
    print(build_graph_proposal_prompt(
        problem_str[:300] + "...",
        sample_graph,
        agent_id=1,
        total_agents=num_agents,
        hints=hints,
    ))

    print("\n--- VERIFY PROMPT ---")
    print(build_verify_prompt(
        problem_str[:300] + "...",
        sample_graph,
        "Agent 1: [sample proposals]\nAgent 2: [sample proposals]",
        hints,
        round_num=1,
        max_rounds=max_rounds,
    ))

    print("\n--- ANSWER PROPOSAL PROMPT ---")
    print(build_answer_proposal_prompt(
        problem_str[:300] + "...",
        sample_graph,
        agent_id=1,
        total_agents=num_agents,
        answer_format=answer_format,
        hints=hints,
    ))

    print("\n--- FINAL SELECT PROMPT ---")
    print(build_final_select_prompt(
        problem_str[:300] + "...",
        sample_graph,
        "Agent 1: {...}\nAgent 2: {...}",
        answer_format,
    ))

    print("\n--- JSON SCHEMAS ---")
    print("GRAPH_DELTA_INSTRUCTIONS:")
    print(GRAPH_DELTA_INSTRUCTIONS)
    print("\nVERIFY_INSTRUCTIONS:")
    print(VERIFY_INSTRUCTIONS)
    print("\nANSWER_INSTRUCTIONS:")
    print(ANSWER_INSTRUCTIONS)
    print("\nFINAL_ANSWER_INSTRUCTIONS:")
    print(FINAL_ANSWER_INSTRUCTIONS)

    print("\n" + "=" * 60)
    print("STARTING ROUNDS")
    print("=" * 60)

    # =========================================
    # Setup agents
    # =========================================
    verifier = SimpleAgent(client, model, name="verifier", verbose=verbose)
    reasoners = [
        SimpleAgent(client, model, name=f"reasoner_{i+1}", verbose=verbose)
        for i in range(num_agents)
    ]

    # =========================================
    # Initialize graph
    # =========================================
    G = init_graph(problem_id=problem_id)
    add_node(G, f"{problem_id}:problem", type="problem", label="problem")

    all_verdicts = []
    all_deltas = []

    # =========================================
    # Verifier creates seed graph
    # =========================================
    print(f"\n{'='*40}")
    print("SEED GRAPH (verifier)")
    print(f"{'='*40}")

    seed_prompt = build_graph_seed_prompt(problem_str, hints)
    seed_delta = verifier.call_json(
        phase="seed",
        prompt=seed_prompt,
        instructions=GRAPH_DELTA_INSTRUCTIONS,
        round_num=0,
        max_tokens=MAX_TOKENS_GRAPH_DELTA,
    )
    apply_graph_delta(G, seed_delta)
    all_deltas.append({"agent_id": "verifier", "delta": seed_delta, "phase": "seed"})
    print(f"[seed] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # =========================================
    # Graph Building Rounds (agents propose, verifier reviews)
    # =========================================
    round_num = 0
    continue_building = True

    while continue_building and round_num < max_rounds:
        round_num += 1
        print(f"\n{'='*40}")
        print(f"ROUND {round_num}")
        print(f"{'='*40}")

        # Build prompts - each agent gets their focus area
        graph_text = graph_summary_text(G)
        proposal_prompts = [
            build_graph_proposal_prompt(
                problem_str, graph_text, i+1, num_agents, hints
            )
            for i in range(num_agents)
        ]

        # Parallel proposal calls
        print(f"[round {round_num}] {num_agents} agents proposing in parallel...")
        deltas = run_parallel_calls(
            agents=reasoners,
            prompts=proposal_prompts,
            instructions=GRAPH_DELTA_INSTRUCTIONS,
            phase=f"graph_r{round_num}",
            round_num=round_num,
            max_tokens=MAX_TOKENS_GRAPH_DELTA,
        )

        # Print node/edge counts for each agent
        print(f"\n[round {round_num}] Proposal summary:")
        for i, d in enumerate(deltas):
            n_nodes = len(d.get("nodes", []))
            n_edges = len(d.get("edges", []))
            print(f"  Agent {i+1}: {n_nodes} nodes, {n_edges} edges")

        round_deltas = [{"agent_id": i+1, "delta": d} for i, d in enumerate(deltas)]
        all_deltas.extend(round_deltas)

        # Build detailed preview for verifier
        proposals_preview = format_proposals_preview(deltas)

        # 1 verification call
        verify_prompt = build_verify_prompt(
            problem_str,
            graph_summary_text(G),
            proposals_preview,
            hints,
            round_num,
            max_rounds,
        )

        verdict = verifier.call_json(
            phase=f"verify_r{round_num}",
            prompt=verify_prompt,
            instructions=VERIFY_INSTRUCTIONS,
            round_num=round_num,
            max_tokens=MAX_TOKENS_VERIFICATION,
        )

        all_verdicts.append({"round": round_num, "verdict": verdict})

        # Apply accepted proposals
        accept = set(verdict.get("accept_agents", []))
        print(f"[round {round_num}] accepted: {sorted(accept)}")
        for rj in verdict.get("reject", []):
            print(f"[round {round_num}] rejected agent {rj.get('agent_id')}: {rj.get('reason', '')[:80]}")

        for i, d in enumerate(deltas):
            if (i + 1) in accept:
                apply_graph_delta(G, d)

        # Prune
        prune_graph(G, verdict.get("prune", {}), verbose=verbose)

        print(f"[round {round_num}] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Check continue
        continue_building = verdict.get("continue", False)
        print(f"[round {round_num}] continue={continue_building}")

    # =========================================
    # Answer Proposals (4 parallel)
    # =========================================
    print(f"\n{'='*40}")
    print("ANSWER PHASE")
    print(f"{'='*40}")

    graph_text = graph_summary_text(G)
    answer_prompts = [
        build_answer_proposal_prompt(
            problem_str, graph_text, i+1, num_agents, answer_format, hints
        )
        for i in range(num_agents)
    ]

    print(f"[answer] {num_agents} agents proposing answers in parallel...")
    answer_proposals = run_parallel_calls(
        agents=reasoners,
        prompts=answer_prompts,
        instructions=ANSWER_INSTRUCTIONS,
        phase="answer",
        round_num=0,
        max_tokens=MAX_TOKENS_ANSWER,
    )

    for i, a in enumerate(answer_proposals):
        ans_preview = str(a.get("answer", ""))[:80]
        conf = a.get("confidence", "?")
        print(f"[answer] agent {i+1}: {ans_preview} (confidence: {conf})")

    # =========================================
    # Final Answer Selection (1 call)
    # =========================================
    proposals_text = "\n\n".join(
        f"Agent {i+1}:\n{json.dumps(a, indent=2)}"
        for i, a in enumerate(answer_proposals)
    )

    select_prompt = build_final_select_prompt(
        problem_str, graph_text, proposals_text, answer_format
    )

    final = verifier.call_json(
        phase="final_select",
        prompt=select_prompt,
        instructions=FINAL_ANSWER_INSTRUCTIONS,
        round_num=0,
        max_tokens=MAX_TOKENS_ANSWER,
    )

    # =========================================
    # Parse answer
    # =========================================
    parsed_answer = problem_config.parse_final_answer(final)
    is_valid = problem_config.validate_answer(parsed_answer)

    print(f"\n{'='*40}")
    print(f"FINAL ANSWER: {parsed_answer}")
    print(f"Valid: {is_valid}")
    print(f"Chosen agent: {final.get('chosen_agent')}")
    print(f"Why: {final.get('why', '')[:200]}")
    print(f"{'='*40}")

    # =========================================
    # Save artifacts
    # =========================================
    graph_path = os.path.join(out_dir, f"{problem_id}.graphml")
    nx.write_graphml(G, graph_path)

    trace = {
        "problem_id": problem_id,
        "problem_type": problem_config.problem_type,
        "model": model,
        "num_rounds": round_num,
        "hints": {
            "graph_seed": hints.graph_seed_hints,
            "graph_proposal": hints.graph_proposal_hints,
            "answer": hints.answer_hints,
        },
        "verdicts": all_verdicts,
        "all_deltas": all_deltas,
        "answer_proposals": [
            {"agent_id": i+1, "answer": a}
            for i, a in enumerate(answer_proposals)
        ],
        "final": final,
        "parsed_answer": str(parsed_answer),
    }
    trace_path = os.path.join(out_dir, f"{problem_id}_trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    histories = {
        "verifier": [
            {"phase": h.phase, "round": h.round_num, "prompt": h.prompt, "output": h.output_text}
            for h in verifier.history
        ],
        "reasoners": {
            f"reasoner_{i+1}": [
                {"phase": h.phase, "round": h.round_num, "prompt": h.prompt, "output": h.output_text}
                for h in r.history
            ]
            for i, r in enumerate(reasoners)
        },
    }
    histories_path = os.path.join(out_dir, f"{problem_id}_histories.json")
    with open(histories_path, "w") as f:
        json.dump(histories, f, indent=2, ensure_ascii=False)

    print(f"\n[saved] {graph_path}")
    print(f"[saved] {trace_path}")
    print(f"[saved] {histories_path}")
    print(f"[saved] {log_path}")

    # Restore stdout and close log file
    sys.stdout = logger.terminal
    logger.log_file.close()

    return {
        "answer": parsed_answer,
        "raw_final": final,
        "is_valid": is_valid,
        "num_rounds": round_num,
        "graph_path": graph_path,
        "trace_path": trace_path,
        "histories_path": histories_path,
        "log_path": log_path,
        "metadata": problem_config.get_problem_metadata(),
    }
