# Multi-agent Collaborative Reasoning through a Graph Memory Architecture

A framework for solving problems using multiple reasoning agents that collaborate through a shared knowledge graph.

## Requirements

- **uv** - Python package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **OpenAI API Key** - Required for LLM calls

### Setup

1. Install dependencies with uv:
```bash
uv sync
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add it to a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Project Structure

```
├── frameworks/
│   ├── multiagent_framework_v1.py   # Simple 5-reasoner + verifier framework
│   └── multiagent_framework_v2.py   # Parallel + multi-round framework
├── examples_v1/                      # Examples using framework v1
│   ├── arc_example.py
│   ├── math_example.py
│   └── tictactoe_example.py
├── examples_v2/                      # Examples using framework v2
│   ├── arc_agi_example.py
│   ├── math_example.py
│   └── tictactoe_example.py
├── data/                             # Sample data and test cases
│   └── tictactoe_failed_examples.json
├── visualize.py                      # Graph visualization tool
└── README.md
```

## Framework Versions

### Framework v1 (Simple)

Sequential execution with explicit control over each phase. Examples define their own prompts and solving logic while importing shared components:

- `SimpleAgent` - LLM wrapper with history tracking
- Graph utilities (`init_graph`, `add_node`, `add_edge`, `apply_graph_delta`, `prune_graph`)
- Shared JSON instructions for graph operations

**Best for:** Learning the architecture, custom workflows, full control over prompts.

### Framework v2 (Parallel + Multi-Round)

Automated parallel execution with dynamic agent count and iterative graph building:

- **Parallel LLM calls** using ThreadPoolExecutor
- **Multiple graph-building rounds** with verifier-controlled continuation
- **Auto-generated domain hints** to guide agents
- **Simplified ProblemConfig** - only 3 methods required

**Best for:** Complex problems, faster execution.

## Architecture

Both frameworks use a **verifier + reasoners** pattern:

### Phase 1: Knowledge Graph Building
- **Verifier** seeds an initial knowledge graph
- **Reasoners** (2-8 agents) propose additions in parallel (v2) or sequentially (v1)
- **Verifier** accepts/rejects proposals and prunes redundant information

### Phase 2: Answer Proposals
- Each **Reasoner** independently proposes a solution using the graph
- Solutions are problem-specific (numbers, grids, moves, etc.)

### Phase 3: Final Answer Selection
- **Verifier** evaluates all proposals and selects the best answer

## Running Examples

### Framework v1 Examples

```bash
# Math problem
uv run python examples_v1/math_example.py

# ARC-AGI visual reasoning
uv run python examples_v1/arc_example.py

# Tic-tac-toe
uv run python examples_v1/tictactoe_example.py
```

### Framework v2 Examples

```bash
# Math problem
uv run python examples_v2/math_example.py

# ARC-AGI visual reasoning
uv run python examples_v2/arc_agi_example.py

# Tic-tac-toe
uv run python examples_v2/tictactoe_example.py
```

## Visualizing Knowledge Graphs

The framework outputs `.graphml` files that can be visualized:

```bash
# Display graph interactively
uv run python visualize.py output/math/math_primes.graphml

# Save to PNG
uv run python visualize.py output/math/math_primes.graphml --output graph.png

# Different layout algorithms
uv run python visualize.py output/math/math_primes.graphml --layout kamada_kawai
uv run python visualize.py output/math/math_primes.graphml --layout circular

# Show edge labels
uv run python visualize.py output/math/math_primes.graphml --edge-labels

# Adjust spacing for dense graphs
uv run python visualize.py output/math/math_primes.graphml -k 8.0 --iterations 1000
```

### Visualization Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Save to file (PNG/PDF/SVG) |
| `-l, --layout` | Layout: `spring`, `circular`, `shell`, `kamada_kawai`, `spectral` |
| `--edge-labels` | Show relationship labels on edges |
| `--figsize W H` | Figure dimensions (default: 20 15) |
| `--node-size` | Node size (default: 40) |
| `--font-size` | Label font size (default: 5) |
| `-k` | Spring layout spacing (default: 5.0, higher = more spread) |
| `--iterations` | Spring layout iterations (default: 800) |

## Output Artifacts

Each run produces:

| File | Description |
|------|-------------|
| `{problem_id}.graphml` | Final knowledge graph |
| `{problem_id}_trace.json` | Verdicts, proposals, final answer |
| `{problem_id}_agent_histories.json` | Complete conversation logs |
| `{problem_id}_phase1.graphml` | Graph after Phase 1 (optional) |
| `{problem_id}_{timestamp}.log` | Console output (v2 only) |

## Creating Custom Problem Types

### Using Framework v1

Import shared components and write your solving logic:

```python
import sys
import os
# Add frameworks directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frameworks"))

from multiagent_framework_v1 import (
    SimpleAgent, init_graph, add_node, add_edge,
    graph_summary_text, apply_graph_delta, prune_graph,
    GRAPH_DELTA_INSTRUCTIONS, VERIFY_GRAPH_INSTRUCTIONS,
)

# Define domain-specific helpers and prompts
# Implement your solving function
```

See `examples_v1/` for complete examples.

### Using Framework v2

Subclass `ProblemConfig` with only 3 required methods:

```python
import sys
import os
# Add frameworks directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frameworks"))

from multiagent_framework_v2 import ProblemConfig, solve_problem_collaborative

class MyProblemConfig(ProblemConfig[InputType, OutputType]):
    def format_problem_for_prompt(self) -> str:
        """Convert problem to text for LLM."""
        return str(self.problem_data)

    def get_answer_format(self) -> str:
        """Describe expected answer format."""
        return "A single integer"

    def parse_final_answer(self, final: dict) -> OutputType:
        """Extract typed answer from JSON response."""
        return final.get("answer")

# Run solver
config = MyProblemConfig(problem_data, "my_problem_type")
result = solve_problem_collaborative(
    problem_config=config,
    problem_id="problem_001",
    model="gpt-5",
    out_dir="output",
)
```

See `examples_v2/` for complete examples.
