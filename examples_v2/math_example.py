"""
Math Problem Solver using the Simplified v2 Framework

Only 3 methods needed in ProblemConfig!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frameworks"))

from typing import Dict, Any
from multiagent_framework_v2 import ProblemConfig, solve_problem_collaborative


class MathProblemConfig(ProblemConfig[str, str]):
    """
    Math problem configuration.

    Only 3 methods needed:
    - format_problem_for_prompt(): Return the problem text
    - get_answer_format(): Describe expected answer format
    - parse_final_answer(): Extract answer from JSON
    """

    def __init__(self, problem_text: str):
        super().__init__(problem_data=problem_text, problem_type="math")

    def format_problem_for_prompt(self) -> str:
        return self.problem_data

    def get_answer_format(self) -> str:
        return "A single integer or number as a string"

    def parse_final_answer(self, final: Dict[str, Any]) -> str:
        return str(final.get("answer", ""))


# Convenience function
def solve_math_problem(
    problem_text: str,
    problem_id: str = "math_001",
    **kwargs,
):
    """Solve a math problem using the multiagent framework."""
    config = MathProblemConfig(problem_text)
    return solve_problem_collaborative(
        problem_config=config,
        problem_id=problem_id,
        **kwargs,
    )


if __name__ == "__main__":
    problem="What is the sum of all prime numbers less than 30? Return only the integer."
    #problem = "A sequence y_1, ..., y_k is zigzag if differences alternate in sign. Let X_1,...,X_n ~ Uniform[0,1]. Let a(X_1,...,X_n) be the longest zigzag subsequence. Find E[a(X_1,...,X_n)] for n ≥ 2."
    #problem = "Let $\mathcal{P}$ be a regular $101$-gon of circumradius $1$. Draw each diagonal of $\mathcal{P}$ with probability $0.001$. This splits $\mathcal{P}$ into several closed regions. Let $E$ be the expected value of the perimeter of the region containing the center of $\mathcal{P}$. Compute $\left\lfloor 10^9 E \right\rfloor$."

    result = solve_math_problem(
        problem_text=problem,
        problem_id="math_primes",
        model="gpt-5-nano",
        out_dir="output/math",
        verbose=True,
    )

    print(f"\nFinal Answer: {result['answer']}")
    print(f"Valid: {result['is_valid']}")
    print(f"Rounds: {result['num_rounds']}")