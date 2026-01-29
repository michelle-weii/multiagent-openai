"""
Tic-Tac-Toe Solver using the Simplified v2 Framework

Only 3 methods needed in ProblemConfig!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frameworks"))

from typing import Dict, Any, List, Tuple
from multiagent_framework_v2 import ProblemConfig, solve_problem_collaborative


def format_board(board: List[List[str]]) -> str:
    """Format board as visual string."""
    lines = []
    for i, row in enumerate(board):
        display_row = [" " if cell in (".", "") else cell for cell in row]
        lines.append(f" {display_row[0]} | {display_row[1]} | {display_row[2]} ")
        if i < 2:
            lines.append("-----------")
    return "\n".join(lines)


def get_legal_moves(board: List[List[str]]) -> List[Tuple[int, int]]:
    """Get list of empty squares."""
    moves = []
    for r in range(3):
        for c in range(3):
            if board[r][c] in (".", "", " "):
                moves.append((r, c))
    return moves


class TicTacToeConfig(ProblemConfig[dict, Tuple[int, int]]):
    """
    Tic-Tac-Toe problem configuration.

    Problem data format:
    {
        "board": [["X", ".", "O"], [".", "X", "."], [".", ".", "."]],
        "player": "X"
    }
    """

    def __init__(self, board: List[List[str]], player: str):
        problem_data = {
            "board": board,
            "player": player,
        }
        super().__init__(problem_data=problem_data, problem_type="game_tictactoe")

    def format_problem_for_prompt(self) -> str:
        board = self.problem_data["board"]
        player = self.problem_data["player"]
        legal = get_legal_moves(board)

        return f"""Tic-Tac-Toe Game
================

Current player: {player}

Board state:
{format_board(board)}

Board coordinates:
 (0,0) | (0,1) | (0,2)
-------+-------+------
 (1,0) | (1,1) | (1,2)
-------+-------+------
 (2,0) | (2,1) | (2,2)

Legal moves (empty squares): {legal}

Task: Find the OPTIMAL move for player {player}.
Consider: winning moves, blocking opponent wins, creating forks, center control."""

    def get_answer_format(self) -> str:
        return "Row (0-2) and column (0-2) as integers, e.g., row=1, col=2"

    def parse_final_answer(self, final: Dict[str, Any]) -> Tuple[int, int]:
        row = final.get("row", -1)
        col = final.get("col", -1)

        # Handle if answer is nested
        if isinstance(final.get("answer"), dict):
            row = final["answer"].get("row", row)
            col = final["answer"].get("col", col)

        return (int(row), int(col))

    def validate_answer(self, answer: Tuple[int, int]) -> bool:
        """Check move is within bounds and on empty square."""
        row, col = answer
        if not (0 <= row <= 2 and 0 <= col <= 2):
            return False
        board = self.problem_data["board"]
        return board[row][col] in (".", "", " ")


# Convenience function
def solve_tictactoe(
    board: List[List[str]],
    player: str,
    problem_id: str = "ttt_001",
    **kwargs,
):
    """Find optimal move for a tic-tac-toe position."""
    config = TicTacToeConfig(board, player)
    return solve_problem_collaborative(
        problem_config=config,
        problem_id=problem_id,
        **kwargs,
    )


if __name__ == "__main__":
    # Example 1: X can win immediately
    print("\n" + "="*60)
    print("EXAMPLE 1: X can win on diagonal")
    print("="*60)

    board1 = [
        ["X", "O", "."],
        [".", "X", "O"],
        [".", ".", "."]
    ]

    result1 = solve_tictactoe(
        board=board1,
        player="X",
        problem_id="ttt_win",
        model="gpt-5-nano",
        out_dir="output/tictactoe",
        # num_agents auto-determined (2-8), max_rounds auto (verifier decides)
        verbose=True,
    )

    print(f"\nChosen move: row={result1['answer'][0]}, col={result1['answer'][1]}")
    print(f"Expected: (2, 2) - completes diagonal")
    print(f"Valid: {result1['is_valid']}")

    # # Example 2: X must block O
    # print("\n" + "="*60)
    # print("EXAMPLE 2: X must block O's win")
    # print("="*60)
    #
    # board2 = [
    #     ["O", ".", "O"],
    #     [".", "X", "."],
    #     ["X", ".", "."]
    # ]
    #
    # result2 = solve_tictactoe(
    #     board=board2,
    #     player="X",
    #     problem_id="ttt_block",
    #     model="gpt-5-nano",
    #     out_dir="output/tictactoe",
    # )
    #
    # print(f"\nChosen move: row={result2['answer'][0]}, col={result2['answer'][1]}")
    # print(f"Expected: (0, 1) - blocks O's top row")
    # print(f"Valid: {result2['is_valid']}")

    # # Example 3: Opening move
    # print("\n" + "="*60)
    # print("EXAMPLE 3: Opening move")
    # print("="*60)
    #
    # board3 = [
    #     [".", ".", "."],
    #     [".", ".", "."],
    #     [".", ".", "."]
    # ]
    #
    # result3 = solve_tictactoe(
    #     board=board3,
    #     player="X",
    #     problem_id="ttt_opening",
    #     model="gpt-5-nano",
    #     out_dir="output/tictactoe",
    # )
    #
    # print(f"\nChosen move: row={result3['answer'][0]}, col={result3['answer'][1]}")
    # print(f"Optimal openings: (1,1) center, or corners")
    # print(f"Valid: {result3['is_valid']}")
