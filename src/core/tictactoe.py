import numpy as np
from enum import IntEnum
from numpy.typing import NDArray


class Cell(IntEnum):
    """
    Enum representing the states of a cell in the TicTacToe game.

    @enum Cell
    @value B: Represents a blank cell (0).
    @value X: Represents player X's move (1).
    @value O: Represents player O's move (2).
    """

    B = 0  # Blank
    X = 1  # Player X's move
    O = 2  # Player O's move


class TicTacToe:
    """
    A class representing a TicTacToe game with a 3x3 game board.

    The board is represented as a 3x3 NumPy array where each cell is one of the
    values from the `Cell` enum (Blank, X, O).

    Attributes:
        board (NDArray[np.int64]): The 3x3 game board, initialized with blank cells (Cell.B).
    """

    def __init__(self):
        """
        Initializes a new TicTacToe game with an empty board.

        The board is a 3x3 NumPy array with all cells initialized to Cell.B (Blank).
        """
        self.board: NDArray[np.int64] = np.full((3, 3), Cell.B.value)

    def display_board(self) -> None:
        """
        Displays the current state of the TicTacToe board in a human-readable format.
        """
        for i, row in enumerate(self.board):
            print(" | ".join(str(Cell(cell).name) for cell in row))
            if i < len(self.board) - 1:
                print("-" * 9)

    def make_move(self, row: int, col: int, player: Cell) -> None:
        """
        Makes a move for the given player at the specified row and column.
        Raises an error if the move is invalid (e.g., out of bounds or on an already filled cell).
        """
        if row < 0 or row > 2 or col < 0 or col > 2:
            raise ValueError("Row and column must be between 0 and 2")

        if self.board[row, col] != Cell.B.value:
            raise ValueError(f"Cell ({row}, {col}) is already occupied.")

        self.board[row, col] = player.value

    def check_winner(self):
        """
        Checks if there's a winner (either player X or O).

        Returns:
            `Cell.X` if player X wins, `Cell.O` if player O wins, or `None` if no winner.
        """
        # Check rows for a win
        for row in self.board:
            if row[0] == row[1] == row[2] != Cell.B.value:
                return Cell(row[0])

        # Check columns for a win
        for col in range(3):
            if (
                self.board[0, col]
                == self.board[1, col]
                == self.board[2, col]
                != Cell.B.value
            ):
                return Cell(self.board[0, col])

        # Check diagonals for a win
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != Cell.B.value:
            return Cell(self.board[0, 0])

        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != Cell.B.value:
            return Cell(self.board[0, 2])

        return None  # No winner


class SuperTicTacToe:
    def __init__(self):
        pass
