from typing import cast
from enum import IntEnum

import numpy as np
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
        board (NDArray[np.int8]): The 3x3 game board, initialized with blank cells (Cell.B).
    """

    def __init__(self):
        """
        Initializes a new TicTacToe game with an empty board.

        The board is a 3x3 NumPy array with all cells initialized to Cell.B (Blank).
        """
        self.board: NDArray[np.int8] = np.full((3, 3), Cell.B.value)

    def display_board(self):
        """
        Displays the current state of the TicTacToe board in a human-readable format.
        """
        for i, row in enumerate(self.board):
            print(
                " | ".join(
                    "🀫" if Cell(cell).name == "B" else Cell(cell).name for cell in row
                )
            )
            if i < len(self.board) - 1:
                print("-" * 9)

    def make_move(self, index: tuple[int, int], player: Cell):
        """
        Makes a move for the given player at the specified row and column.
        Raises an error if the move is invalid (e.g., out of bounds or on an already filled cell).

        @param index A tuple of the row and column
        """
        row, column = index
        if 2 < row < 0 or 2 < column < 0:
            raise ValueError("Row and column must be between 1 and 3.")

        if self.board[index] != Cell.B.value:
            raise ValueError(f"Cell {(index[0]+1, index[1]+1)} is already occupied.")

        self.board[index[0], index[1]] = player.value

    def check_winner(self) -> Cell | None:
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

        # Check if there is a draw
        if not (
            Cell.B.value in self.board[0]
            or Cell.B.value in self.board[1]
            or Cell.B.value in self.board[2]
        ):
            return Cell.B

        return None  # No winner yet


class SuperTicTacToe:
    def __init__(self):
        """
        Initialize a 3x3 grid of TicTacToe boards and prepare game state.

        @var boards A 3x3 NumPy array of TicTacToe instances representing the game board.
        @var next_board A tuple indicating the required board for the next move, or None if free choice.
        """
        self.boards: NDArray[np.int8] = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                self.boards[i, j] = TicTacToe()
        self.next_board: tuple[int, int] | None = None
        self.X_flat_board: NDArray[np.int8] = np.zeros(180, dtype=np.int8)
        self.O_flat_board: NDArray[np.int8] = np.zeros(180, dtype=np.int8)

    def make_move(
        self,
        outer_index: tuple[int, int],
        inner_index: tuple[int, int],
        player: Cell,
    ):
        """
        Execute a move for the specified player on a given inner board. Assumes the player knows what they're doing, i.e. what board they want next if the next board that's supposed to be played isn't playable

        @param outer_index A tuple (row, col) indicating which sub-board to play in.
        @param inner_index A tuple (row, col) within the sub-board to place the move.
        @param player The player making the move (Cell.X or Cell.O).

        @throws ValueError If the move is made on an invalid or full board.
        """
        outer_row, outer_col = outer_index
        inner_row, inner_col = inner_index
        # If the next board is predetermined, automatically directs there
        if self.next_board is None:
            if 2 < outer_row < 0 or 2 < outer_col < 0:
                raise ValueError("Row and column must be between 1 and 3.")
            cur_board_index = outer_index
        else:
            cur_board_index = self.next_board

        board: TicTacToe = cast(
            TicTacToe, self.boards[cur_board_index]
        )  # Cast for type safety

        # If board is full or already won, raises error
        if board.check_winner() or not np.any(board.board == Cell.B.value):
            raise ValueError("Board input is not valid.")

        board.make_move(inner_index, player)
        # Changes the flattened board according to the move
        flat_index = 2 * np.ravel_multi_index(
            (outer_row * 3 + inner_row, outer_col * 3 + inner_col), dims=(9, 9)
        )
        self.X_flat_board[flat_index : flat_index + 2] = (
            [1, 0] if player == Cell.X else [0, 1]
        )
        self.O_flat_board[flat_index : flat_index + 2] = (
            [1, 0] if player == Cell.O else [0, 1]
        )

        # Check for winner
        winner = board.check_winner()
        if winner is not None:
            flat_super_index = 162 + 2 * np.ravel_multi_index(
                (outer_row, outer_col), dims=(3, 3)
            )
            self.X_flat_board[flat_super_index : flat_super_index + 2] = (
                [1, 0] if player == Cell.X else [0, 1]
            )
            self.O_flat_board[flat_super_index : flat_super_index + 2] = (
                [1, 0] if player == Cell.O else [0, 1]
            )

        # Determine next required board based on the move's location
        self.next_board = inner_index
        next_board: TicTacToe = cast(TicTacToe, self.boards[self.next_board])

        # Determine if next player can choose board
        if next_board.check_winner() or not np.any(next_board.board == Cell.B.value):
            self.next_board = None

    # A little thrown together but this isn't useful for the AI anyways
    # Also, not entirely sure why the last column of blanks is slightly smaller, but that might just be on my machine
    def display_board(self):
        """
        Displays the current state of the SuperTicTacToe board in a human-readable format.
        """
        for super_row in range(3):
            for inner_row in range(3):
                row_parts: list[str] = []
                for outer_col in range(3):
                    board: TicTacToe = cast(
                        TicTacToe, self.boards[super_row, outer_col]
                    )
                    row = board.board[inner_row]
                    row_str: str = " ".join(
                        "🀫" if Cell(cell).name == "B" else Cell(cell).name
                        for cell in row
                    )
                    row_parts.append(row_str)
                print(" || ".join(row_parts))
            if super_row < 2:
                print("=" * 25)

    def get_winner_matrix(self) -> NDArray[np.int8]:
        """
        Return a 3x3 matrix of the current winner of each sub-board.
        Each cell is:
            Cell.X → 1
            Cell.O → 2
            Cell.B (no winner yet) → 0
        """
        winners = np.zeros((3, 3), dtype=np.int8)
        for i in range(3):
            for j in range(3):
                board: TicTacToe = cast(TicTacToe, self.boards[i, j])
                winner = board.check_winner()
                winners[i, j] = winner.value if winner is not None else Cell.B.value
        return winners

    def check_winner(self) -> Cell | None:
        """
        Checks if there's a winner (either player X or O).

        Returns:
            `Cell.X` if player X wins, `Cell.O` if player O wins, or `None` if no winner.
        """
        winners = self.get_winner_matrix()

        # Check rows for a win
        for row in winners:
            if row[0] == row[1] == row[2] != Cell.B.value:
                return Cell(row[0])

        # Check columns for a win
        for col in range(3):
            if (
                self.boards[0, col]
                == self.boards[1, col]
                == self.boards[2, col]
                != Cell.B.value
            ):
                return Cell(self.boards[0, col])

        # Check diagonals for a win
        if self.boards[0, 0] == self.boards[1, 1] == self.boards[2, 2] != None:
            return Cell(self.boards[0, 0])

        if self.boards[0, 2] == self.boards[1, 1] == self.boards[2, 0] != None:
            return Cell(self.boards[0, 2])

        return None  # No winner

    def X_flatten_board(self) -> NDArray[np.int8]:
        """
        Return the flattened SuperTicTacToe board containing 180 binary values.

        Each cell is encoded as:
            X -> [1, 0]
            O -> [0, 1]
            B -> [0, 0]

        @return A NumPy array of shape (180,) with 0s and 1s.
        """
        return self.X_flat_board

    def O_flatten_board(self) -> NDArray[np.int8]:
        """
        Return the flattened SuperTicTacToe board containing 180 binary values.

        Each cell is encoded as:
            X -> [1, 0]
            O -> [0, 1]
            B -> [0, 0]

        @return A NumPy array of shape (180,) with 0s and 1s.
        """
        return self.O_flat_board


# For testing purposes only
def main():
    print("Welcome to Super Tic Tac Toe!")
    game = SuperTicTacToe()
    current_player = Cell.X  # Start with Player X

    while True:
        try:
            print("\nCurrent state of the Super Board:")
            game.display_board()
            # print(game.X_flatten_board())
            # print(game.O_flatten_board())

            # Only ask for board number if the player is allowed to choose any board
            outer_index: tuple[int, int]
            if game.next_board is None:
                outer_row = int(input("Choose an outer row (1-3): ")) - 1
                outer_col = int(input("Choose an outer column (1-3): ")) - 1
                outer_index = outer_row, outer_col
            else:
                outer_index = game.next_board
                print(
                    f"\nPlayer {current_player.name}, you must play in board {outer_index[0]+1}, {outer_index[1]+1}."
                )

            inner_row = int(input("Choose an inner row (1-3): ")) - 1
            inner_col = int(input("Choose an inner column (1-3): ")) - 1
            inner_index: tuple[int, int] = inner_row, inner_col

            game.make_move(outer_index, inner_index, current_player)

            # Switch players
            current_player = Cell.O if current_player == Cell.X else Cell.X

        except Exception as e:
            print(f"Error: {e}")

        if game.check_winner() != None:
            break


if __name__ == "__main__":
    main()
