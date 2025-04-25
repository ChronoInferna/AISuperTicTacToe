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
            print(" | ".join('🀫' if Cell(cell).name == 'B' else Cell(cell).name for cell in row))
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
    def __init__ (self):
        self.boards = []
        for item in range(0,9):
            self.boards.append(TicTacToe())
        self.main_board = TicTacToe()
        self.next_board_index = None #tracks next move


    def make_move(self, boardNum: int, row: int, col: int, player: Cell):
        # Determine which board to play on - auto determined in some cases
        if self.next_board_index is None:
            if 1 <= boardNum <= 9:
                board_index = boardNum - 1
            else:
                raise ValueError("Board number must be between 1 and 9")
        else:
            board_index = self.next_board_index
            boardNum = board_index + 1
            print(f"Automatically directing to board {boardNum}.")

        board = self.boards[board_index]

        # If board is full or already won, allow player to choose any valid board
        if board.check_winner() or not np.any(board.board == Cell.B.value):
            print("Target board is full or won. Choose any valid board.")
            valid = False
            while not valid:
                boardNum = int(input(f"Player {player.name}, choose a board (1-9): "))
                if not (1 <= boardNum <= 9):
                    print("Board number must be between 1 and 9")
                    continue
                board_index = boardNum - 1
                board = self.boards[board_index]
                if board.check_winner() or not np.any(board.board == Cell.B.value):
                    print("That board is not available. Try again.")
                else:
                    valid = True

        # Make the move
        board.make_move(row, col, player)
        board.display_board()

        # Determine next required board based on the move's location
        self.next_board_index = row * 3 + col
        next_board = self.boards[self.next_board_index]

        # Next player can pick any board
        if next_board.check_winner() or not np.any(next_board.board == Cell.B.value):
            self.next_board_index = None


    #display each tic tac toe object in a board format
    def display_all_boards(self):
        for block_row in range(3):
            for row in range(3):
                line = []
                for block_col in range(3):  # For each board in that row
                    board_index = block_row * 3 + block_col
                    line.append(" | ".join('🀫' if Cell(cell).name == 'B' else Cell(cell).name for cell in self.boards[board_index].board[row]))
                print("   ||   ".join(line))
            if block_row < 2:
                print("=" * 50)


#for testing purposes only
def main():
    print("Welcome to Super Tic Tac Toe!")
    game = SuperTicTacToe()
    current_player = Cell.X  # Start with Player X

    while True:
        try:
            print("\nCurrent state of the Super Board:")
            game.display_all_boards()

            # Only ask for board number if the player is allowed to choose any board
            if game.next_board_index is None:
                board_num = int(input(f"\nPlayer {current_player.name}, choose a board (1-9): "))
            else:
                board_num = game.next_board_index + 1
                print(f"\nPlayer {current_player.name}, you must play in board {board_num}.")

            row = int(input("Choose a row (0-2): "))
            col = int(input("Choose a column (0-2): "))
            game.make_move(board_num, row, col, current_player)

            # Switch players
            current_player = Cell.O if current_player == Cell.X else Cell.X

        except Exception as e:
            print(f"Error: {e}")

# Run the game
if __name__ == "__main__":
    main()


