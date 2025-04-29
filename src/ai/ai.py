import torch
from torch import nn

import numpy as np

import random

class TicTacMaster(nn.Module):
    def __init__(self):
        super().__init__()
        # 9*9*2+9*2 = 180, for the input flattened board + one hot encoding
        # 9*9 = 81, for the output, choosing a square to place on
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(189, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 81),
            nn.Softmax() # Necessary because we want the bot's credence for the best move
        )

    def forward(self, x):
        # No need to use flatten since "x" is already 1D.
        return self.linear_relu_stack(x)


# Takes a game and a list of credences, chooses the highest credence that is a VALID move.
# Returns the number of the square move. A number from 0-81.
def max_valid_move(game, credence_tensor: torch.Tensor) -> int:
    max_index = None
    max_val = -1
    for square in range(81):
        if game.valid_moves[square] == 1:
            if credence_tensor[square] > max_val:
                max_index = square
                max_val = credence_tensor[square]
    return max_index

# Turn a number of a square into the relevant tuples used to make a move.
def square_to_tuple(square: int) -> tuple[tuple[int, int], tuple[int, int]]:
    row = square // 27
    col = square % 27 // 9
    row_m = square % 9 // 3
    col_m = square % 9 % 3
    return (row, col), (row_m, col_m)

#Random player for testing purposes
def tic_tac_rand(board):
    rt = torch.rand(81)
    move_in, move_out = square_to_tuple(max_valid_move(board, rt))
    return move_in, move_out


def train(model: TicTacMaster, epochs: int, learning_rate: float, path: str, portion: float):
    import src.core.tictactoe as TTT
    # Set up the hyperparameters
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(epochs): # Here, epochs is the number of games to train on.
        board_history = []
        move_history = []
        valid_move_history = []

        board_history_b = []
        move_history_b = []
        valid_move_history_b = []

        against_bot = random.random() < portion

        '''
        Game move logic here:
        =======
        '''
        game = TTT.SuperTicTacToe()
        current_player = TTT.Cell.X  # Start with Player X

        bot_character = random.choice([TTT.Cell.X, TTT.Cell.O])

        with torch.no_grad(): # Reduces computations
            model.eval()
            while True:
                if current_player == bot_character:
                    if bot_character == TTT.Cell.X:
                        model_output = model.forward(torch.Tensor(game.X_flatten_board()))
                        board_history.append(game.X_flatten_board())
                    else:
                        model_output = model.forward(torch.Tensor(game.O_flatten_board()))
                        board_history.append(game.O_flatten_board())
                    move_made = max_valid_move(game, model_output)
                    outer_index, inner_index = square_to_tuple(move_made)
                    move_history.append(move_made)
                    valid_move_history.append(game.get_valid_moves())
                else:
                    if not against_bot:
                        outer_index, inner_index = tic_tac_rand(game)
                    else:
                        if bot_character == TTT.Cell.X:
                            model_output = model.forward(torch.Tensor(game.O_flatten_board()))
                            board_history_b.append(game.O_flatten_board())
                        else:
                            model_output = model.forward(torch.Tensor(game.X_flatten_board()))
                            board_history_b.append(game.X_flatten_board())
                        move_made = max_valid_move(game, model_output)
                        outer_index, inner_index = square_to_tuple(move_made)
                        move_history_b.append(move_made)
                        valid_move_history_b.append(game.get_valid_moves())

                game.make_move(outer_index, inner_index, current_player)
                # Switch players
                current_player = TTT.Cell.O if current_player == TTT.Cell.X else TTT.Cell.X
                winner = game.check_winner()
                if winner != TTT.Cell.B:
                    break
        '''
        =======
        '''

        #TRAINING
        winner = game.check_winner()

        tensor_correction = []
        tensor_correction_b = []
        if winner == bot_character:
            # If the bot won, reinforce its actions by telling it to increase its confidence in the moves it made.
            for m in move_history:
                new_tensor = np.zeros(81)
                new_tensor[m] = 1
                tensor_correction.append(new_tensor)
            # If bot playing against itself, penalize it for losing too.
            if against_bot:
                for m in range(len(move_history_b)):
                    new_tensor = valid_move_history_b[m]
                    new_tensor[move_history_b[m]] = 0
                    tensor_correction_b.append(new_tensor)
        elif winner == TTT.Cell.X or winner == TTT.Cell.O:
            # This will set the corrections to a 1 for each valid move OTHER than the one made for a losing bot.
            for m in range(len(move_history)):
                new_tensor = valid_move_history[m]
                new_tensor[move_history[m]] = 0
                tensor_correction.append(new_tensor)
            # Since the bot also won against itself, it should be rewarded.
            if against_bot:
                for m in move_history_b:
                    new_tensor = np.zeros(81)
                    new_tensor[m] = 1
                    tensor_correction_b.append(new_tensor)
        else: # In the case of a draw, both sides are punished
            for m in range(len(move_history)):
                new_tensor = valid_move_history[m]
                new_tensor[move_history[m]] = 0
                tensor_correction.append(new_tensor)
            if against_bot:
                for m in range(len(move_history_b)):
                    new_tensor = valid_move_history_b[m]
                    new_tensor[move_history_b[m]] = 0
                    tensor_correction_b.append(new_tensor)

        tensor_correction = np.array(tensor_correction)
        board_history = np.array(board_history)
        model.train()
        corrections = torch.Tensor(tensor_correction)
        if against_bot:
            board_history_b = np.array(board_history_b)
            tensor_correction_b = np.array(tensor_correction_b)
            corrections_b = torch.Tensor(tensor_correction_b)
        for i1 in range(10):
            # Calculate error between credence and correct move
            outputs = model(torch.Tensor(board_history))
            #print(outputs, corrections)
            loss = loss_fn(outputs, corrections)
            # Backprop
            loss.backward()
            if against_bot:
                outputs = model(torch.Tensor(board_history_b))
                # print(outputs, corrections)
                loss = loss_fn(outputs, corrections_b)
                # Backprop
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if i % (epochs/1000) == 0:
            print(f"{round(i/epochs*100,3)}%")
    torch.save(model.state_dict(), path)
