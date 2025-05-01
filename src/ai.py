import torch
from torch import nn
import numpy as np
import random
import tictactoe as TTT

class TicTacMaster(nn.Module):
    def __init__(self):
        super().__init__()
        # 9*9*2+9*2+9 = 189, for the input board + winner matrix + next board
        # 9*9 = 81, for the output, choosing a square to place on
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(189, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 81),
            nn.Softmax(dim=-1) # Necessary because we want the bot's credence for the best move
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

def bot_vs_random(model, tests, display):
    bot_wins = 0
    rand_wins = 0

    model.eval()
    for i in range(tests):
        game = TTT.SuperTicTacToe()
        current_player = TTT.Cell.X  # Start with Player X

        bot_character = random.choice([TTT.Cell.X, TTT.Cell.O])
        rand_character = TTT.Cell.X if bot_character == TTT.Cell.O else TTT.Cell.O
        while True:
            if current_player == rand_character:
                move = tic_tac_rand(game)
            else:
                with torch.no_grad():
                    if bot_character == TTT.Cell.X:
                        model_output = model.forward(torch.Tensor(game.X_flatten_board()))
                        move = square_to_tuple(max_valid_move(game, model_output))
                    else:
                        model_output = model.forward(torch.Tensor(game.O_flatten_board()))
                        move = square_to_tuple(max_valid_move(game, model_output))
            game.make_move(move[0], move[1], current_player)
            # Switch players
            current_player = TTT.Cell.O if current_player == TTT.Cell.X else TTT.Cell.X
            winner = game.check_winner()
            if winner is not TTT.Cell.B:
                if winner == bot_character:
                    if display:
                        print("The bot won!")
                    bot_wins += 1
                elif winner == rand_character:
                    if display:
                        print("The random player won!")
                    rand_wins += 1
                if display:
                    print("=" * 25)
                    print("=" * 25)
                    game.display_board()
                break
        if i % (tests/1000) == 0:
            print(f"{round(i / tests * 100, 3)}%")
    print(f"Bot wins: {round(bot_wins / tests * 100, 3)}%")
    print(f"Random wins: {round(rand_wins / tests * 100, 3)}%")

def train(model: TicTacMaster, epochs: int, learning_rate: float, bot_portion: float, random_portion: float,
          batch_size: int):
    import src.tictactoe as TTT
    # Set up the hyperparameters
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(epochs): # Here, epochs * batch_size is the total number of games the bot is trained on.
        cur_move = 0
        cur_move_b = 0
        board_history = []
        move_history = []

        board_history_b = []
        move_history_b = []

        tensor_correction = []
        tensor_correction_b = []
        for b in range(batch_size):
            against_bot = random.random() < bot_portion

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
                        is_choice_random = random.random() < random_portion
                        if bot_character == TTT.Cell.X:
                            if not is_choice_random:
                                model_output = model.forward(torch.Tensor(game.X_flatten_board()))
                            else:
                                model_output = torch.rand(81)
                            board_history.append(game.X_flatten_board())
                        else:
                            if not is_choice_random:
                                model_output = model.forward(torch.Tensor(game.O_flatten_board()))
                            else:
                                model_output = torch.rand(81)
                            board_history.append(game.O_flatten_board())
                        move_made = max_valid_move(game, model_output)
                        outer_index, inner_index = square_to_tuple(move_made)
                        move_history.append(move_made)
                    else:
                        if not against_bot:
                            outer_index, inner_index = tic_tac_rand(game)
                        else:
                            is_choice_random = random.random() < random_portion
                            if bot_character == TTT.Cell.X:
                                if not is_choice_random:
                                    model_output = model.forward(torch.Tensor(game.O_flatten_board()))
                                else:
                                    model_output = torch.rand(81)
                                board_history_b.append(game.O_flatten_board())
                            else:
                                if not is_choice_random:
                                    model_output = model.forward(torch.Tensor(game.X_flatten_board()))
                                else:
                                    model_output = torch.rand(81)
                                board_history_b.append(game.X_flatten_board())
                            move_made = max_valid_move(game, model_output)
                            outer_index, inner_index = square_to_tuple(move_made)
                            move_history_b.append(move_made)

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

            if winner == bot_character:
                # If the bot won, reinforce its actions by telling it to increase its confidence in the moves it made.
                for m in move_history[cur_move:]:
                    new_tensor = np.zeros(81)
                    new_tensor[m] = 1
                    tensor_correction.append(new_tensor)
                # If bot playing against itself, penalize it for losing too.
                if against_bot:
                    for m in range(cur_move_b, len(move_history_b)):
                        new_tensor = np.ones(81)
                        new_tensor[move_history_b[m]] = 0
                        tensor_correction_b.append(new_tensor)
            elif winner == TTT.Cell.X or winner == TTT.Cell.O:
                # This will set the corrections to a 1 for each valid move OTHER than the one made for a losing bot.
                for m in range(cur_move, len(move_history)):
                    new_tensor = np.ones(81)
                    new_tensor[move_history[m]] = 0
                    tensor_correction.append(new_tensor)
                # Since the bot also won against itself, it should be rewarded.
                if against_bot:
                    for m in move_history_b[cur_move_b:]:
                        new_tensor = np.zeros(81)
                        new_tensor[m] = 1
                        tensor_correction_b.append(new_tensor)
            else: # In the case of a draw, both sides are punished
                for m in range(cur_move, len(move_history)):
                    new_tensor = np.ones(81)
                    new_tensor[move_history[m]] = 0
                    tensor_correction.append(new_tensor)
                if against_bot:
                    for m in range(cur_move_b, len(move_history_b)):
                        new_tensor = np.ones(81)
                        new_tensor[move_history_b[m]] = 0
                        tensor_correction_b.append(new_tensor)
            cur_move = len(move_history)
            cur_move_b = len(move_history_b)
        # Combine the history of all moves and board states
        tensor_correction += tensor_correction_b
        board_history += board_history_b
        board_history = torch.Tensor(np.array(board_history))
        corrections = torch.Tensor(np.array(tensor_correction))
        model.train()
        # Calculate error between credence and correct move
        outputs = model(board_history)
        #print(outputs, corrections)
        loss = loss_fn(outputs, corrections)
        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % (epochs/1000) == 0 or epochs < 1000:
            print(f"{round(i/epochs*100,3)}%")
    torch.save(model.state_dict(), "files/model_weights.pth")
