import src.core.tictactoe as TTT
import src.ai.ai as TTM

import random
import torch

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
                move = TTM.tic_tac_rand(game)
            else:
                with torch.no_grad():
                    if bot_character == TTT.Cell.X:
                        model_output = model.forward(torch.Tensor(game.X_flatten_board()))
                        move = TTM.square_to_tuple(
                            TTM.max_valid_move(game, model_output))
                    else:
                        model_output = model.forward(torch.Tensor(game.O_flatten_board()))
                        move = TTM.square_to_tuple(
                            TTM.max_valid_move(game, model_output))
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


# For testing purposes only
def main():
    print("Welcome to Super Tic Tac Toe!")
    mode = input("Would you like to play or train? ").lower()
    if mode == "train":
        epochs = int(input("How many epochs would you like? "))
        tests = int(input("Post training, how many test games against random players would you like? "))
        portion = float(input("What portion of training games should the bot play against itself? The rest will be played against random players. Enter a number from 0-1. "))

        model = TTM.TicTacMaster()
        model.load_state_dict(torch.load('model_weights.pth', weights_only=False))
        TTM.train(model, epochs, .1, "model_weights.pth", portion)

        bot_vs_random(model, tests, False)
    else:
        game_mode = int(input("Would you like to:\n1. Play against another player.\n2. Play against a bot.\
        \n3. Test a bot against random players.\n"))
        if game_mode == 1: # PVP
            game = TTT.SuperTicTacToe()
            current_player = TTT.Cell.X  # Start with Player X
            while True:
                try:
                    print("\nCurrent state of the Super Board:")
                    game.display_board()

                    # Only ask for board number if the player is allowed to choose any board

                    outer_index: tuple[int, int]
                    if game.next_board is None:
                        outer_row = int(input("Choose an outer row (1-3): ")) - 1
                        outer_col = int(input("Choose an outer column (1-3): ")) - 1
                        outer_index = outer_row, outer_col
                    else:
                        outer_index = game.next_board
                        print(
                            f"\nPlayer {current_player.name}, you must play in board {outer_index[0] + 1}, {outer_index[1] + 1}."
                        )

                    inner_row = int(input("Choose an inner row (1-3): ")) - 1
                    inner_col = int(input("Choose an inner column (1-3): ")) - 1
                    inner_index: tuple[int, int] = inner_row, inner_col

                    game.make_move(outer_index, inner_index, current_player)

                    # Switch players
                    current_player = TTT.Cell.O if current_player == TTT.Cell.X else TTT.Cell.X

                except Exception as e:
                    print(f"Error: {e}")

                if game.check_winner() != TTT.Cell.B:
                    print("Game ended!")
                    break
        else:
            model = TTM.TicTacMaster()
            model.load_state_dict(torch.load('model_weights.pth', weights_only=False))
            if game_mode == 3:
                tests = int(input("How many test games against random players would you like? "))
                display = input("Would you like to display the ends of each game? ").lower()=="yes"
                bot_vs_random(model, tests, display)
            elif game_mode == 2:
                game = TTT.SuperTicTacToe()
                current_player = TTT.Cell.X  # Start with Player X
                player = random.choice([TTT.Cell.X, TTT.Cell.O])
                bot = TTT.Cell.O if player==TTT.Cell.X else TTT.Cell.X
                while True:
                    try:
                        if current_player == player:
                            print("\nCurrent state of the Super Board:")
                            game.display_board()

                            # Only ask for board number if the player is allowed to choose any board

                            outer_index: tuple[int, int]
                            if game.next_board is None:
                                outer_row = int(input("Choose an outer row (1-3): ")) - 1
                                outer_col = int(input("Choose an outer column (1-3): ")) - 1
                                outer_index = outer_row, outer_col
                            else:
                                outer_index = game.next_board
                                print(
                                    f"\nPlayer {current_player.name}, you must play in board {outer_index[0] + 1}, {outer_index[1] + 1}."
                                )

                            inner_row = int(input("Choose an inner row (1-3): ")) - 1
                            inner_col = int(input("Choose an inner column (1-3): ")) - 1
                            inner_index: tuple[int, int] = inner_row, inner_col
                        else:
                            if bot == TTT.Cell.X:
                                move = TTM.square_to_tuple(
                                    TTM.max_valid_move(game, model.forward(torch.Tensor(game.X_flatten_board()))))
                            else:
                                move = TTM.square_to_tuple(
                                    TTM.max_valid_move(game, model.forward(torch.Tensor(game.O_flatten_board()))))
                            outer_index, inner_index = move

                        game.make_move(outer_index, inner_index, current_player)

                        # Switch players
                        current_player = TTT.Cell.O if current_player == TTT.Cell.X else TTT.Cell.X

                    except Exception as e:
                        print(f"Error: {e}")

                    if game.check_winner() != TTT.Cell.B:
                        print("Game ended!")
                        break


if __name__ == "__main__":
    main()
