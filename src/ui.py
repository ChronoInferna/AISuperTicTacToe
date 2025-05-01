import tictactoe as TTT
import ai as TTM

import pygame as pg

import random
import torch


tiles = [
    (pg.Rect(25, 23, 45, 46), (0, 0, 0, 0)),
    (pg.Rect(86, 23, 45, 46), (0, 0, 0, 1)),
    (pg.Rect(146, 23, 45, 46), (0, 0, 0, 2)),
    (pg.Rect(223, 23, 45, 46), (0, 1, 0, 0)),
    (pg.Rect(284, 23, 45, 46), (0, 1, 0, 1)),
    (pg.Rect(344, 23, 45, 46), (0, 1, 0, 2)),
    (pg.Rect(422, 23, 45, 46), (0, 2, 0, 0)),
    (pg.Rect(483, 23, 45, 46), (0, 2, 0, 1)),
    (pg.Rect(544, 23, 45, 46), (0, 2, 0, 2)),
    (pg.Rect(25, 84, 45, 46), (0, 0, 1, 0)),
    (pg.Rect(86, 84, 45, 46), (0, 0, 1, 1)),
    (pg.Rect(146, 84, 45, 46), (0, 0, 1, 2)),
    (pg.Rect(223, 84, 45, 46), (0, 1, 1, 0)),
    (pg.Rect(284, 84, 45, 46), (0, 1, 1, 1)),
    (pg.Rect(344, 84, 45, 46), (0, 1, 1, 2)),
    (pg.Rect(422, 84, 45, 46), (0, 2, 1, 0)),
    (pg.Rect(483, 84, 45, 46), (0, 2, 1, 1)),
    (pg.Rect(544, 84, 45, 46), (0, 2, 1, 2)),
    (pg.Rect(25, 145, 45, 46), (0, 0, 2, 0)),
    (pg.Rect(86, 145, 45, 46), (0, 0, 2, 1)),
    (pg.Rect(146, 145, 45, 46), (0, 0, 2, 2)),
    (pg.Rect(223, 145, 45, 46), (0, 1, 2, 0)),
    (pg.Rect(284, 145, 45, 46), (0, 1, 2, 1)),
    (pg.Rect(344, 145, 45, 46), (0, 1, 2, 2)),
    (pg.Rect(422, 145, 45, 46), (0, 2, 2, 0)),
    (pg.Rect(483, 145, 45, 46), (0, 2, 2, 1)),
    (pg.Rect(544, 145, 45, 46), (0, 2, 2, 2)),
    (pg.Rect(25, 221, 45, 46), (1, 0, 0, 0)),
    (pg.Rect(86, 221, 45, 46), (1, 0, 0, 1)),
    (pg.Rect(146, 221, 45, 46), (1, 0, 0, 2)),
    (pg.Rect(223, 221, 45, 46), (1, 1, 0, 0)),
    (pg.Rect(284, 221, 45, 46), (1, 1, 0, 1)),
    (pg.Rect(344, 221, 45, 46), (1, 1, 0, 2)),
    (pg.Rect(422, 221, 45, 46), (1, 2, 0, 0)),
    (pg.Rect(483, 221, 45, 46), (1, 2, 0, 1)),
    (pg.Rect(544, 221, 45, 46), (1, 2, 0, 2)),
    (pg.Rect(25, 282, 45, 46), (1, 0, 1, 0)),
    (pg.Rect(86, 282, 45, 46), (1, 0, 1, 1)),
    (pg.Rect(146, 282, 45, 46), (1, 0, 1, 2)),
    (pg.Rect(223, 282, 45, 46), (1, 1, 1, 0)),
    (pg.Rect(284, 282, 45, 46), (1, 1, 1, 1)),
    (pg.Rect(344, 282, 45, 46), (1, 1, 1, 2)),
    (pg.Rect(422, 282, 45, 46), (1, 2, 1, 0)),
    (pg.Rect(483, 282, 45, 46), (1, 2, 1, 1)),
    (pg.Rect(544, 282, 45, 46), (1, 2, 1, 2)),
    (pg.Rect(25, 343, 45, 46), (1, 0, 2, 0)),
    (pg.Rect(86, 343, 45, 46), (1, 0, 2, 1)),
    (pg.Rect(146, 343, 45, 46), (1, 0, 2, 2)),
    (pg.Rect(223, 343, 45, 46), (1, 1, 2, 0)),
    (pg.Rect(284, 343, 45, 46), (1, 1, 2, 1)),
    (pg.Rect(344, 343, 45, 46), (1, 1, 2, 2)),
    (pg.Rect(422, 343, 45, 46), (1, 2, 2, 0)),
    (pg.Rect(483, 343, 45, 46), (1, 2, 2, 1)),
    (pg.Rect(544, 343, 45, 46), (1, 2, 2, 2)),
    (pg.Rect(25, 421, 45, 46), (2, 0, 0, 0)),
    (pg.Rect(86, 421, 45, 46), (2, 0, 0, 1)),
    (pg.Rect(146, 421, 45, 46), (2, 0, 0, 2)),
    (pg.Rect(223, 421, 45, 46), (2, 1, 0, 0)),
    (pg.Rect(284, 421, 45, 46), (2, 1, 0, 1)),
    (pg.Rect(344, 421, 45, 46), (2, 1, 0, 2)),
    (pg.Rect(422, 421, 45, 46), (2, 2, 0, 0)),
    (pg.Rect(483, 421, 45, 46), (2, 2, 0, 1)),
    (pg.Rect(544, 421, 45, 46), (2, 2, 0, 2)),
    (pg.Rect(25, 482, 45, 46), (2, 0, 1, 0)),
    (pg.Rect(86, 482, 45, 46), (2, 0, 1, 1)),
    (pg.Rect(146, 482, 45, 46), (2, 0, 1, 2)),
    (pg.Rect(223, 482, 45, 46), (2, 1, 1, 0)),
    (pg.Rect(284, 482, 45, 46), (2, 1, 1, 1)),
    (pg.Rect(344, 482, 45, 46), (2, 1, 1, 2)),
    (pg.Rect(422, 482, 45, 46), (2, 2, 1, 0)),
    (pg.Rect(483, 482, 45, 46), (2, 2, 1, 1)),
    (pg.Rect(544, 482, 45, 46), (2, 2, 1, 2)),
    (pg.Rect(25, 543, 45, 46), (2, 0, 2, 0)),
    (pg.Rect(86, 543, 45, 46), (2, 0, 2, 1)),
    (pg.Rect(146, 543, 45, 46), (2, 0, 2, 2)),
    (pg.Rect(223, 543, 45, 46), (2, 1, 2, 0)),
    (pg.Rect(284, 543, 45, 46), (2, 1, 2, 1)),
    (pg.Rect(344, 543, 45, 46), (2, 1, 2, 2)),
    (pg.Rect(422, 543, 45, 46), (2, 2, 2, 0)),
    (pg.Rect(483, 543, 45, 46), (2, 2, 2, 1)),
    (pg.Rect(544, 543, 45, 46), (2, 2, 2, 2)),
]


# A function that prevents players from crashing the game with their invalid inputs.
def ask_until_valid(prompt, req):
    user_input = input(prompt)
    while not req(user_input):
        user_input = input("Invalid input. Please try again: ")
    return user_input


def guide_player(game, char):
    if game.next_board is None:
        print(f"Player {char}, you may go anywhere.")
    elif game.next_board == (0, 0):
        print(f"Player {char}, you must play in the top left board.")
    elif game.next_board == (0, 1):
        print(f"Player {char}, you must play in the top middle board.")
    elif game.next_board == (0, 2):
        print(f"Player {char}, you must play in the top right board.")
    elif game.next_board == (1, 0):
        print(f"Player {char}, you must play in the middle left board.")
    elif game.next_board == (1, 1):
        print(f"Player {char}, you must play in the middle board.")
    elif game.next_board == (1, 2):
        print(f"Player {char}, you must play in the middle right board.")
    elif game.next_board == (2, 0):
        print(f"Player {char}, you must play in the bottom left board.")
    elif game.next_board == (2, 1):
        print(f"Player {char}, you must play in the bottom middle board.")
    elif game.next_board == (2, 2):
        print(f"Player {char}, you must play in the bottom right board.")


def main():

    print("Welcome to Super Tic Tac Toe!")
    mode = ask_until_valid(
        "Would you like to play or train? ", lambda x: x.lower() in ["train", "play"]
    ).lower()
    if mode == "train":
        batch_size = int(
            ask_until_valid(
                "What batch size would you like? ",
                lambda x: x.isnumeric() and int(x) > 0,
            )
        )
        epochs = int(
            ask_until_valid(
                "How many epochs would you like? ",
                lambda x: x.isnumeric() and int(x) > 0,
            )
        )
        learning_rate = float(
            ask_until_valid(
                "What learning rate would you like? ",
                lambda x: x.replace(".", "", 1).isnumeric() and float(x) > 0,
            )
        )
        print("The following inputs should be a number from 0-1.")
        random_portion = float(
            ask_until_valid(
                "During training, what portion of the bot's actions should be random? ",
                lambda x: x.replace(".", "", 1).isnumeric() and 0 <= float(x) <= 1,
            )
        )
        bot_portion = float(
            ask_until_valid(
                "What portion of training games should the bot play against itself? ",
                lambda x: x.replace(".", "", 1).isnumeric() and 0 <= float(x) <= 1,
            )
        )
        print("Training...")

        model = TTM.TicTacMaster()
        TTM.train(model, epochs, learning_rate, bot_portion, random_portion, batch_size)

        print("Testing...")
        TTM.bot_vs_random(model, 10000, False)

    game_mode = int(
        ask_until_valid(
            "Would you like to:\n1. Play against another player.\n2. Play against a bot.\
        \n3. Test a bot against random players.\n",
            lambda x: x in ["1", "2", "3"],
        )
    )
    model = TTM.TicTacMaster()
    model.load_state_dict(torch.load("files/model_weights.pth", weights_only=False))
    if game_mode == 3:
        tests = int(
            ask_until_valid(
                "How many test games against random players would you like? ",
                lambda x: x.isnumeric() and int(x) > 0,
            )
        )
        display = (
            ask_until_valid(
                "Would you like to display the ends of each game? ",
                lambda x: x.lower() in ["yes", "no"],
            ).lower()
            == "yes"
        )
        TTM.bot_vs_random(model, tests, display)
    else:
        pg.init()
        screen = pg.display.set_mode((612, 612))
        clock = pg.time.Clock()
        running = True

        board = pg.transform.scale(pg.image.load("files/board.png"), (612, 612))
        board_rect = board.get_rect()
        X = pg.transform.scale(pg.image.load("files/x.png"), (46, 46))
        O = pg.transform.scale(pg.image.load("files/o.png"), (46, 46))

        screen.blit(board, board_rect)

        player = random.choice([TTT.Cell.X, TTT.Cell.O])
        game = TTT.SuperTicTacToe()
        current_player = TTT.Cell.X
        has_won = False

        while running:
            winner = game.check_winner()
            if not has_won and winner != TTT.Cell.B:
                player_as_char = "X" if winner == TTT.Cell.X else "O"
                print(f"{player_as_char} won!")
                has_won = True
            if game_mode == 2 and not has_won and current_player != player:
                bot_move = TTM.square_to_tuple(
                    TTM.max_valid_move(
                        game,
                        model.forward(
                            torch.Tensor(
                                game.X_flatten_board()
                                if current_player == TTT.Cell.X
                                else game.O_flatten_board()
                            )
                        ),
                    )
                )
                game.make_move(bot_move[0], bot_move[1], current_player)
                for rec1, tile1 in tiles:
                    if tile1 == (
                        bot_move[0][0],
                        bot_move[0][1],
                        bot_move[1][0],
                        bot_move[1][1],
                    ):
                        (
                            screen.blit(X, rec1)
                            if current_player == TTT.Cell.X
                            else screen.blit(O, rec1)
                        )
                        break
                current_player = (
                    TTT.Cell.O if current_player == TTT.Cell.X else TTT.Cell.X
                )
                player_as_char = "X" if current_player == TTT.Cell.X else "O"
                guide_player(game, player_as_char)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_r:  # Reset the game by pressing r
                        game = TTT.SuperTicTacToe()
                        current_player = TTT.Cell.X
                        screen.blit(board, board_rect)
                        player = random.choice([TTT.Cell.X, TTT.Cell.O])
                elif event.type == pg.MOUSEBUTTONDOWN and not has_won:
                    pos = pg.mouse.get_pos()
                    for rec, tile in tiles:
                        if rec.collidepoint(pos) and (
                            game.next_board is None
                            or (tile[0], tile[1]) == game.next_board
                        ):
                            if (
                                game.boards[tile[0], tile[1]].board[tile[2], tile[3]]
                                == TTT.Cell.B
                            ):
                                (
                                    screen.blit(X, rec)
                                    if current_player == TTT.Cell.X
                                    else screen.blit(O, rec)
                                )
                                game.make_move(
                                    (tile[0], tile[1]),
                                    (tile[2], tile[3]),
                                    current_player,
                                )
                                current_player = (
                                    TTT.Cell.O
                                    if current_player == TTT.Cell.X
                                    else TTT.Cell.X
                                )
                                if game_mode != 2:
                                    player_as_char = (
                                        "X" if current_player == TTT.Cell.X else "O"
                                    )
                                    guide_player(game, player_as_char)
                                break
            pg.display.flip()
            clock.tick(60) / 1000
        pg.quit()
    input()  # Prevent console window from closing before you can read results


if __name__ == "__main__":
    main()
