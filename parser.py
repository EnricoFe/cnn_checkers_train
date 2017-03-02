# Title: parser.py
# Author: Chris Larson
# CS-6700 Final Project
# This program parses a .txt file containing ~20K checkers games and extracts each of the ...
# board-move pairs into one of three pandas dataframes depending on whether that move was part ...
# of a win, loss, or draw.

# Comments ========================================================================================
#
# Example entry from OCA_2.0.pdn:
#
# [Event "Manchester 1841"]
# [Date "1841-??-??"]
# [Black "Moorhead, W."]
# [White "Wyllie, J."]
# [Site "Manchester"]
# [Result "0-1"]
# 1. 11-15 24-20 2. 8-11 28-24 3. 9-13 22-18 4. 15x22 25x18 5. 4-8 26-22 6. 10-14
# 18x9 7. 5x 14 22-18 8. 1-5 18x9 9. 5x14 29-25 10. 11-15 24-19 11. 15x24 25-22 12.
# 24-28 22-18 13. 6-9 27-24 14. 8-11 24-19 15. 7-10 20-16 16. 11x20 18-15 17. 2-6
# 15-11 18. 12-16 19x12 19. 10-15 11-8 20. 15-18 21-17 21. 13x22 30-26 22. 18x27
# 26x17x10x1 0-1
#
# Board encoding:
#
# From OCA_2.0.pdn:
# 32 31 30 29
# 28 27 26 25
# 24 23 22 21
# 20 19 18 17
# 16 15 14 13
# 12 11 10 09
# 08 07 06 05
# 04 03 02 01
#
# This program:
# 00 01 02 03
# 04 05 06 07
# 08 09 10 11
# 12 13 14 15
# 16 17 18 19
# 20 21 22 23
# 24 25 26 27
# 28 29 30 31
#
# Board transformation: orig_pos - 1
# Flip board: 32 - orig_pos
#
# Model output: 32x4 vector denoting piece to move (col) and direction (row)
#
# Multiple jumps are treated as separate moves for simplicity. During game play, the program ...
# will check if they are available after each move.
#
# end comments ====================================================================================

import os
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle


class Board(object):

    global initial_board, jumps, empty, odd_list, even_list

    def __init__(self):
        self.state = initial_board.copy()

    def board_state(self, player_type):
        if player_type == 'white':
            return -self.state.iloc[::-1, ::-1]
        elif player_type == 'black':
            return self.state

    @staticmethod
    def move(positions, player_type):

        # Create empty training label
        label = pd.DataFrame(np.zeros([32, 4]))
        pos_init, pos_final = positions[0], positions[1]

        # Extract the initial and final positions into ints, assign move to label
        if player_type == 'black':
            pos_init, pos_final = int(pos_init) - 1, int(pos_final) - 1

        elif player_type == 'white':
            pos_init, pos_final = 32 - int(pos_init), 32 - int(pos_final)

        # Encode move up, right, left, down
        delta = pos_final - pos_init

        direction = None
        if pos_init in odd_list:
            if delta == -3 or delta == -7:
                direction = 0  # Up Right
            elif delta == 5 or delta == 9:
                direction = 1  # Down Right
            elif delta == 4 or delta == 7:
                direction = 2  # Down Left
            elif delta == -4 or delta == -9:
                direction = 3   # Up Left

        elif pos_init in even_list:
            if delta == -4 or delta == -7:
                direction = 0   # Up Right
            elif delta == 4 or delta == 9:
                direction = 1   # Down Right
            elif delta == 3 or delta == 7:
                direction = 2   # Down Left
            elif delta == -5 or delta == -9:
                direction = 3   # Up Left

        label.iloc[pos_init, direction] = 1
        return label

    def update(self, positions, player_type, move_type):

        if player_type == 'black':
            king_pos = black_king_pos
            king_value = black_king
            chkr_value = black_chkr
        else:
            king_pos = white_king_pos
            king_value = white_king
            chkr_value = white_chkr

        # Extract the initial and final positions into ints
        [pos_init, pos_final] = positions[0], positions[1]

        # Vectorize the board to set final pos value = initial pos value >>> inital pos value = 0
        pos_init, pos_final = int(pos_init) - 1, int(pos_final) - 1
        # print(pos_init, pos_final)
        board_vec = self.state.copy()
        board_vec = np.reshape(board_vec.as_matrix(), (32,))

        if board_vec[pos_init] == chkr_value or king_value and board_vec[pos_final] == empty:
            board_vec[pos_final] = board_vec[pos_init]
            board_vec[pos_init] = empty

            # Assign kings
            if pos_final in king_pos:
                board_vec[pos_final] = king_value

            # Remove eliminated pieces
            if move_type == 'jump':
                eliminated = int(jumps.iloc[pos_init, pos_final])
                # print('Position eliminated: %d' % eliminated)
                assert board_vec[eliminated] == -chkr_value or -king_value
                board_vec[eliminated] = empty

            # Update the board
            board_vec = pd.DataFrame(np.reshape(board_vec, (8, 4)))
            self.state = board_vec
            return False
        else:
            return True


def remove_any_comments(data):
    # Remove any parenthetical comments
    if '{' in data or '}' in data:
        str1 = data[0: data.index('{')]
        str2 = data[data.index('}') + 1:]
        return str1.join(str2)
    else:
        return data


def make_single_line(data):
    data = data.split('\n')
    return ''.join(data)


def tokenize_moves(data):
    move_tokens = data.split('.')
    return move_tokens[1:]


def parse_moves_into_dict(moves, contestants):
    for player in contestants:
        ind_player = contestants.index(player)

        if len(moves[ind_player].split('-')) == 2:
            entries = moves[ind_player].split('-')

            try:
                if int(entries[0]) and int(entries[1]) in valid_entries:
                    player_state = board.board_state(player_type=player)
                    player_move = board.move(entries, player_type=player)
                    game_corrupt = board.update(entries,
                                                player_type=player,
                                                move_type='standard')
                else:
                    game_corrupt = True
                    game_corrupt_count += 1
                    print('Corrupt game # %d: Invalid Entry *' % game_corrupt_count)
                    print(event)
                    print(entries[0], entries[1])
            except ValueError:
                game_corrupt = True
                game_corrupt_count += 1
                print('Corrupt game # %d: entry non-convertable to integer'
                      % game_corrupt_count)
                print(event)
                print(entries[0], entries[1])

        elif len(moves[ind_player].split('x')) >= 2:
            entries = moves[ind_player].split('x')

            for i in range(len(entries) - 1):

                try:
                    if int(entries[i]) and int(entries[i + 1]) in valid_entries:
                        player_state = board.board_state(player_type=player)
                        player_move = board.move(entries[i: i + 2], player_type=player)
                        game_corrupt = board.update(entries[i: i + 2],
                                                    player_type=player,
                                                    move_type='jump')
                    else:
                        game_corrupt = True
                        game_corrupt_count += 1
                        print('Corrupt game # %d: Invalid Entry **' % game_corrupt_count)
                        print(event)
                        print(entries[i], entries[i+1])
                except ValueError:
                    game_corrupt = True
                    game_corrupt_count += 1
                    print('Corrupt game # %d: entry non-convertable to integer'
                          % game_corrupt_count)
                    print(event)
                    print(entries[i], entries[i+1])

        else:
            game_corrupt = True
            game_corrupt_count += 1
            print("Corrupt game # %d: moves could not be split on '-' or 'x'"
                  % game_corrupt_count)
            print(event)
            print(moves)

        if not game_corrupt:

            if player == result and g == 'win':
                win_dict[win_counter] = [player_state, player_move]
                win_counter += 1
            elif result == 'draw' and g == 'draw':
                draw_dict[draw_counter] = [player_state, player_move]
                draw_counter += 1
            elif g == 'lose':
                loss_dict[loss_counter] = [player_state, player_move]
                loss_counter += 1
        else:
            break

# if __name__ == '__main__':
def main():
    # Create Win, Loss, & Draw dicts and counters that will be used as keys
    win_dict, win_counter = dict(), 0
    loss_dict, loss_counter = dict(), 0
    draw_dict, draw_counter = dict(), 0

    # Define initial board assignments, and jumps
    global initial_board, jumps
    initial_board = pd.read_csv(filepath_or_buffer='board_init.csv', header=-1, index_col=None)
    jumps = pd.read_csv(filepath_or_buffer='jumps.csv', header=-1, index_col=None)

    # Define board entries and valid positions
    empty = 0
    black_chkr = 1
    black_king = 3
    black_king_pos = [28, 29, 30, 31]
    white_chkr = -black_chkr
    white_king = -black_king
    white_king_pos = [0, 1, 2, 3]
    valid_positions = range(32)
    valid_entries = range(1, 33)

    # Parse file
    game_count = 0
    board = 0
    result = 0
    data = list()
    contestants = ['black', 'white']
    data_source = 'OCA_2.0.pdn'
    odd_list = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    even_list = [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]
    game_corrupt = False
    game_corrupt_count = 0

    print('Parsing document. This may take a while ...')
    for g in ['win', 'lose', 'draw']:
        game_corrupt = False
        game_corrupt_count = 0
        event = ''
        with open(data_source, 'r') as f:

            while True:
                line = f.readline()

                if 'Event' in line:
                    event = line
                elif 'Result' in line or 'result' in line:
                    if '1-0' in line:
                        result = 'black'
                    elif '1/2-1/2' in line:
                        result = 'draw'
                    elif '0-1' in line:
                        result = 'white'
                elif line[0:2] == '1.':
                    board = Board()
                    while line.strip() != '':
                        data.append(line)
                        line = f.readline()

                    data = ' '.join(data)
                    data = remove_any_comments(data)
                    data = make_single_line(data)

                    for moves in tokenize_moves(data):
                        moves = moves.strip().split(' ')

                        if len(moves) == 1:
                            contestants = ['black']

                        if len(moves) in range(1, 3):
                            parse_moves_into_dict(moves, contestants)
                        else:
                            print('Corrupt game: Moves length out of range, line %d' % lineno
                            break

                        board = 0
                        result = 0
                        data = list()
                        contestants = ['black', 'white']

                        if not game_corrupt:
                            game_count += 1

                        game_corrupt = False

                        if game_count % 1000 == 0:
                            print('%d games parsed so far' % game_count)

                # # Optional
                # if game_count == 500:
                #     break

        f.close()
        print('Parsing complete.')

        # Save win, loss, draw dicts into binaries for later use
        d = {}
        if g == 'win':
            checkers_library = 'checkers_library_win.pickle'
            d = win_dict
        elif g == 'draw':
            checkers_library = 'checkers_library_draw.pickle'
            d = draw_dict
        elif g == 'lose':
            checkers_library = 'checkers_library_lose.pickle'
            d = loss_dict
        print('Saving boards and moves to %s ...' % checkers_library)
        try:
            file_ = open(checkers_library, 'wb')
            data = {'library': d}
            pickle.dump(data, file_, pickle.HIGHEST_PROTOCOL)

        except Exception as exception:
            print('Unable to save data to' + checkers_library + ': %d' % exception)
            raise

        print('Compressed checkers_library size: %d Bytes' % os.stat(checkers_library).st_size)
        print('Number of games logged: %d' % game_count)
        print('Number of winning entries: %d' % win_counter)
        print('Number of losing entries: %d' % loss_counter)
        print('Number of draw entries: %d' % draw_counter)
        print('Number of corrupt games: %d' % game_corrupt_count)


main()