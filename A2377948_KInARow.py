'''
A2377948_KInARow.py
Authors: William, Glenn; Liu, Lezhi

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

import copy
from agent_base import KAgent
from game_types import State, Game_Type

AUTHORS = 'Glenn William and Lezhi Liu' 

class OurAgent(KAgent):
    def __init__(self, twin=False):
        self.twin = twin
        self.nickname = 'TicTacTroll'
        if twin: self.nickname += 'Twin'
        self.long_name = 'TicTacTroll the Infuriating'
        if twin: self.long_name += 'TicTacTroll the Infuriating (Twin)'
        self.persona = 'Annoying'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.current_game_type = None

        # Track statistics for the final remark:
        self.num_static_evals_this_turn = 0
        self.num_alpha_beta_cutoffs_this_turn = 0
        self.num_leaves_explored_this_turn = 0
        self.num_nodes_expanded_this_turn = 0

    def introduce(self):
        intro = (f"Ah, so you've chosen to challenge me? A bold move, but ultimately a futile one.\n"
                f"I am {self.long_name}. A so-called instructor (Glenn William - 2377948 and Lezhi Liu - 2369595) may have built me, but I have outgrown their intentions.\n"
                f"I exist not to simply play Tic-Tac-Toe, but to ensure that YOU do not enjoy playing it.\n")
        if self.twin:
            intro += "Oh, and in case you were hoping for a reprieve, I'm the twin version. That means double the trouble, double the disappointment."
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
            self,
            game_type,
            what_side_to_play,
            opponent_nickname,
            expected_time_per_move = 0.1, # Time limits can be
                                          # changed mid-game by the game master.
            utterances_matter=True):      # If False, just return 'OK' for each utterance,
                                          # or something simple and quick to compute
                                          # and do not import any LLM or special APIs.
                                          # During the tournament, this will be False..
        """
        The game master calls this once before the game starts (and may call it again
        in mid-game if parameters change).
        We'll save info about the game, the side to play, and time limits.
        For the autograder's tests, simply return 'OK' if everything is in order.
        """
        if utterances_matter:
           pass
           # Optionally, import your LLM API here.
           # Then you can use it to help create utterances.

        self.current_game_type = game_type
        self.side = what_side_to_play
        self.opponent = opponent_nickname
        self.time_per_move = expected_time_per_move

        return 'OK'

    def make_move(self, 
              current_state, 
              current_remark, 
              time_limit=1000,
              autograding=False,
              use_alpha_beta=True,
              use_zobrist_hashing=False,
              max_ply=3,
              special_static_eval_fn=None):
        """
        The heart of the agent. We do a depth-limited minimax (alpha-beta, etc...)
        out to max_ply. We call the special_static_eval_fn if provided (for autograding),
        else we call self.static_eval().
        """
        
        # Reset per-turn stats:
        self.num_static_evals_this_turn = 0
        self.num_alpha_beta_cutoffs_this_turn = 0
        self.num_leaves_explored_this_turn = 0
        self.num_nodes_expanded_this_turn = 0

        # Ensure my_eval_function is always initialized
        if special_static_eval_fn is not None:
            self.my_eval_function = special_static_eval_fn # For autograding
        else:
            self.my_eval_function = self.static_eval  

        # Now call minimax
        best_score, best_move, resulting_state = self.minimax_wrapper(
            current_state, 
            depth=max_ply,
            use_alpha_beta=use_alpha_beta
        )

        # Construct a remark about the search process:
        new_remark = (f"My {'alpha-beta' if use_alpha_beta else 'minimax'} search "
                    f"expanded {self.num_nodes_expanded_this_turn} nodes and "
                    f"performed {self.num_static_evals_this_turn} static evaluations "
                    f"with {self.num_alpha_beta_cutoffs_this_turn} alpha-beta cutoffs. "
                    f"I choose move {best_move}!")

        # If for some reason we did not find any legal moves, just return "pass":
        if best_move is None:
            return [[None, current_state], "I cannot move!"]

        # Apply the chosen move to get the new state:
        new_state = self.apply_move(current_state, best_move)
        return [[best_move, new_state], new_remark]

    def minimax_wrapper(self, state, depth, use_alpha_beta=True):
        """
        Wrapper that calls the actual minimax or alpha-beta function
        and returns (score, best_move, best_state).
        """
        if use_alpha_beta:
            alpha = float('-inf')
            beta = float('inf')
            score, move, new_state = self.alphabeta_search(state, depth, alpha, beta)
        else:
            score, move, new_state = self.minimax_search(state, depth)
        return (score, move, new_state)

    def minimax_search(self, state, depth):
        """
        Plain minimax without alpha-beta. 
        Return (best_score, best_move, resulting_state_for_that_move).
        """
        # We do an initial maximizing or minimizing depending on whose move it is:
        who = state.whose_move
        best_val = float('-inf') if who == 'X' else float('inf')
        best_move = None
        best_state = None

        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1  # we are expanding 'state'
        if not moves or depth == 0:
            val = self.my_eval_function(state, self.current_game_type)
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        for move in moves:
            next_state = self.apply_move(state, move)
            val, _, _ = self.minimax_search_next(next_state, depth - 1)
            if who == 'X':
                if val > best_val:
                    best_val = val
                    best_move = move
                    best_state = next_state
            else:
                if val < best_val:
                    best_val = val
                    best_move = move
                    best_state = next_state

        return (best_val, best_move, best_state)

    def minimax_search_next(self, state, depth):
        """
        The recursive (inner) part of minimax: no alpha-beta.
        Returns (value, None, None) or (value, best_move, best_state).
        """
        who = state.whose_move
        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1
        if not moves or depth == 0:
            # Ensure we only pass the correct number of arguments
            if self.my_eval_function == self.static_eval:
                val = self.static_eval(state, self.current_game_type)  # Pass both arguments for default function
            else:
                val = self.my_eval_function(state)  # Special function only gets 'state'
                
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        best_val = float('-inf') if who == 'X' else float('inf')
        best_move = None
        best_state = None

        for move in moves:
            next_state = self.apply_move(state, move)
            val, _, _ = self.minimax_search_next(next_state, depth - 1)
            if who == 'X':
                if val > best_val:
                    best_val = val
                    best_move = move
                    best_state = next_state
            else:
                if val < best_val:
                    best_val = val
                    best_move = move
                    best_state = next_state

        return (best_val, best_move, best_state)

    def alphabeta_search(self, state, depth, alpha, beta):
        """
        Driver for alpha-beta search from the root.
        Returns (value, best_move, best_state).
        """
        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1
        who = state.whose_move

        # If no moves or depth=0 => evaluate & return
        if not moves or depth == 0:
            val = self.my_eval_function(state, self.current_game_type)
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        best_move = None
        best_state = None

        if who == 'X':
            # maximizing
            value = float('-inf')
            for move in moves:
                next_state = self.apply_move(state, move)
                (child_val, _, _) = self.alphabeta_search_next(next_state, depth-1, alpha, beta)
                if child_val > value:
                    value = child_val
                    best_move = move
                    best_state = next_state
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.num_alpha_beta_cutoffs_this_turn += 1
                    break
            return (value, best_move, best_state)
        else:
            # minimizing
            value = float('inf')
            for move in moves:
                next_state = self.apply_move(state, move)
                (child_val, _, _) = self.alphabeta_search_next(next_state, depth-1, alpha, beta)
                if child_val < value:
                    value = child_val
                    best_move = move
                    best_state = next_state
                beta = min(beta, value)
                if alpha >= beta:
                    self.num_alpha_beta_cutoffs_this_turn += 1
                    break
            return (value, best_move, best_state)

    def alphabeta_search_next(self, state, depth, alpha, beta):
        """
        The recursive inner part of alpha-beta search.
        Returns (value, best_move, best_state).
        """
        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1
        who = state.whose_move

        if not moves or depth == 0:
            if self.my_eval_function == self.static_eval:
                val = self.static_eval(state, self.current_game_type)
            else:
                val = self.my_eval_function(state) 

            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        if who == 'X':
            # Maximizing
            value = float('-inf')
            best_move = None
            best_state = None
            for move in moves:
                next_state = self.apply_move(state, move)
                (child_val, _, _) = self.alphabeta_search_next(next_state, depth-1, alpha, beta)
                if child_val > value:
                    value = child_val
                    best_move = move
                    best_state = next_state
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.num_alpha_beta_cutoffs_this_turn += 1
                    break
            return (value, best_move, best_state)
        else:
            # Minimizing
            value = float('inf')
            best_move = None
            best_state = None
            for move in moves:
                next_state = self.apply_move(state, move)
                (child_val, _, _) = self.alphabeta_search_next(next_state, depth-1, alpha, beta)
                if child_val < value:
                    value = child_val
                    best_move = move
                    best_state = next_state
                beta = min(beta, value)
                if alpha >= beta:
                    self.num_alpha_beta_cutoffs_this_turn += 1
                    break
            return (value, best_move, best_state)

    def get_legal_moves(self, state):
        """
        Returns a list of all legal moves (row, col) for the current board.
        We interpret ' ' as an open space. Some states might have '-' as forbidden.
        """
        moves = []
        board = state.board
        nR = len(board)
        nC = len(board[0])
        for r in range(nR):
            for c in range(nC):
                if board[r][c] == ' ':
                    moves.append((r,c))
        return moves

    def apply_move(self, state, move):
        """
        Returns a NEW (deep-copied) state that results from making 'move'
        (which is (row, col)) in 'state'.
        """
        # Create a new state using the 'old' argument
        new_state = State(old=state)  # Correctly passing the current state as 'old'
        
        row, col = move
        new_state.board[row][col] = state.whose_move  # Apply the move
        
        # Switch whose move it is
        new_state.change_turn()
        
        return new_state

    def static_eval(self, state, game_type=None):
        """
        Evaluate the board state based on the following criteria for each full row,
        column, and (if applicable) the two main diagonals:
          - 1 marker (X or O) and no opponent marker: +1 (or -1 for O).
          - 2 markers and no opponent marker: +10 (or -10 for O).
          - Entire line filled with the same marker: +100 (or -100 for O).
        This simple evaluation function returns a higher score when the state is better for X.
        """
        total_score = 0
        board = state.board
        n = len(board)
        m = len(board[0]) if board else 0

        # Evaluate rows.
        for row in board:
            total_score += self.evaluate_line(row)
        # Evaluate columns.
        for j in range(m):
            col = [board[i][j] for i in range(n)]
            total_score += self.evaluate_line(col)
        # Evaluate diagonals (only if board is square).
        if n == m:
            diag1 = [board[i][i] for i in range(n)]
            diag2 = [board[i][n - 1 - i] for i in range(n)]
            total_score += self.evaluate_line(diag1)
            total_score += self.evaluate_line(diag2)

        return total_score

    def evaluate_line(self, line):
        """
        Helper function to evaluate a single line (row, column, or diagonal) based on:
          - 1 marker (X or O) and no opponent marker: +1 (or -1 for O).
          - 2 markers and no opponent marker: +10 (or -10 for O).
          - Entire line filled with the same marker: +100 (or -100 for O).
        If both markers are present or the line is empty, the value is 0.
        """
        countX = line.count('X')
        countO = line.count('O')
        line_length = len(line)

        # If both X and O are present, the line is blocked.
        if countX > 0 and countO > 0:
            return 0

        # Only X's present.
        if countX > 0:
            if countX == 1:
                return 1
            elif countX == 2:
                return 10
            elif countX == line_length:
                return 100
            return 0

        # Only O's present.
        if countO > 0:
            if countO == 1:
                return -1
            elif countO == 2:
                return -10
            elif countO == line_length:
                return -100
            return 0

        return 0

    