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
import random

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
        self.zobrist_table_num_entries_this_turn = 0
        self.zobrist_table_num_hits_this_turn = 0
        self.zobrist_writes_this_turn = 0 
        self.current_game_type = None
        
        self.zobrist_table = {}
        self.transposition_table = {}

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
            utterances_matter=True,# If False, just return 'OK' for each utterance,
                                          # or something simple and quick to compute
                                          # and do not import any LLM or special APIs.
                                          # During the tournament, this will be False..
            apis_ok=False):      
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
        self.utterances_matter = utterances_matter
        self.apis_ok = apis_ok
        self.init_zobrist_table()

        return 'OK'

    def make_move(self, 
              current_state, 
              current_remark, 
              time_limit=1000,
              autograding=False,
              use_alpha_beta=True,
              use_zobrist_hashing=True,
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
        self.zobrist_writes_this_turn = 0
        self.zobrist_read_attempts_this_turn = 0
        self.zobrist_hits_this_turn = 0

        # Ensure my_eval_function is always initialized
        if special_static_eval_fn is not None:
            self.my_eval_function = special_static_eval_fn # For autograding
        else:
            self.my_eval_function = self.static_eval  

        zobrist_hash = None
        
        if use_zobrist_hashing:
            zobrist_hash = self.compute_zobrist_hash(current_state)
            self.zobrist_read_attempts_this_turn += 1
            if zobrist_hash in self.transposition_table:
                self.zobrist_hits_this_turn += 1
                best_move, best_score, depth = self.transposition_table[zobrist_hash]
                return [[best_move, self.apply_move(current_state, best_move)],
                        f"Using cached evaluation: {best_score} (depth={depth})"]

        # Now call minimax
        best_score, best_move, resulting_state = self.minimax_wrapper(
            current_state, 
            depth=max_ply,
            use_alpha_beta=use_alpha_beta,
            use_zobrist_hashing=use_zobrist_hashing
        )

        if zobrist_hash is not None:
            self.transposition_table[zobrist_hash] = (best_move, best_score, max_ply)
            self.zobrist_writes_this_turn += 1

        # Construct a remark about the search process:
        new_remark = (f"My {'alpha-beta' if use_alpha_beta else 'minimax'} search "
                    f"expanded {self.num_nodes_expanded_this_turn} nodes and "
                    f"performed {self.num_static_evals_this_turn} static evaluations "
                    f"with {self.num_alpha_beta_cutoffs_this_turn} alpha-beta cutoffs. "
                    f"Zobrist: {self.zobrist_hits_this_turn}/{self.zobrist_read_attempts_this_turn} hits, "
                    f"{self.zobrist_writes_this_turn} writes. "
                    f"I choose move {best_move}!")
    

        if self.apis_ok:
            response_text = self.generate_response(
                f"I'm playing {self.current_game_type.short_name}. My last move was {best_move}. "
                f"{new_remark} How would a strategic AI respond to its opponent? Limit your response to seven sentences."
            )
        else:
            response_text = f"{new_remark} Let's see what you do next!"

        # If for some reason we did not find any legal moves, just return "pass":
        if best_move is None:
            return [[None, current_state], "I cannot move!"]

        # Apply the chosen move to get the new state:
        new_state = self.apply_move(current_state, best_move)
        return [[best_move, resulting_state], response_text]

    def minimax_wrapper(self, state, depth, use_alpha_beta=True, use_zobrist_hashing=False):
        """
        Wrapper that calls the actual minimax or alpha-beta function
        and returns (score, best_move, best_state).
        """
        if use_alpha_beta:
            alpha = float('-inf')
            beta = float('inf')
            score, move, new_state = self.alphabeta_search(state, depth, alpha, beta, use_zobrist_hashing)
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

    def alphabeta_search(self, state, depth, alpha, beta, use_zobrist_hashing):
        """
        Driver for alpha-beta search from the root.
        Returns (value, best_move, best_state).
        """

        zobrist_hash = None
        if use_zobrist_hashing:
            zobrist_hash = self.compute_zobrist_hash(state)
            self.zobrist_read_attempts_this_turn += 1

            if zobrist_hash in self.transposition_table:
                self.zobrist_hits_this_turn += 1
                cached_value, cached_move, cached_depth = self.transposition_table[zobrist_hash]
                if cached_move is not None:
                    best_state = self.apply_move(state, cached_move)
                else:
                    best_state = None
                return (cached_value, cached_move, best_state)
            
        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1
        who = state.whose_move

        # If no moves or depth=0 => evaluate & return
        if not moves or depth == 0:
            val = self.my_eval_function(state, self.current_game_type)
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1

            if use_zobrist_hashing and zobrist_hash is not None:
                self.transposition_table[zobrist_hash] = (val, None, depth)
                self.zobrist_writes_this_turn += 1
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

        # storing scores for future references
        if use_zobrist_hashing and zobrist_hash is not None:
            self.transposition_table[zobrist_hash] = (value, best_move, depth)
            self.zobrist_writes_this_turn += 1

        return (value, best_move, best_state)

    def alphabeta_search_next(self, state, depth, alpha, beta):
        """
        The recursive inner part of alpha-beta search.
        Returns (value, best_move, best_state).
        """
        zobrist_hash = None
        if self.transposition_table is not None:
            zobrist_hash = self.compute_zobrist_hash(state)
            self.zobrist_read_attempts_this_turn += 1
            if zobrist_hash in self.transposition_table:
                self.zobrist_hits_this_turn += 1
                return self.transposition_table[zobrist_hash]
            
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
            # storing scores at the leaf level
            if zobrist_hash is not None:
                self.transposition_table[zobrist_hash] = (val, None, depth)
                self.zobrist_writes_this_turn += 1

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
        if zobrist_hash is not None:
            self.transposition_table[zobrist_hash] = (value, best_move, depth)
            self.zobrist_writes_this_turn += 1

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
        new_state = copy.deepcopy(state)
        # new_state = State(old=state)  # Correctly passing the current state as 'old'
        
        row, col = move
        new_state.board[row][col] = state.whose_move  # Apply the move
        
        # Switch whose move it is
        new_state.change_turn()
        
        return new_state
    
    def static_eval(self, state, game_type=None):
        """
        Evaluates the current state and returns the score which a high score is better for
        X and vice versa
        """
        board = state.board
        k = game_type.k if game_type else 3  
        nR, nC = len(board), len(board[0])
        score = 0

        for r in range(nR):
            for c in range(nC):
                if board[r][c] != '-':
                    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:  
                        x_count, o_count, space_count, open_ends, forbidden_count = self.count_consecutive(r,
                                                                                 c, dr, dc, board, k, nR, nC)

                        #already reach K
                        if x_count == k:
                            return 10000  
                        if o_count == k:
                            return -10000  

                        #K-1
                        if x_count == k-1 and space_count > 0:
                            if open_ends == 2:  # both ends open
                                score += 2000
                            elif open_ends == 1:  # one end open
                                score += 1000
                            else:  # both ends closed
                                score += 500

                        if o_count == k-1 and space_count > 0:
                            if open_ends == 2:
                                score -= 2000
                            elif open_ends == 1:
                                score -= 1000
                            else:
                                score -= 500

                        # other consecutives
                        if x_count > 0 and o_count == 0:  
                            score += (10 ** x_count) * (space_count + 1)
                        if o_count > 0 and x_count == 0:  
                            score -= (10 ** o_count) * (space_count + 1)

                        # Forbidden Squares
                        if forbidden_count > 0:
                            score -= forbidden_count * 200

        return score

    
    def count_consecutive(self, r, c, dr, dc, board, k, nR, nC):
        """
        From (r,c), travel in the direction of (dr, dc) to find consecutive
        Xs and Os, as well as checking if the ends of these consecutive Xs and
        Os have opponents on their two ends
        """
        x_count, o_count, space_count, forbidden_count = 0, 0, 0, 0
        open_ends = 0  # 0 = both ends closed, 1 = one end closed, 2 = both ends open

        for i in range(k):  
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < nR and 0 <= nc < nC:
                if board[nr][nc] == 'X':
                    x_count += 1
                elif board[nr][nc] == 'O':
                    o_count += 1
                elif board[nr][nc] == '-':  # Forbidden Square
                    forbidden_count += 1
                else:
                    space_count += 1  

        # check left side open
        left_r, left_c = r - dr, c - dc
        if 0 <= left_r < nR and 0 <= left_c < nC:
            if board[left_r][left_c] == ' ':
                open_ends += 1
            elif board[left_r][left_c] == '-':
                forbidden_count += 1

        # check right side open
        right_r, right_c = r + k * dr, c + k * dc
        if 0 <= right_r < nR and 0 <= right_c < nC:
            if board[right_r][right_c] == ' ':
                open_ends += 1
            elif board[right_r][right_c] == '-':
                forbidden_count += 1

        return x_count, o_count, space_count, open_ends, forbidden_count
    
    def init_zobrist_table(self):
        if self.current_game_type is None:
            return
        
        nR, nC = self.current_game_type.n, self.current_game_type.m
        pieces = ['X', 'O', ' ', '-']

        self.zobrist_table = {}
        for r in range(nR):
            for c in range(nC):
                self.zobrist_table[(r, c)] = {p: random.getrandbits(64) for p in pieces}

    
    def compute_zobrist_hash(self, state):
        hash_value = 0
        board = state.board
        nR, nC = len(board), len(board[0])

        for r in range(nR):
            for c in range(nC):
                piece = board[r][c]
                if piece in self.zobrist_table[(r, c)]:
                    hash_value ^= self.zobrist_table[(r, c)][piece]
                else:
                    print(f"⚠️ WARNING: Zobrist table missing entry for ({r}, {c})")  

        return hash_value
    
    def generate_response(self, prompt):
        if not self.apis_ok:
            return "I'm afraid my AI capabilities are limited at the moment."

        try:
            from google import genai

            client = genai.Client(api_key="api key")

            completion = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
                )
            return completion.text

        except Exception as e:
            return f"Oops, I encountered an issue: {str(e)}"

    