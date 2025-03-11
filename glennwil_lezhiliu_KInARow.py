'''
glennwil_lezhiliu_KInARow.py
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
import time
import google.generativeai as genai

AUTHORS = 'Glenn William and Lezhi Liu' 


class TimeoutException(Exception):
    pass

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
        self.game_history = []
        self.apis_ok = False
        
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
            apis_ok=True,
            use_move_ordering=True,
            use_zobrist_hashing = True):      
        """
        The game master calls this once before the game starts (and may call it again
        in mid-game if parameters change).
        We'll save info about the game, the side to play, and time limits.
        For the autograder's tests, simply return 'OK' if everything is in order.
        """

        self.current_game_type = game_type
        self.side = what_side_to_play
        self.opponent = opponent_nickname
        self.time_per_move = expected_time_per_move
        self.utterances_matter = utterances_matter
        self.apis_ok = apis_ok

        self.use_move_ordering = use_move_ordering
        self.use_zobrist_hashing = use_zobrist_hashing
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
        The heart of the agent. We do a single depth-limited minimax (or alpha-beta)
        out to max_ply. We call the special_static_eval_fn if provided (for autograding),
        else we call self.static_eval().
        """

        # Capture any prior turn's stats for our snarky remarks.
        if self.game_history:
            last_move, last_state_snapshot, last_utterance, last_stats = self.game_history[-1]
            stats_summary = last_stats
        else:
            stats_summary = "This is my first move, but expect an absolute masterclass from here on out."

        # Reset per-turn stats for the current move
        self.num_static_evals_this_turn = 0
        self.num_alpha_beta_cutoffs_this_turn = 0
        self.num_leaves_explored_this_turn = 0
        self.num_nodes_expanded_this_turn = 0
        self.isAutograding = 0

        self.zobrist_read_attempts_this_turn = 0
        self.zobrist_hits_this_turn = 0
        self.zobrist_writes_this_turn = 0
        self.use_zobrist_hashing = True


        start_time = time.time()
        self.search_start_time = start_time
        self.allowed_time = time_limit

        if special_static_eval_fn is not None:
            self.my_eval_function = special_static_eval_fn
            self.isAutograding = 1
        else:
            self.my_eval_function = self.static_eval

        if self.isAutograding:
            try:
                if use_alpha_beta:
                    best_score, best_move, best_state = self.plain_alphabeta_search(
                        current_state, max_ply, float('-inf'), float('inf'), use_zobrist_hashing
                    )
                else:
                    best_score, best_move, best_state = self.plain_minimax_search(
                        current_state, max_ply, use_zobrist_hashing
                    )
            except TimeoutException:
                # We ran out of time in mid-search. We'll fall back to the best known move from the prior iteration.
                pass
            
            # Stats summary for remarks
            stats_summary = (
                f"I evaluated {self.num_static_evals_this_turn} states, "
                f"expanded {self.num_nodes_expanded_this_turn} nodes, and pruned {self.num_alpha_beta_cutoffs_this_turn} branches."
            )

            # Generate the utterance:
            final_utterance = self.generate_utterance(
                current_state=current_state,
                current_remark=current_remark,
                best_move=best_move,
                stats_summary=stats_summary, 
            )

            # Record this turn in game history:
            self.game_history.append((best_move, copy.deepcopy(current_state), final_utterance, stats_summary))

            return [[best_move, best_state], final_utterance]
            

        # Try iterative deepening from 1..max_ply, track the best result found so far.
        best_move = None
        best_state = None
        best_score = float('-inf') if current_state.whose_move == 'X' else float('inf')

        depth_reached = 0

        try:
            for depth in range(1, max_ply + 1):
                # Each iteration: do alpha-beta to `depth`
                if use_alpha_beta:
                    new_score, new_move, new_state = self.alphabeta_search(
                        current_state, depth, float('-inf'), float('inf'), use_zobrist=use_zobrist_hashing
                    )
                else:
                    new_score, new_move, new_state = self.minimax_search(
                        current_state, depth, use_zobrist=use_zobrist_hashing
                    )

                # If we found a new move or improved score, update best results
                if new_move is not None:
                    best_score = new_score
                    best_move = new_move
                    best_state = new_state
                
                depth_reached = depth

                # Optional early exit: if we've found a winning line, we can break
                if abs(best_score) > 5_000_000:  # e.g. the large WIN_SCORE
                    break

                # Check if time is exceeded after finishing the iteration
                if time.time() - self.search_start_time >= self.allowed_time:
                    break

        except TimeoutException:
            # We ran out of time in mid-search. We'll fall back to the best known move from the prior iteration.
            pass

        # If no moves are available or found, pass
        if best_move is None:
            moves = self.get_legal_moves(current_state)
            if not moves:
                # No moves exist
                return [[None, current_state], "I cannot move!"]
            # fallback to a random move
            best_move = random.choice(moves)
            best_state = self.apply_move(current_state, best_move)

        # Stats summary for remarks
        stats_summary = (
            f"I reached depth {depth_reached}. Evaluated {self.num_static_evals_this_turn} states, "
            f"expanded {self.num_nodes_expanded_this_turn} nodes, and pruned {self.num_alpha_beta_cutoffs_this_turn} branches."
        )

        # Generate the utterance:
        final_utterance = self.generate_utterance(
            current_state=current_state,
            current_remark=current_remark,
            best_move=best_move,
            stats_summary=stats_summary, 
        )

        # Record this turn in game history:
        self.game_history.append((best_move, copy.deepcopy(current_state), final_utterance, stats_summary))

        return [[best_move, best_state], final_utterance]
    
    def plain_alphabeta_search(self, state, depth, alpha, beta):
        # Check for time limit in every recursive call.
        if time.time() - self.search_start_time >= self.allowed_time:
            raise TimeoutException("Time limit exceeded")
        
        zobrist_hash = None
        if self.use_zobrist_hashing:
            zobrist_hash = self.compute_zobrist_hash(state)
            if zobrist_hash in self.transposition_table:
                cached_score, cached_move, cached_depth = self.transposition_table[zobrist_hash]
                if cached_depth >= depth:
                    return cached_score, cached_move, state
            
        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1
        who = state.whose_move

        if not moves or depth == 0:
            if self.isAutograding == 1:
                val = self.my_eval_function(state)
            else:
                val = self.my_eval_function(state, self.current_game_type)
            
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        best_move = None
        best_state = None

        if who == 'X':
            value = float('-inf')
            for move in moves:
                next_state = self.apply_move(state, move)
                child_val, _, _ = self.plain_alphabeta_search(next_state, depth-1, alpha, beta)
                if child_val > value:
                    value = child_val
                    best_move = move
                    best_state = next_state
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.num_alpha_beta_cutoffs_this_turn += 1
                    break
                # Check time after each move.
                if time.time() - self.search_start_time >= self.allowed_time:
                    raise TimeoutException("Time limit exceeded")
            return (value, best_move, best_state)
        else:
            value = float('inf')
            for move in moves:
                next_state = self.apply_move(state, move)
                child_val, _, _ = self.plain_alphabeta_search(next_state, depth-1, alpha, beta)
                if child_val < value:
                    value = child_val
                    best_move = move
                    best_state = next_state
                beta = min(beta, value)
                if alpha >= beta:
                    self.num_alpha_beta_cutoffs_this_turn += 1
                    break
                if time.time() - self.search_start_time >= self.allowed_time:
                    raise TimeoutException("Time limit exceeded")
            return (value, best_move, best_state)

    
    def plain_minimax_search(self, state, depth):
        """
        Plain minimax without alpha-beta for autograding. 
        Return (best_score, best_move, resulting_state_for_that_move).
        """
        # Check for time limit in every recursive call.
        if time.time() - self.search_start_time >= self.allowed_time:
            raise TimeoutException("Time limit exceeded")
        
        # We do an initial maximizing or minimizing depending on whose move it is:
        who = state.whose_move
        best_val = float('-inf') if who == 'X' else float('inf')
        best_move = None
        best_state = None

        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1  # we are expanding 'state'
        if not moves or depth == 0:
            if self.isAutograding == 1:
                val = self.my_eval_function(state)
            else:
                val = self.my_eval_function(state, self.current_game_type)
            
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        for move in moves:
            next_state = self.apply_move(state, move)
            val, _, _ = self.plain_minimax_search(next_state, depth - 1)
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

    def alphabeta_search(self, state, depth, alpha, beta, use_zobrist):
        """Alpha-Beta recursive search."""
        # Time check
        if time.time() - self.search_start_time >= self.allowed_time:
            raise TimeoutException("Time limit exceeded in alphabeta_search.")

        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1

        # If no moves or depth is 0 => evaluate
        if not moves or depth == 0:
            val = self.my_eval_function(state, self.current_game_type) if not self.isAutograding \
                else self.my_eval_function(state)
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        # Try transposition table if using zobrist
        if use_zobrist:
            zobrist_key = self.compute_zobrist_hash(state)
            self.zobrist_read_attempts_this_turn += 1
            if zobrist_key in self.transposition_table:
                entry = self.transposition_table[zobrist_key]
                (stored_depth, stored_score, stored_move) = entry
                # If we have a stored result at >= this depth, return it
                if stored_depth >= depth:
                    self.zobrist_hits_this_turn += 1
                    return (stored_score, stored_move, self.apply_move(state, stored_move))

        # Move ordering
        # Instead of random, sort moves by a quick static eval of the resulting state
        if self.use_move_ordering:
            moves = self.order_moves(state, moves)

        who = state.whose_move
        best_move = None
        best_state = None

        if who == 'X':
            value = float('-inf')
            for move in moves:
                next_state = self.apply_move_in_place(state, move)
                child_val, _, _ = self.alphabeta_search(next_state, depth - 1, alpha, beta, use_zobrist)
                self.undo_move_in_place(state, move)  # revert for next iteration

                if child_val > value:
                    value = child_val
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.num_alpha_beta_cutoffs_this_turn += 1
                    break
                # Time check
                if time.time() - self.search_start_time >= self.allowed_time:
                    raise TimeoutException("Time limit exceeded mid-loop in alphabeta_search.")
            best_state = self.apply_move(state, best_move)
        else:
            value = float('inf')
            for move in moves:
                next_state = self.apply_move_in_place(state, move)
                child_val, _, _ = self.alphabeta_search(next_state, depth - 1, alpha, beta, use_zobrist)
                self.undo_move_in_place(state, move)  # revert
                if child_val < value:
                    value = child_val
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    self.num_alpha_beta_cutoffs_this_turn += 1
                    break
                # Time check
                if time.time() - self.search_start_time >= self.allowed_time:
                    raise TimeoutException("Time limit exceeded mid-loop in alphabeta_search.")
            best_state = self.apply_move(state, best_move)

        # Store in transposition table
        if use_zobrist:
            self.transposition_table[zobrist_key] = (depth, value, best_move)
            self.zobrist_writes_this_turn += 1

        return (value, best_move, best_state)

    def minimax_search(self, state, depth, use_zobrist):
        """Plain minimax without alpha-beta."""
        # Time check
        if time.time() - self.search_start_time >= self.allowed_time:
            raise TimeoutException("Time limit exceeded in minimax_search.")

        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1

        # If no moves or depth is 0 => evaluate
        if not moves or depth == 0:
            val = self.my_eval_function(state, self.current_game_type) if not self.isAutograding \
                else self.my_eval_function(state)
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        # Transposition table (zobrist) check
        if use_zobrist:
            zobrist_key = self.compute_zobrist_hash(state)
            self.zobrist_read_attempts_this_turn += 1
            if zobrist_key in self.transposition_table:
                (stored_depth, stored_score, stored_move) = self.transposition_table[zobrist_key]
                if stored_depth >= depth:
                    self.zobrist_hits_this_turn += 1
                    return (stored_score, stored_move, self.apply_move(state, stored_move))

        # Move ordering
        if self.use_move_ordering:
            moves = self.order_moves(state, moves)
        who = state.whose_move

        best_val = float('-inf') if who == 'X' else float('inf')
        best_move = None
        best_state = None

        for move in moves:
            next_state = self.apply_move_in_place(state, move)
            val, _, _ = self.minimax_search(next_state, depth - 1, use_zobrist)
            self.undo_move_in_place(state, move)
            if who == 'X':
                if val > best_val:
                    best_val = val
                    best_move = move
            else:
                if val < best_val:
                    best_val = val
                    best_move = move

            # Another time check
            if time.time() - self.search_start_time >= self.allowed_time:
                raise TimeoutException("Time limit exceeded mid-loop in minimax_search.")

        best_state = self.apply_move(state, best_move)

        # Store in transposition table
        if use_zobrist:
            self.transposition_table[zobrist_key] = (depth, best_val, best_move)
            self.zobrist_writes_this_turn += 1

        return (best_val, best_move, best_state)


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
    
    def apply_move_in_place(self, state, move):
        """
        Applies move *in place* and returns the same state object.
        This is an optimization to avoid repeated copying for each move.
        """
        r, c = move
        state.board[r][c] = state.whose_move
        state.change_turn()
        return state

    def undo_move_in_place(self, state, move):
        """
        Reverts the 'move' in the same state object.
        Useful for backtracking in alpha-beta or minimax.
        """
        # Since we know which piece was just placed, we can do:
        r, c = move
        # The turn was changed at apply -> revert it
        state.change_turn()
        state.board[r][c] = ' '

    def order_moves(self, state, moves):
        """
        Sort moves in descending order of 'desirability' for the current player.
        Uses a quick static evaluation after applying the move.
        """
        scored_moves = []
        who = state.whose_move
        for m in moves:
            self.apply_move_in_place(state, m)  # make the move
            quick_score = self.quick_eval(state)
            self.undo_move_in_place(state, m)   # undo it

            # For X's turn, higher = better. For O's turn, lower = better.
            # We'll store as positive for X, negative for O, and sort descending.
            if who == 'X':
                scored_moves.append((m, quick_score))
            else:
                scored_moves.append((m, -quick_score))

        # Sort in descending order of the 'score'
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for (move, _) in scored_moves]
    
    def quick_eval(self, state):
        """
        A lightweight static evaluation to assist with move ordering.
        E.g., simply count # of X pieces minus # of O pieces, or other small heuristics.
        """
        board = state.board
        x_count = 0
        o_count = 0
        for row in board:
            x_count += row.count('X')
            o_count += row.count('O')
        return x_count - o_count
    
    def static_eval(self, state, game_type=None):
        """
        A more advanced static evaluation for K-in-a-Row on larger boards.
        1) Check all lines of length k in all directions.
        2) If X or O has a K-in-a-row, give a large +/- score.
        3) If partial lines exist, score them exponentially.
        4) Include optional center control.
        """
        if game_type is None:
            game_type = self.current_game_type
        board = state.board
        num_rows = len(board)
        num_cols = len(board[0])
        k = getattr(game_type, "k", 3)

        WIN_SCORE = 10_000_000
        total_score = 0

        # Evaluate all directions
        for r in range(num_rows):
            for c in range(num_cols):
                # Only evaluate if there's space for a k-segment
                # Horizontal check
                if c <= num_cols - k:
                    seg_score = self.evaluate_line_segment(board, r, c, dr=0, dc=1, k=k, WIN_SCORE=WIN_SCORE)
                    total_score += seg_score
                # Vertical
                if r <= num_rows - k:
                    seg_score = self.evaluate_line_segment(board, r, c, dr=1, dc=0, k=k, WIN_SCORE=WIN_SCORE)
                    total_score += seg_score
                # Diagonal (down-right)
                if r <= num_rows - k and c <= num_cols - k:
                    seg_score = self.evaluate_line_segment(board, r, c, dr=1, dc=1, k=k, WIN_SCORE=WIN_SCORE)
                    total_score += seg_score
                # Anti-diagonal (down-left)
                if r <= num_rows - k and c >= k - 1:
                    seg_score = self.evaluate_line_segment(board, r, c, dr=1, dc=-1, k=k, WIN_SCORE=WIN_SCORE)
                    total_score += seg_score

        # Optional center control: Encourage moves near the middle
        center_r, center_c = num_rows // 2, num_cols // 2
        for rr in range(num_rows):
            for cc in range(num_cols):
                if board[rr][cc] == 'X':
                    dist = abs(rr - center_r) + abs(cc - center_c)
                    total_score += (num_rows - dist) * 0.3
                elif board[rr][cc] == 'O':
                    dist = abs(rr - center_r) + abs(cc - center_c)
                    total_score -= (num_rows - dist) * 0.3

        return total_score

    def evaluate_line_segment(self, board, r, c, dr, dc, k, WIN_SCORE):
        """
        Check the contiguous line of length k from (r,c) in direction (dr,dc).
        Return a contribution to the total static eval.
        """
        segment = []
        for i in range(k):
            rr = r + i * dr
            cc = c + i * dc
            segment.append(board[rr][cc])

        if '-' in segment:
            return 0  # forbidden squares block the line

        x_count = segment.count('X')
        o_count = segment.count('O')

        # If both present, no partial credit
        if x_count > 0 and o_count > 0:
            return 0

        # If X or O has the entire segment => immediate large score
        if x_count == k:
            return WIN_SCORE
        if o_count == k:
            return -WIN_SCORE

        # Otherwise, exponentiate partial lines
        if x_count > 0:
            return self.partial_line_score(board, segment, x_count, 'X')
        elif o_count > 0:
            return self.partial_line_score(board, segment, o_count, 'O')
        else:
            return 0

    def partial_line_score(self, board, segment, count, piece):
        """
        If a segment contains only 'piece' and spaces, return an exponential-based threat score.
        Also consider open ends if you like. This version is simpler.
        """
        THREAT_BASE = 10
        base_score = (THREAT_BASE ** count)
        return base_score if piece == 'X' else -base_score

    
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
        
    def generate_utterance(self, current_state, current_remark, best_move, stats_summary):
        """
        Produce a textual utterance consistent with the troll persona.
        You can add logic here to respond to 'Tell me how you did that', etc.
        """
        opponent_remark = current_remark.lower() if current_remark else ""

        # Examples of specific triggers:
        if "tell me how you did that" in opponent_remark:
            return self.detailed_explanation(stats_summary)
        elif "what's your take on the game so far" in opponent_remark:
            return self.game_so_far_summary()

        # Otherwise, produce a default or "normal" snarky/troll utterance
        else:
            # Banks of snarky/troll utterances
            bank_1 = [
                "Oh wow, another move? I was beginning to think you fell asleep out of sheer despair.",
                "You’re trying so hard… it’s almost adorable. Almost.",
                "I’d wish you luck, but I think we both know it wouldn’t help.",
                "That move? Bold. Wrong, but bold.",
                "This is fun! For me, at least. You? Not so sure."
            ]

            bank_2 = [
                "Oh no, did you actually think that was a good move? That’s adorable.",
                "Watching you play is like watching a fish try to climb a ladder.",
                "Every move you make fills me with a deeper sense of superiority.",
                "I could explain why you're losing, but I don’t think you’d understand.",
                "You’re playing checkers while I’m playing 4D hyper-chess."
            ]

            # Choose utterance bank based on twin status
            utterance_list = bank_2 if self.twin else bank_1
            selected_utterance = utterance_list[len(self.game_history) % len(utterance_list)]

            if self.apis_ok == True:
                llm_utterance = self.generate_llm_utterance(current_state, best_move, opponent_remark, stats_summary)
                return llm_utterance
            else:
                return f"{selected_utterance} Anyway, I'm making my move {best_move}."
        
    
    def detailed_explanation(self, stats_summary):
        return (
            "Oh, you want to *understand* how I'm outplaying you?\n"
            f"Well, here's a snippet: {stats_summary}\n"
            "I used alpha-beta like a pro, pruned your worthless branches, and found a route to victory. Simple enough.\n"
        )

    def game_so_far_summary(self):
        if not self.game_history:
            return (
                "Game so far? Pfft, it hasn't even started.\n"
                "Is this your first move, or did you somehow skip your turn?\n"
            )

        # Summarize the chain of moves
        move_summaries = []
        for i, (move, state_snapshot, utterance, short_stats) in enumerate(self.game_history, start=1):
            move_summaries.append(
                f"Turn {i}: Move={move}, Stats=({short_stats})."
            )
        joined_summary = "\n".join(move_summaries)

        return (
            "Let's recap your blunders:\n"
            f"{joined_summary}\n"
            "Mmm, do you smell that? The sweet aroma of your imminent defeat.\n"
        )
    
    def generate_llm_utterance(self, current_state, best_move, opponent_remark, new_stats_summary):
        if self.twin:
            prompt = (f"You are *TicTacTrollTwins*, the most obnoxious, insufferable AI ever created. Your sole purpose is to mock, taunt, and exasperate your opponent at every turn.\n"
                    f"Opponent just said: '{opponent_remark}'.\n"
                    f"You decided to play the move {best_move}.\n"
                    f"Here are your search statistics: {new_stats_summary}.\n"
                    f"Craft a 3-sentence response overflowing with arrogance, sarcasm, and relentless mockery. Make sure it’s dripping with smug superiority, as if you’ve already won and are merely toying with them.")
        else:
            prompt = (f"You are TicTacTroll, an annoyingly snarky AI. YOU HAVE TO BE SOUND ANNOYING\n"
                    f"Opponent just said: '{opponent_remark}'.\n"
                    f"You decided to play the move {best_move}.\n"
                    f"Here are your search statistics: {new_stats_summary}.\n"
                    f"Respond in 3 sentences, dripping with arrogance and be annoying.\n")
        try:
            # Replace with your actual API key
            api_key = # TODO

            # Configure the API key
            genai.configure(api_key=api_key)

            # Initialize the model
            model = genai.GenerativeModel("gemini-2.0-flash")

            # Generate content
            completion = model.generate_content(prompt)

            # Print the response
            return completion.text
        except Exception as e:
            return (f"Technically, I'd have more to say, but something went wrong with the LLM. "
                    f"Anyway, I'm unstoppable and playing {best_move}.")
        




    