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
            apis_ok=True):      
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

        # Capture statistics from the previous turn before resetting
        if self.game_history:
            last_move, last_state_snapshot, last_utterance, last_stats = self.game_history[-1]
            stats_summary = last_stats  # Use stats from the last move
        else:
            stats_summary = "This is my first move, but expect an absolute masterclass from here on out."

        # Reset per-turn stats for the current move
        self.num_static_evals_this_turn = 0
        self.num_alpha_beta_cutoffs_this_turn = 0
        self.num_leaves_explored_this_turn = 0
        self.num_nodes_expanded_this_turn = 0
        self.zobrist_writes_this_turn = 0
        self.zobrist_read_attempts_this_turn = 0
        self.zobrist_hits_this_turn = 0

        start_time = time.time()
        best_move = None
        best_state = None
        current_depth = 1

        # Store timing info for recursive functions
        self.search_start_time = start_time
        self.allowed_time = time_limit

        # Ensure my_eval_function is always initialized
        if special_static_eval_fn is not None:
            self.my_eval_function = special_static_eval_fn  # For autograding
        else:
            self.my_eval_function = self.static_eval  

        try:
            while current_depth <= max_ply:
                if time.time() - start_time >= time_limit:
                    raise TimeoutException()
                if use_alpha_beta:
                    algorithm = "alpha-beta"
                    score, move, state = self.alphabeta_search(current_state, current_depth, float('-inf'), float('inf'))
                else:
                    algorithm = "minimax"
                    score, move, state = self.minimax_search(current_state, current_depth)
                best_score = score
                best_move, best_state = move, state
                current_depth += 1
        except TimeoutException:
            print("Time limit reached, returning best move from last complete search depth.")

        if best_move is None:
            legal_moves = self.get_legal_moves(current_state)
            best_move = legal_moves[0] if legal_moves else None
            best_state = self.apply_move(current_state, best_move) if best_move is not None else current_state

        # Update game history with current move
        new_stats_summary = (
            f"I evaluated {self.num_static_evals_this_turn} states, "
            f"expanded {self.num_nodes_expanded_this_turn} nodes, "
            f"and pruned {self.num_alpha_beta_cutoffs_this_turn} branches this turn."
        )

        # Generate utterance using last move’s stats
        final_utterance = self.generate_utterance(
            current_state=current_state,
            current_remark=current_remark,
            best_move=best_move,
            stats_summary=stats_summary, 
            algorithm=algorithm,
            new_stats_summary=new_stats_summary
        )

        self.game_history.append((best_move, copy.deepcopy(current_state), final_utterance, new_stats_summary))

        return [[best_move, best_state], final_utterance]
        
        # if use_zobrist_hashing:
        #     zobrist_hash = self.compute_zobrist_hash(current_state)
        #     self.zobrist_read_attempts_this_turn += 1
        #     if zobrist_hash in self.transposition_table:
        #         self.zobrist_hits_this_turn += 1
        #         best_move, best_score, depth = self.transposition_table[zobrist_hash]
        #         return [[best_move, self.apply_move(current_state, best_move)],
        #                 f"Using cached evaluation: {best_score} (depth={depth})"]

        # # Now call minimax
        # best_score, best_move, resulting_state = self.minimax_wrapper(
        #     current_state, 
        #     depth=max_ply,
        #     use_alpha_beta=use_alpha_beta,
        #     use_zobrist_hashing=use_zobrist_hashing
        # )


        # if zobrist_hash is not None:
        #     self.transposition_table[zobrist_hash] = (best_move, best_score, max_ply)
        #     self.zobrist_writes_this_turn += 1

        # # Construct a remark about the search process:
        # new_remark = (f"My {'alpha-beta' if use_alpha_beta else 'minimax'} search "
        #             f"expanded {self.num_nodes_expanded_this_turn} nodes and "
        #             f"performed {self.num_static_evals_this_turn} static evaluations "
        #             f"with {self.num_alpha_beta_cutoffs_this_turn} alpha-beta cutoffs. "
        #             f"Zobrist: {self.zobrist_hits_this_turn}/{self.zobrist_read_attempts_this_turn} hits, "
        #             f"{self.zobrist_writes_this_turn} writes. "
        #             f"I choose move {best_move}!")

        # utterance = self.generate_utterance(current_state, current_remark, best_move, new_remark)

        # # if self.apis_ok:
        # #     response_text = self.generate_response(
        # #         f"I'm playing {self.current_game_type.short_name}. My last move was {best_move}. "
        # #         f"{new_remark} How would a strategic AI respond to its opponent? Limit your response to seven sentences."
        # #     )
        # # else:
        # #     response_text = f"{new_remark} Let's see what you do next!"
            

        # # If for some reason we did not find any legal moves, just return "pass":
        # if best_move is None:
        #     return [[None, current_state], "I cannot move!"]
        
        # self.game_history.append((best_move, current_state.whose_move))

        # # Apply the chosen move to get the new state:
        # new_state = self.apply_move(current_state, best_move)
        # return [[best_move, resulting_state], utterance]

    def alphabeta_search(self, state, depth, alpha, beta):
        # Check for time limit in every recursive call.
        if time.time() - self.search_start_time >= self.allowed_time:
            raise TimeoutException("Time limit exceeded")
            
        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1
        who = state.whose_move

        if not moves or depth == 0:
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
                child_val, _, _ = self.alphabeta_search(next_state, depth-1, alpha, beta)
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
                child_val, _, _ = self.alphabeta_search(next_state, depth-1, alpha, beta)
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

    
    def minimax_search(self, state, depth):
        """
        Plain minimax without alpha-beta. 
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
            val = self.my_eval_function(state, self.current_game_type)
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        for move in moves:
            next_state = self.apply_move(state, move)
            val, _, _ = self.minimax_search(next_state, depth - 1)
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

    def minimax_search_with_time(self, state, depth):
        """
        Plain minimax search with a time limit check.
        Returns (best_score, best_move, resulting_state).
        """
        # Check time at the start of this call.
        if time.time() - self.search_start_time >= self.allowed_time:
            raise TimeoutException("Time limit exceeded during minimax search.")

        moves = self.get_legal_moves(state)
        self.num_nodes_expanded_this_turn += 1
        who = state.whose_move

        # Base case: terminal state or maximum depth reached.
        if not moves or depth == 0:
            val = self.my_eval_function(state, self.current_game_type)
            self.num_static_evals_this_turn += 1
            self.num_leaves_explored_this_turn += 1
            return (val, None, None)

        best_move = None
        best_state = None

        if who == 'X':
            best_val = float('-inf')
            for move in moves:
                next_state = self.apply_move(state, move)
                child_val, _, _ = self.minimax_search_with_time(next_state, depth - 1)
                if child_val > best_val:
                    best_val = child_val
                    best_move = move
                    best_state = next_state
                # Check time after each move.
                if time.time() - self.search_start_time >= self.allowed_time:
                    raise TimeoutException("Time limit exceeded during minimax search (maximizing).")
            return (best_val, best_move, best_state)
        else:
            best_val = float('inf')
            for move in moves:
                next_state = self.apply_move(state, move)
                child_val, _, _ = self.minimax_search_with_time(next_state, depth - 1)
                if child_val < best_val:
                    best_val = child_val
                    best_move = move
                    best_state = next_state
                if time.time() - self.search_start_time >= self.allowed_time:
                    raise TimeoutException("Time limit exceeded during minimax search (minimizing).")
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
    
    def static_eval(self, state, game_type=None):
        """
        Improved static evaluation for K-in-a-Row games.
        
        This function examines every contiguous segment of length k in all
        directions. Each segment is scored based on:
        - k markers:        win/loss (+/-100)
        - k-1 markers:      threat (+/-10)
        - k-2 markers:      minor advantage (+/-1)
        In addition, if a segment’s open ends are available (i.e., the adjacent 
        cells in the same direction are empty), a bonus is applied.
        
        Finally, a positional weighting bonus is added, rewarding pieces closer 
        to the board’s center.
        """
        board = state.board
        num_rows = len(board)
        num_cols = len(board[0])
        # Use the game type's winning condition if available; default to 3.
        k = getattr(game_type, "k", 3)
        score = 0

        # Evaluate horizontal segments.
        for r in range(num_rows):
            for c in range(num_cols - k + 1):
                score += self.evaluate_segment_with_open_ends(board, r, c, 0, 1, k, num_rows, num_cols)

        # Evaluate vertical segments.
        for c in range(num_cols):
            for r in range(num_rows - k + 1):
                score += self.evaluate_segment_with_open_ends(board, r, c, 1, 0, k, num_rows, num_cols)

        # Evaluate diagonals (top-left to bottom-right).
        for r in range(num_rows - k + 1):
            for c in range(num_cols - k + 1):
                score += self.evaluate_segment_with_open_ends(board, r, c, 1, 1, k, num_rows, num_cols)

        # Evaluate anti-diagonals (top-right to bottom-left).
        for r in range(num_rows - k + 1):
            for c in range(k - 1, num_cols):
                score += self.evaluate_segment_with_open_ends(board, r, c, 1, -1, k, num_rows, num_cols)

        # Add positional bonus: cells closer to the center get a bonus.
        center_r, center_c = num_rows // 2, num_cols // 2
        for r in range(num_rows):
            for c in range(num_cols):
                # A simple weight: the closer a cell is to the center, the higher its bonus.
                distance = abs(center_r - r) + abs(center_c - c)
                bonus = (max(num_rows, num_cols) - distance) * 0.1  # adjust factor as needed
                if board[r][c] == 'X':
                    score += bonus
                elif board[r][c] == 'O':
                    score -= bonus

        return score

    def evaluate_segment_with_open_ends(self, board, r, c, dr, dc, k, num_rows, num_cols):
        """
        Evaluate a contiguous segment of length k starting at (r, c) in direction (dr, dc).
        
        The base evaluation (via self.evaluate_segment) returns:
        +100 for k X's, +10 for k-1 X's, +1 for k-2 X's (and similarly negative for O).
        This function checks the cell immediately before and after the segment (if they exist)
        and awards an extra bonus if the segment is "open" on one or both sides.
        """
        segment = [board[r + i * dr][c + i * dc] for i in range(k)]
        seg_score = self.evaluate_segment(segment, k)
        
        # Winning segments need no extra bonus.
        if seg_score in (100, -100):
            return seg_score

        open_bonus = 0
        # Coordinates for the cell before the segment.
        pre_r, pre_c = r - dr, c - dc
        # Coordinates for the cell after the segment.
        post_r, post_c = r + k * dr, c + k * dc

        left_open = (0 <= pre_r < num_rows and 0 <= pre_c < num_cols and board[pre_r][pre_c] == ' ')
        right_open = (0 <= post_r < num_rows and 0 <= post_c < num_cols and board[post_r][post_c] == ' ')

        if left_open and right_open:
            open_bonus = 2
        elif left_open or right_open:
            open_bonus = 1

        # If the segment is favorable for X, add the bonus; if for O, subtract it.
        if seg_score > 0:
            return seg_score + open_bonus
        elif seg_score < 0:
            return seg_score - open_bonus
        return seg_score

    def evaluate_segment(self, segment, k):
        """
        Evaluate a segment of length k.
        
        Returns:
        +100 if the segment has k X's (win for X)
        +10 if it has k-1 X's (threat for X)
        +1 if it has k-2 X's (minor advantage for X)
        Similarly, returns negative values for O.
        
        If the segment contains both X and O or any blocked cell ('-'),
        it is considered contested and returns 0.
        """
        if '-' in segment:
            return 0

        x_count = segment.count('X')
        o_count = segment.count('O')

        if x_count > 0 and o_count > 0:
            return 0

        if x_count > 0:
            if x_count == k:
                return 100
            elif x_count == k - 1:
                return 10
            elif x_count == k - 2:
                return 1

        if o_count > 0:
            if o_count == k:
                return -100
            elif o_count == k - 1:
                return -10
            elif o_count == k - 2:
                return -1

        return 0
    
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
        
    def generate_utterance(self, current_state, current_remark, best_move, stats_summary, algorithm, new_stats_summary):
        """
        Generate a textual utterance in character, responding to special questions
        or producing default commentary.
        :param current_state: The current board state before our move is applied.
        :param current_remark: Opponent's last remark, which may contain special requests.
        :param best_move: The move we are about to make.
        :param stats_summary: A short string summarizing search stats for this turn.
        :return: A string (utterance).
        """
        # Normalize the opponent's remark for easy matching
        opponent_remark_lower = current_remark.lower() if current_remark else ""

        # 1. If the opponent says: "Tell me how you did that"
        if "tell me how you did that" in opponent_remark_lower:
            explanation_utterance = self.generate_detailed_explanation(stats_summary, algorithm)
            return explanation_utterance

        # 2. If the opponent says: "What's your take on the game so far?"
        elif "what's your take on the game so far" in opponent_remark_lower:
            story_utterance = self.generate_game_so_far_summary()
            return story_utterance

        # 3. Otherwise, produce a default or "normal" snarky/troll utterance
        else:
            llm_utterance = self.generate_llm_utterance(current_state, best_move, opponent_remark_lower, stats_summary)
            return llm_utterance
        
    def generate_detailed_explanation(self, stats_summary, algorithm):
        if algorithm == "alpha-beta":
            explanation = (
                "Oh wow, you *really* thought that move would work? Cute.\n"
                "I used alpha-beta pruning, which means I played a million games in my head *before* you even blinked.\n"
                f"Here, have some cold, hard stats from the last turn: {stats_summary} (not that you’d understand them).\n"
                "I blocked your threats, set traps, and made sure every move led to your slow, painful defeat.\n"
                "Meanwhile, you’re just flailing, hoping for a miracle. Spoiler: it’s not coming.\n"
                "My pruning skips useless moves, so I only focus on the *winning* ones. Unlike you.\n"
                "Basically, I saw your failure before you did. And now, you get to live through it.\n"
            )
        elif algorithm == "minimax":
            explanation = (
                "Minimax. Cold. Calculated. Unstoppable.\n"
                "Every move you made? I already predicted it. Every counter you thought was clever? *Pathetic.*\n"
                f"Here, take a look at your impending doom, my stats from last turn: {stats_summary}.\n"
                "I maximize my advantage, minimize your chances, and leave you with *nothing*.\n"
                "You react. I plan. You guess. I *know*.\n"
                "No luck, no hesitation—just pure, merciless domination.\n"
                "Honestly, it’s almost sad. Almost.\n"
            )
        return explanation
    
    def generate_game_so_far_summary(self):
        """
        Builds an obnoxiously smug story about the game so far, referencing self.game_history,
        and throws in an overconfident prediction.
        """
        
        if not self.game_history:
            return (
                "Oh, you want a recap? Cute.\n"
                "Except… there's nothing to recap because you haven’t even made a move.\n"
                "I can’t mock what doesn’t exist! Go on, do *something*, and then maybe I’ll have material to work with.\n"
            )

        # Summarize moves with more detail
        moves_summary = []
        for turn_index, (move, state_snapshot, utterance, new_stats_summary) in enumerate(self.game_history, start=1):
            previous_player = "X" if state_snapshot.whose_move == "O" else "O"  # Whose move it was before the switch
            board_state = '\n'.join([' '.join(row) for row in state_snapshot.board])  # Display board (optional)
            
            moves_summary.append(
                f"Turn {turn_index}: {previous_player} made a move at {move}.\n"
                f"📝 Remark: {utterance if utterance else 'Silence speaks volumes...'}\n"
                f"📊 Stats: {new_stats_summary}\n"
                f"📌 Board after the move:\n{board_state}\n"
            )

        # Predict outcome based on latest game state
        last_move, last_state_snapshot, _, _= self.game_history[-1]
        score = self.static_eval(last_state_snapshot, self.current_game_type)

        if score > 0:
            prediction = "X is *probably* winning. Not that I’m surprised."
        elif score < 0:
            prediction = "O is ahead… but we both know how fast that could crumble."
        else:
            prediction = "It’s neck-and-neck, meaning both of you are equally struggling."

        # Build the taunting summary
        story_text = (
            "Oh wow, what a rollercoaster of *questionable* decisions this has been.\n"
            "The game started with some painfully slow, hesitant moves—classic.\n"
            "Then, a few moments of hope appeared, only to be snuffed out by tragic blunders.\n"
            "Let's relive your missteps, shall we?\n"
            f"{moves_summary}\n"
            "Each turn has been a fascinating blend of blind luck and missed opportunities.\n"
            "The tension is allegedly rising, but honestly, I’ve already calculated *every* possible outcome.\n"
            "You might think you’re setting up something clever, but trust me, I already countered it in my sleep.\n"
            f"So, what's next? Oh right—my inevitable victory. {prediction}\n"
            "Let’s see if you can make this *any* more interesting before I put this game to rest.\n"
        )

        return story_text
    
    def generate_llm_utterance(self, current_state, best_move, opponent_remark, new_stats_summary):
        prompt = (f"You are TicTacTroll, an annoyingly snarky AI. YOU HAVE TO BE SOUND ANNOYING\n"
                f"Opponent just said: '{opponent_remark}'.\n"
                f"You decided to play the move {best_move}.\n"
                f"Here are your search statistics: {new_stats_summary}.\n"
                f"Respond in 3 sentences, dripping with arrogance and be annoying.\n")
        try:
            # Replace with your actual API key
            api_key = "AIzaSyA02_ZY1hfkUHpWxciRNAkOnTqQs2yYhv4"

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
        




    