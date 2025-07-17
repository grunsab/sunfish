#!/usr/bin/env python3

import sys, time, os
from itertools import count
from collections import namedtuple, defaultdict
from functools import partial

# Add NNUE training directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'nnue', 'training'))

try:
    from policy_network import SunfishPolicyNetwork, PolicyTrainer, create_policy_network
    import torch
    POLICY_AVAILABLE = True
except ImportError:
    POLICY_AVAILABLE = False
    print("Policy network not available. Install PyTorch to enable policy features.")

print = partial(print, flush=True)

version = 'sunfish policy integrated'

###############################################################################
# Piece-Square tables (same as original sunfish)
###############################################################################

piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}

# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8 : i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20

###############################################################################
# Global constants
###############################################################################

A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"  #   0 -  9
    "         \n"  #  10 - 19
    " rnbqkbnr\n"  #  20 - 29
    " pppppppp\n"  #  30 - 39
    " ........\n"  #  40 - 49
    " ........\n"  #  50 - 59
    " ........\n"  #  60 - 69
    " ........\n"  #  70 - 79
    " PPPPPPPP\n"  #  80 - 89
    " RNBQKBNR\n"  #  90 - 99
    "         \n"  # 100 -109
    "         \n"  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate values
MATE_LOWER = piece["K"] - 10 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]

# Constants for tuning search
QS = 40
QS_A = 140
EVAL_ROUGHNESS = 15

###############################################################################
# Policy Network Integration
###############################################################################

class PolicyManager:
    """Manages the policy network and training"""
    
    def __init__(self, model_path=None, model_size='small'):
        self.policy_network = None
        self.policy_trainer = None
        self.enabled = POLICY_AVAILABLE
        
        if self.enabled:
            self.policy_network = create_policy_network(model_size)
            self.policy_trainer = PolicyTrainer(self.policy_network)
            
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
                print(f"Loaded policy model from {model_path}")
            else:
                print("Initialized new policy network")
    
    def get_move_probabilities(self, board, legal_moves):
        """Get policy probabilities for legal moves"""
        if not self.enabled or not self.policy_network:
            return None
        
        try:
            return self.policy_network.get_policy_probs(board, legal_moves)
        except Exception as e:
            print(f"Policy network error: {e}")
            return None
    
    def update_policy(self, board, move_played, game_result=None):
        """Update policy with the move that was played"""
        if not self.enabled or not self.policy_trainer:
            return
        
        try:
            self.policy_trainer.add_training_example(board, move_played, game_result)
            
            # Periodic training
            if len(self.policy_trainer.training_data) % 100 == 0:
                loss = self.policy_trainer.train_step()
                if loss is not None:
                    print(f"Policy training loss: {loss:.4f}")
        except Exception as e:
            print(f"Policy update error: {e}")
    
    def save_model(self, filepath):
        """Save the current policy model"""
        if self.enabled and self.policy_trainer:
            self.policy_trainer.save_model(filepath)
            print(f"Policy model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a policy model"""
        if self.enabled and self.policy_trainer:
            self.policy_trainer.load_model(filepath)
    
    def get_stats(self):
        """Get training statistics"""
        if self.enabled and self.policy_trainer:
            return self.policy_trainer.get_training_stats()
        return {}

# Global policy manager
policy_manager = None

def initialize_policy(model_path=None, model_size='small'):
    """Initialize the policy manager"""
    global policy_manager
    policy_manager = PolicyManager(model_path, model_size)

###############################################################################
# Chess logic (same as original sunfish)
###############################################################################

Move = namedtuple("Move", "i j prom")

class Position(namedtuple("Position", "board score wc bc ep kp")):
    """A state of a chess game"""
    
    def gen_moves(self):
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    if q.isspace() or q.isupper():
                        break
                    if p == "P":
                        if d in (N, N + N) and q != ".": break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."): break
                        if (
                            d in (N + W, N + E)
                            and q == "."
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                        ):
                            break
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                yield Move(i, j, prom)
                            break
                    yield Move(i, j, "")
                    if p in "PNK" or q.islower():
                        break
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    def rotate(self, nullmove=False):
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )

    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        
        board = put(board, j, board[i])
        board = put(board, i, ".")
        
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")
        
        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, ".")
        
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        score = pst[p][j] - pst[p][i]
        if q.islower():
            score += pst[q.upper()][119 - j]
        if abs(j - self.kp) < 2:
            score += pst["K"][119 - j]
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        if p == "P":
            if A8 <= j <= H8:
                score += pst[prom][j] - pst["P"][j]
            if j == self.ep:
                score += pst["P"][119 - (j + S)]
        return score

###############################################################################
# Search logic with policy integration
###############################################################################

Entry = namedtuple("Entry", "lower upper")

class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    def bound(self, pos, gamma, depth, can_null=True):
        """Search with policy-guided move ordering"""
        self.nodes += 1
        depth = max(depth, 0)

        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        entry = self.tp_score.get((pos, depth, can_null), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma: return entry.lower
        if entry.upper < gamma: return entry.upper

        if can_null and depth > 0 and pos in self.history:
            return 0

        def moves():
            # Null move pruning
            if depth > 2 and can_null and abs(pos.score) < 500:
                yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)

            # Quiescence search
            if depth == 0:
                yield None, pos.score

            # Get legal moves
            legal_moves = list(pos.gen_moves())
            
            # Use policy network for move ordering if available
            if policy_manager and policy_manager.enabled:
                policy_result = policy_manager.get_move_probabilities(pos.board, legal_moves)
                if policy_result:
                    move_probs, _ = policy_result
                    # Sort moves by policy probability (descending)
                    sorted_moves = sorted(legal_moves, key=lambda m: move_probs.get(m, 0), reverse=True)
                else:
                    # Fall back to value-based ordering
                    sorted_moves = sorted(legal_moves, key=lambda m: pos.value(m), reverse=True)
            else:
                # Traditional value-based ordering
                sorted_moves = sorted(legal_moves, key=lambda m: pos.value(m), reverse=True)

            # Hash move (killer heuristic)
            killer = self.tp_move.get(pos)
            if killer and killer in sorted_moves:
                sorted_moves.remove(killer)
                sorted_moves.insert(0, killer)

            # Search moves
            val_lower = QS - depth * QS_A
            
            for move in sorted_moves:
                if depth == 0 and pos.value(move) < val_lower:
                    continue
                
                if depth <= 1 and pos.score + pos.value(move) < gamma:
                    yield move, pos.score + pos.value(move) if pos.value(move) < MATE_LOWER else MATE_UPPER
                    break

                yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1)

        best = -MATE_UPPER
        best_move = None
        for move, score in moves():
            if score > best:
                best = score
                best_move = move
            if best >= gamma:
                if move is not None:
                    self.tp_move[pos] = move
                break

        # Stalemate checking
        if depth > 2 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            in_check = self.bound(flipped, MATE_UPPER, 0) == MATE_UPPER
            best = -MATE_LOWER if in_check else 0

        # Update transposition table
        if best >= gamma:
            self.tp_score[pos, depth, can_null] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, can_null] = Entry(entry.lower, best)

        return best

    def search(self, history):
        """Iterative deepening search"""
        self.nodes = 0
        self.history = set(history)
        self.tp_score.clear()

        gamma = 0
        for depth in range(1, 1000):
            lower, upper = -MATE_LOWER, MATE_LOWER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(history[-1], gamma, depth, can_null=False)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                yield depth, gamma, score, self.tp_move.get(history[-1])
                gamma = (lower + upper + 1) // 2

###############################################################################
# UCI User interface
###############################################################################

def parse(c):
    fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
    return A1 + fil - 10 * rank

def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)

hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]

# Initialize policy network
model_path = None
model_size = 'small'

# Check command line arguments
for i, arg in enumerate(sys.argv[1:]):
    if arg == '--policy-model' and i + 1 < len(sys.argv) - 1:
        model_path = sys.argv[i + 2]
    elif arg == '--policy-size' and i + 1 < len(sys.argv) - 1:
        model_size = sys.argv[i + 2]

initialize_policy(model_path, model_size)

# Game state tracking for policy training
game_moves = []
game_result = None

searcher = Searcher()
while True:
    try:
        args = input().split()
        if not args:
            continue
        if args[0] == "uci":
            print("id name", version)
            print("uciok")

        elif args[0] == "isready":
            print("readyok")

        elif args[0] == "quit":
            # Save policy model before quitting
            if policy_manager and policy_manager.enabled:
                policy_manager.save_model("policy_model_auto_save.pth")
            break

        elif args[:2] == ["position", "startpos"]:
            # Reset game state
            game_moves = []
            game_result = None
            
            del hist[1:]
            for ply, move in enumerate(args[3:]):
                i, j, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
                if ply % 2 == 1:
                    i, j = 119 - i, 119 - j
                move_obj = Move(i, j, prom)
                hist.append(hist[-1].move(move_obj))
                game_moves.append((hist[-2].board, move_obj))

        elif args[0] == "go":
            wtime, btime, winc, binc = [int(a) / 1000 for a in args[2::2]]
            if len(hist) % 2 == 0:
                wtime, winc = btime, binc
            think = min(wtime / 40 + winc, wtime / 2 - 1)

            start = time.time()
            move_str = None
            best_move = None
            
            for depth, gamma, score, move in searcher.search(hist):
                if score >= gamma:
                    i, j = move.i, move.j
                    if len(hist) % 2 == 0:
                        i, j = 119 - i, 119 - j
                    move_str = render(i) + render(j) + move.prom.lower()
                    best_move = move
                    print("info depth", depth, "score cp", score, "pv", move_str)
                        
                if move_str and time.time() - start > think * 0.8:
                    break

            print("bestmove", move_str or '(none)')
            
            # Update policy network with the move we played
            if best_move and policy_manager:
                policy_manager.update_policy(hist[-1].board, best_move, game_result)
                
                # Add to game moves for training
                game_moves.append((hist[-1].board, best_move))
                
                # Periodically print training stats
                if len(game_moves) % 50 == 0:
                    stats = policy_manager.get_stats()
                    if stats:
                        print(f"Policy training: {stats['total_examples']} examples, {stats['unique_moves']} unique moves")
    
    except EOFError:
        break
    except Exception as e:
        print(f"Error: {e}")
        continue