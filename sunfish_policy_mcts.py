#!/usr/bin/env python3

import sys, time, os, threading, math, random
from itertools import count
from collections import namedtuple, defaultdict
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

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

version = 'sunfish policy mcts integrated'

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

    def is_terminal(self):
        """Check if position is terminal (checkmate or stalemate)"""
        moves = list(self.gen_moves())
        if not moves:
            return True
        return False

    def get_game_result(self):
        """Get game result from current player's perspective"""
        if not self.is_terminal():
            return None
        
        # Check if in checkmate vs stalemate
        flipped = self.rotate(nullmove=True)
        # Simple check detection (not perfect but good enough for MCTS)
        in_check = any(flipped.board[move.j] == 'K' for move in flipped.gen_moves())
        
        if in_check:
            return -1  # Checkmate, current player loses
        else:
            return 0   # Stalemate, draw

###############################################################################
# MCTS Implementation
###############################################################################

@dataclass
class MCTSEdge:
    """Represents an edge in the MCTS tree (a move)"""
    move: Move
    prior_prob: float
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_losses: int = 0
    child_node: Optional['MCTSNode'] = None
    
    def get_q_value(self):
        """Get Q-value (average value) for this edge"""
        total_visits = self.visit_count + self.virtual_losses
        if total_visits == 0:
            return 0.0
        return self.value_sum / total_visits
    
    def get_uct_value(self, parent_visits: int, c_puct: float = 1.4):
        """Get UCT value for move selection"""
        if self.visit_count == 0:
            return float('inf')
        
        q_value = self.get_q_value()
        exploration = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q_value + exploration

class MCTSNode:
    """Represents a node in the MCTS tree"""
    
    def __init__(self, position: Position, parent: Optional['MCTSNode'] = None):
        self.position = position
        self.parent = parent
        self.edges: List[MCTSEdge] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False
        self.terminal = False
        self.terminal_result = None
        self.lock = threading.Lock()
        
        # Check if terminal
        if position.is_terminal():
            self.terminal = True
            self.terminal_result = position.get_game_result()
    
    def is_expanded(self):
        """Check if node has been expanded"""
        return self.expanded
    
    def expand(self, policy_manager):
        """Expand node by adding edges for all legal moves"""
        if self.expanded or self.terminal:
            return
        
        legal_moves = list(self.position.gen_moves())
        if not legal_moves:
            self.terminal = True
            self.terminal_result = self.position.get_game_result()
            return
        
        # Get policy probabilities
        policy_probs = {}
        if policy_manager and policy_manager.enabled:
            policy_result = policy_manager.get_move_probabilities(self.position.board, legal_moves)
            if policy_result:
                policy_probs, _ = policy_result
        
        # Create edges with uniform priors if no policy
        if not policy_probs:
            uniform_prob = 1.0 / len(legal_moves)
            policy_probs = {move: uniform_prob for move in legal_moves}
        
        # Normalize probabilities
        total_prob = sum(policy_probs.values())
        if total_prob > 0:
            for move in policy_probs:
                policy_probs[move] /= total_prob
        
        # Create edges
        for move in legal_moves:
            prior_prob = policy_probs.get(move, 0.0)
            edge = MCTSEdge(move, prior_prob)
            self.edges.append(edge)
        
        self.expanded = True
    
    def select_edge(self, c_puct: float = 1.4):
        """Select best edge using UCT"""
        if not self.expanded:
            return None
        
        best_edge = None
        best_value = float('-inf')
        
        for edge in self.edges:
            uct_value = edge.get_uct_value(self.visit_count, c_puct)
            if uct_value > best_value:
                best_value = uct_value
                best_edge = edge
        
        return best_edge
    
    def get_best_move(self):
        """Get the best move based on visit counts"""
        if not self.expanded:
            return None
        
        best_edge = max(self.edges, key=lambda e: e.visit_count)
        return best_edge.move
    
    def get_move_probabilities(self, temperature: float = 1.0):
        """Get move probabilities based on visit counts"""
        if not self.expanded:
            return {}
        
        if temperature == 0:
            # Greedy selection
            best_edge = max(self.edges, key=lambda e: e.visit_count)
            return {edge.move: 1.0 if edge == best_edge else 0.0 for edge in self.edges}
        
        # Temperature-based selection
        visit_counts = [max(edge.visit_count, 1) for edge in self.edges]
        if temperature != 1.0:
            visit_counts = [count ** (1.0 / temperature) for count in visit_counts]
        
        total = sum(visit_counts)
        if total == 0:
            return {}
        
        return {edge.move: count / total for edge, count in zip(self.edges, visit_counts)}

class MCTS:
    """Monte Carlo Tree Search implementation"""
    
    def __init__(self, policy_manager, num_threads: int = 4, c_puct: float = 1.4):
        self.policy_manager = policy_manager
        self.num_threads = num_threads
        self.c_puct = c_puct
        self.root = None
        self.nodes_evaluated = 0
        
    def search(self, position: Position, num_rollouts: int = 1000, think_time: float = None):
        """Run MCTS search"""
        self.root = MCTSNode(position)
        self.nodes_evaluated = 0
        
        start_time = time.time()
        
        # Determine stopping condition
        if think_time:
            def should_stop():
                return time.time() - start_time > think_time
        else:
            rollouts_done = 0
            def should_stop():
                nonlocal rollouts_done
                return rollouts_done >= num_rollouts
        
        # Run parallel rollouts
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            
            while not should_stop():
                # Submit rollout tasks
                for _ in range(min(self.num_threads, num_rollouts - (rollouts_done if not think_time else 0))):
                    if should_stop():
                        break
                    future = executor.submit(self._rollout)
                    futures.append(future)
                
                # Wait for some results
                completed_futures = []
                for future in futures:
                    if future.done():
                        completed_futures.append(future)
                        if not think_time:
                            rollouts_done += 1
                
                # Remove completed futures
                for future in completed_futures:
                    futures.remove(future)
                
                # Small delay to prevent tight loop
                time.sleep(0.001)
            
            # Wait for remaining futures
            for future in futures:
                future.result()
        
        return self.root.get_best_move()
    
    def _rollout(self):
        """Single MCTS rollout"""
        path = []
        node = self.root
        
        # Selection phase
        while node.is_expanded() and not node.terminal:
            edge = node.select_edge(self.c_puct)
            if edge is None:
                break
            
            # Add virtual loss
            with edge.child_node.lock if edge.child_node else threading.Lock():
                edge.virtual_losses += 1
            
            path.append(edge)
            
            # Get or create child node
            if edge.child_node is None:
                child_pos = node.position.move(edge.move)
                edge.child_node = MCTSNode(child_pos, node)
            
            node = edge.child_node
        
        # Expansion phase
        value = 0.0
        if node.terminal:
            value = node.terminal_result if node.terminal_result is not None else 0.0
        else:
            if not node.is_expanded():
                node.expand(self.policy_manager)
            
            # Evaluation phase
            value = self._evaluate_position(node.position)
        
        # Backpropagation phase
        self._backpropagate(path, value)
        
        self.nodes_evaluated += 1
    
    def _evaluate_position(self, position: Position):
        """Evaluate position using policy network or simple heuristic"""
        if self.policy_manager and self.policy_manager.enabled:
            try:
                # Try to get value from policy network
                legal_moves = list(position.gen_moves())
                if legal_moves:
                    policy_result = self.policy_manager.get_move_probabilities(position.board, legal_moves)
                    if policy_result:
                        _, value = policy_result
                        return value
            except:
                pass
        
        # Fallback to simple position evaluation
        return self._simple_evaluate(position)
    
    def _simple_evaluate(self, position: Position):
        """Simple position evaluation based on material and position"""
        # Normalize score to [-1, 1] range
        score = position.score
        
        # Apply sigmoid-like function
        if score > 0:
            return min(1.0, score / 1000.0)
        else:
            return max(-1.0, score / 1000.0)
    
    def _backpropagate(self, path: List[MCTSEdge], value: float):
        """Backpropagate value through the path"""
        # Value alternates sign as we go up the tree
        current_value = value
        
        for edge in reversed(path):
            with edge.child_node.lock if edge.child_node else threading.Lock():
                # Remove virtual loss
                edge.virtual_losses -= 1
                
                # Update statistics
                edge.visit_count += 1
                edge.value_sum += current_value
                
                if edge.child_node:
                    edge.child_node.visit_count += 1
                    edge.child_node.value_sum += current_value
            
            # Flip value for parent
            current_value = -current_value

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

###############################################################################
# MCTS Searcher
###############################################################################

class MCTSSearcher:
    """MCTS-based searcher for UCI interface"""
    
    def __init__(self, policy_manager, num_threads=4, num_rollouts=1000):
        self.policy_manager = policy_manager
        self.num_threads = num_threads
        self.num_rollouts = num_rollouts
        self.mcts = MCTS(policy_manager, num_threads)
        self.nodes = 0
    
    def search(self, history, think_time=None):
        """Search for best move using MCTS"""
        position = history[-1]
        
        # Run MCTS search
        best_move = self.mcts.search(position, self.num_rollouts, think_time)
        self.nodes = self.mcts.nodes_evaluated
        
        if best_move:
            # Calculate approximate score for UCI
            root_node = self.mcts.root
            if root_node.is_expanded():
                best_edge = None
                for edge in root_node.edges:
                    if edge.move == best_move:
                        best_edge = edge
                        break
                
                if best_edge:
                    q_value = best_edge.get_q_value()
                    score = int(q_value * 1000)  # Convert to centipawns
                    yield 1, score, score, best_move
        
        return best_move

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
num_threads = 4
num_rollouts = 1000

# Check command line arguments
for i, arg in enumerate(sys.argv[1:]):
    if arg == '--policy-model' and i + 1 < len(sys.argv) - 1:
        model_path = sys.argv[i + 2]
    elif arg == '--policy-size' and i + 1 < len(sys.argv) - 1:
        model_size = sys.argv[i + 2]
    elif arg == '--threads' and i + 1 < len(sys.argv) - 1:
        num_threads = int(sys.argv[i + 2])
    elif arg == '--rollouts' and i + 1 < len(sys.argv) - 1:
        num_rollouts = int(sys.argv[i + 2])

policy_manager = PolicyManager(model_path, model_size)

# Game state tracking for policy training
game_moves = []
game_result = None

searcher = MCTSSearcher(policy_manager, num_threads, num_rollouts)

while True:
    try:
        args = input().split()
        if not args:
            continue
            
        if args[0] == "uci":
            print("id name", version)
            print("id author Rishi Sachdev")
            print("option name Threads type spin default 4 min 1 max 32")
            print("option name Rollouts type spin default 1000 min 100 max 10000")
            print("uciok")

        elif args[0] == "isready":
            print("readyok")

        elif args[0] == "setoption":
            if "name" in args and "value" in args:
                name_idx = args.index("name") + 1
                value_idx = args.index("value") + 1
                if name_idx < len(args) and value_idx < len(args):
                    option_name = args[name_idx]
                    option_value = args[value_idx]
                    
                    if option_name == "Threads":
                        searcher.num_threads = int(option_value)
                        searcher.mcts.num_threads = int(option_value)
                    elif option_name == "Rollouts":
                        searcher.num_rollouts = int(option_value)

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
            # Parse time control
            wtime = btime = winc = binc = 0
            movetime = None
            
            for i, arg in enumerate(args):
                if arg == "wtime" and i + 1 < len(args):
                    wtime = int(args[i + 1]) / 1000.0
                elif arg == "btime" and i + 1 < len(args):
                    btime = int(args[i + 1]) / 1000.0
                elif arg == "winc" and i + 1 < len(args):
                    winc = int(args[i + 1]) / 1000.0
                elif arg == "binc" and i + 1 < len(args):
                    binc = int(args[i + 1]) / 1000.0
                elif arg == "movetime" and i + 1 < len(args):
                    movetime = int(args[i + 1]) / 1000.0
            
            # Calculate thinking time
            if movetime:
                think_time = movetime
            else:
                if len(hist) % 2 == 0:
                    time_left, increment = btime, binc
                else:
                    time_left, increment = wtime, winc
                think_time = min(time_left / 30 + increment, time_left / 2 - 1)
                think_time = max(think_time, 0.1)  # Minimum think time
            
            # Search for best move
            start_time = time.time()
            best_move = None
            
            for depth, gamma, score, move in searcher.search(hist, think_time):
                if move:
                    best_move = move
                    i, j = move.i, move.j
                    if len(hist) % 2 == 0:
                        i, j = 119 - i, 119 - j
                    move_str = render(i) + render(j) + move.prom.lower()
                    print(f"info depth {depth} score cp {score} nodes {searcher.nodes} pv {move_str}")
                    break
            
            # Output best move
            if best_move:
                i, j = best_move.i, best_move.j
                if len(hist) % 2 == 0:
                    i, j = 119 - i, 119 - j
                move_str = render(i) + render(j) + best_move.prom.lower()
                print("bestmove", move_str)
                
                # Update policy network
                if policy_manager:
                    policy_manager.update_policy(hist[-1].board, best_move, game_result)
                    game_moves.append((hist[-1].board, best_move))
                    
                    # Periodically print training stats
                    if len(game_moves) % 50 == 0:
                        stats = policy_manager.get_stats()
                        if stats:
                            print(f"Policy training: {stats['total_examples']} examples")
            else:
                print("bestmove (none)")
    
    except EOFError:
        break
    except Exception as e:
        print(f"Error: {e}")
        continue