#!/usr/bin/env python3
"""
Policy Network Architecture for Sunfish Chess Engine.
Implements a policy where the policy equals the next move that's played.
Based on pytorch-alpha-zero but adapted for Sunfish's 120-char board representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
from pathlib import Path
import sys
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class PolicyHead(nn.Module):
    def __init__(self, input_channels, num_moves=4096):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 8 * 8, num_moves)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ValueHead(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))

class SunfishPolicyNetwork(nn.Module):
    """
    Policy network that learns to predict the next move that will be played.
    Architecture similar to AlphaZero but adapted for Sunfish's board representation.
    """
    
    def __init__(self, input_channels=16, num_residual_blocks=4, num_channels=256, num_moves=4096):
        super().__init__()
        
        self.input_conv = ConvBlock(input_channels, num_channels)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])
        self.policy_head = PolicyHead(num_channels, num_moves)
        self.value_head = ValueHead(num_channels)
        
        # Move encoding/decoding
        self.move_to_index = {}
        self.index_to_move = {}
        self._build_move_mappings()
        
    def _build_move_mappings(self):
        """Build mapping between moves and network output indices"""
        index = 0
        
        # Regular moves (from square to square)
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq != to_sq:
                    move_key = (from_sq, to_sq)
                    self.move_to_index[move_key] = index
                    self.index_to_move[index] = move_key
                    index += 1
        
        # Promotion moves
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq != to_sq:
                    for prom in ['N', 'B', 'R', 'Q']:
                        move_key = (from_sq, to_sq, prom)
                        self.move_to_index[move_key] = index
                        self.index_to_move[index] = move_key
                        index += 1
        
        print(f"Total moves encoded: {index}")
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.input_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
    def sunfish_pos_to_square(self, pos):
        """Convert Sunfish 120-char position to 8x8 square index"""
        # Sunfish uses 10x12 board with padding
        # Position 21 = A8, 22 = B8, etc.
        rank, file = divmod(pos - 21, 10)
        if 0 <= rank < 8 and 0 <= file < 8:
            return rank * 8 + file
        return None
    
    def move_to_key(self, move):
        """Convert Sunfish Move object to network key"""
        from_sq = self.sunfish_pos_to_square(move.i)
        to_sq = self.sunfish_pos_to_square(move.j)
        
        if from_sq is None or to_sq is None:
            return None
        
        if move.prom:
            return (from_sq, to_sq, move.prom)
        else:
            return (from_sq, to_sq)
    
    def board_to_tensor(self, board_string):
        """Convert Sunfish board string to tensor representation"""
        # Create 16-channel tensor (6 pieces x 2 colors + 4 game state features)
        tensor = torch.zeros(16, 8, 8)
        
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        # Extract 8x8 board from 120-char representation
        for rank in range(8):
            for file in range(8):
                pos = 21 + rank * 10 + file  # Convert to sunfish indexing
                piece = board_string[pos]
                
                if piece in piece_map:
                    tensor[piece_map[piece], rank, file] = 1.0
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def get_policy_probs(self, board_string, legal_moves):
        """Get policy probabilities for legal moves"""
        self.eval()
        with torch.no_grad():
            board_tensor = self.board_to_tensor(board_string)
            policy_logits, value = self(board_tensor)
            
            # Convert to probabilities
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            # Extract probabilities for legal moves
            move_probs = {}
            total_prob = 0.0
            
            for move in legal_moves:
                move_key = self.move_to_key(move)
                if move_key and move_key in self.move_to_index:
                    prob = policy_probs[self.move_to_index[move_key]]
                    move_probs[move] = prob
                    total_prob += prob
                else:
                    # Small default probability for unmapped moves
                    move_probs[move] = 0.01
                    total_prob += 0.01
            
            # Normalize probabilities
            if total_prob > 0:
                for move in move_probs:
                    move_probs[move] /= total_prob
            else:
                # Uniform distribution fallback
                uniform_prob = 1.0 / len(legal_moves)
                move_probs = {move: uniform_prob for move in legal_moves}
            
            return move_probs, value.item()

class PolicyTrainer:
    """
    Trainer for the policy network that learns from played moves.
    Implements the concept where policy equals the next move played.
    """
    
    def __init__(self, network, learning_rate=0.001, device='cpu'):
        self.network = network
        self.device = device
        self.network.to(device)
        
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training data storage
        self.training_data = []
        self.batch_size = 32
        self.max_data_size = 10000
        
    def add_training_example(self, board_string, move_played, game_result=None):
        """
        Add a training example where the policy should predict the move that was played.
        This implements the core concept: policy = next_move_played
        """
        
        # Convert move to target index
        move_key = self.network.move_to_key(move_played)
        if move_key and move_key in self.network.move_to_index:
            target_index = self.network.move_to_index[move_key]
            
            # Store training example
            example = {
                'board': board_string,
                'move': move_played,
                'target_index': target_index,
                'game_result': game_result
            }
            
            self.training_data.append(example)
            
            # Keep only recent examples
            if len(self.training_data) > self.max_data_size:
                self.training_data.pop(0)
    
    def train_step(self):
        """Perform one training step on accumulated data"""
        if len(self.training_data) < self.batch_size:
            return None
        
        # Sample batch
        batch_indices = np.random.choice(len(self.training_data), self.batch_size, replace=False)
        batch = [self.training_data[i] for i in batch_indices]
        
        # Prepare tensors
        board_tensors = []
        target_indices = []
        
        for example in batch:
            board_tensor = self.network.board_to_tensor(example['board'])
            board_tensors.append(board_tensor)
            target_indices.append(example['target_index'])
        
        # Stack tensors
        board_batch = torch.cat(board_tensors, dim=0).to(self.device)
        target_batch = torch.tensor(target_indices, dtype=torch.long).to(self.device)
        
        # Forward pass
        policy_logits, value = self.network(board_batch)
        
        # Compute loss
        policy_loss = self.criterion(policy_logits, target_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_data_size': len(self.training_data),
            'move_to_index': self.network.move_to_index,
            'index_to_move': self.network.index_to_move
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.network.move_to_index = checkpoint['move_to_index']
        self.network.index_to_move = checkpoint['index_to_move']
    
    def export_training_data(self, filepath):
        """Export training data for analysis"""
        with open(filepath, 'w') as f:
            json.dump(self.training_data, f, indent=2, default=str)
    
    def get_training_stats(self):
        """Get statistics about training data"""
        if not self.training_data:
            return {}
        
        # Count move frequencies
        move_counts = defaultdict(int)
        for example in self.training_data:
            move_key = self.network.move_to_key(example['move'])
            if move_key:
                move_counts[move_key] += 1
        
        return {
            'total_examples': len(self.training_data),
            'unique_moves': len(move_counts),
            'most_common_moves': sorted(move_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }

def create_policy_network(model_size='small', device='cpu'):
    """Create a policy network with specified size configuration"""
    
    size_configs = {
        'tiny': {'num_residual_blocks': 2, 'num_channels': 64, 'num_moves': 2048},
        'small': {'num_residual_blocks': 4, 'num_channels': 128, 'num_moves': 4096},
        'medium': {'num_residual_blocks': 6, 'num_channels': 256, 'num_moves': 4096},
        'large': {'num_residual_blocks': 8, 'num_channels': 512, 'num_moves': 4096}
    }
    
    config = size_configs.get(model_size, size_configs['small'])
    
    network = SunfishPolicyNetwork(
        input_channels=16,
        num_residual_blocks=config['num_residual_blocks'],
        num_channels=config['num_channels'],
        num_moves=config['num_moves']
    )
    
    return network

def test_policy_network():
    """Test the policy network functionality"""
    
    # Create test network
    network = create_policy_network('tiny')
    trainer = PolicyTrainer(network)
    
    # Test board representation
    from sunfish import initial, Position, Move
    
    initial_board = initial
    print(f"Initial board length: {len(initial_board)}")
    
    # Test tensor conversion
    tensor = network.board_to_tensor(initial_board)
    print(f"Board tensor shape: {tensor.shape}")
    
    # Test move encoding
    test_move = Move(81, 71, "")  # e2-e4 in sunfish coordinates
    move_key = network.move_to_key(test_move)
    print(f"Move key for e2-e4: {move_key}")
    
    # Test policy prediction
    pos = Position(initial_board, 0, (True, True), (True, True), 0, 0)
    legal_moves = list(pos.gen_moves())
    print(f"Legal moves: {len(legal_moves)}")
    
    move_probs, value = network.get_policy_probs(initial_board, legal_moves[:10])
    print(f"Policy probabilities: {len(move_probs)}")
    print(f"Value: {value}")
    
    # Test training
    trainer.add_training_example(initial_board, legal_moves[0])
    print(f"Training data size: {len(trainer.training_data)}")
    
    print("Policy network test completed successfully!")

if __name__ == "__main__":
    test_policy_network()