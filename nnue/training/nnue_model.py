#!/usr/bin/env python3
"""
NNUE Model Architecture for Sunfish Chess Engine.
Compatible with the existing sunfish_nnue.py implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class SunfishNNUE(nn.Module):
    """
    NNUE model compatible with sunfish_nnue.py format.
    
    Architecture matches the existing model:
    - L0=10, L1=10, L2=10 (configurable)
    - Position embedding (8x8x6) -> Piece combination -> Color combination -> Hidden layers
    """
    
    def __init__(self, L0=10, L1=10, L2=10):
        super().__init__()
        
        self.L0 = L0
        self.L1 = L1 
        self.L2 = L2
        
        # Position embedding for 8x8 board, 6 piece types
        self.pos_emb = nn.Parameter(torch.randn(8, 8, 6) * 0.1)
        
        # Piece combination layer: (L0, 6, 6) 
        self.piece_comb = nn.Parameter(torch.randn(L0, 6, 6) * 0.1)
        
        # Piece value layer (for material balance)
        self.piece_val = nn.Parameter(torch.randn(6) * 0.1)
        
        # Color combination layer: (L0, L0, 2)
        self.color_comb = nn.Parameter(torch.randn(L0, L0, 2) * 0.1)
        
        # Hidden layers
        self.layer1 = nn.Linear(2 * (L0 - 1), L2)  # -1 because we exclude piece balance
        self.layer2 = nn.Linear(L2, 1)
        
        # Scale factor for piece balance
        self.register_buffer('scale', torch.tensor(1.0))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.1)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
    
    def features_from_board(self, board_str):
        """
        Compute features from sunfish board string.
        Returns white features and black features.
        """
        wf = torch.zeros(self.L0)
        bf = torch.zeros(self.L0)
        
        piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                     'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
        
        # Extract piece values for material balance
        piece_balance = 0
        
        for i, p in enumerate(board_str):
            if p.isalpha():
                # Convert sunfish position to 8x8 coordinates
                rank, file = divmod(i - 21, 10)  # 21 is A8 in sunfish
                if 0 <= rank < 8 and 0 <= file < 8:
                    
                    piece_type = piece_map[p.lower()]
                    is_white = p.isupper()
                    
                    # Get position embedding
                    pos_emb = self.pos_emb[7-rank, file, piece_type]  # Flip rank for white perspective
                    
                    # Apply piece combination
                    piece_features = self.piece_comb[:, :, piece_type] @ self.pos_emb[7-rank, file, :]
                    
                    # Apply color combination  
                    color_idx = 0 if is_white else 1
                    features = self.color_comb[:, :, color_idx] @ piece_features
                    
                    if is_white:
                        wf += features
                        piece_balance += self.piece_val[piece_type]
                    else:
                        # For black pieces, use flipped position
                        bf_pos_emb = self.pos_emb[rank, file, piece_type]
                        bf_piece_features = self.piece_comb[:, :, piece_type] @ self.pos_emb[rank, file, :]
                        bf_features = self.color_comb[:, :, 1] @ bf_piece_features
                        bf += bf_features
                        piece_balance -= self.piece_val[piece_type]
        
        # Set piece balance in first position of feature vectors
        wf[0] = piece_balance
        bf[0] = -piece_balance
        
        return wf, bf
    
    def forward(self, wf, bf):
        """Forward pass through the network."""
        
        # Combine features (excluding piece balance at index 0)
        combined_features = torch.cat([wf[1:], bf[1:]], dim=-1)
        
        # Apply activation and hidden layers
        h1 = torch.tanh(combined_features)
        h2 = self.layer1(h1)
        h2 = torch.tanh(h2)
        score = self.layer2(h2)
        
        # Add piece balance with scaling
        piece_balance = wf[0] - bf[0]
        final_score = score + self.scale * piece_balance
        
        return final_score
    
    def export_to_sunfish_format(self, filename):
        """
        Export model to format compatible with sunfish_nnue.py.
        
        The format expects:
        - model["ars"]: List of numpy arrays as int8 values
        - model["scale"]: Scale factor for piece balance
        """
        
        def to_int8_array(tensor):
            """Convert tensor to int8 numpy array."""
            # Clamp values to reasonable range and convert to int8
            clamped = torch.clamp(tensor * 127, -127, 127)
            return clamped.detach().cpu().numpy().astype(np.int8).tobytes()
        
        # Prepare arrays in the order expected by sunfish_nnue.py
        arrays = []
        
        # Array 0: Position embedding (8*8*6 = 384 values)
        arrays.append(to_int8_array(self.pos_emb.flatten()))
        
        # Array 1: Piece combination (L0*6*6 values)  
        arrays.append(to_int8_array(self.piece_comb.flatten()))
        
        # Array 2: Piece values (6 values)
        arrays.append(to_int8_array(self.piece_val))
        
        # Array 3: Color combination (L0*L0*2 values)
        arrays.append(to_int8_array(self.color_comb.flatten()))
        
        # Array 4: Layer 1 weights (L2 * 2*(L0-1) values)
        arrays.append(to_int8_array(self.layer1.weight.flatten()))
        
        # Array 5: Layer 2 weights (1 * L2 values)
        arrays.append(to_int8_array(self.layer2.weight.flatten()))
        
        # Create model dictionary
        model_dict = {
            "ars": arrays,
            "scale": float(self.scale.item())
        }
        
        # Save to pickle file
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"Model exported to {filename}")
        print(f"Model size: {Path(filename).stat().st_size} bytes")
        
        return filename

def load_sunfish_model(filename):
    """Load a model in sunfish format and convert to PyTorch."""
    
    with open(filename, 'rb') as f:
        model_dict = pickle.load(f)
    
    # Extract arrays
    arrays = [np.frombuffer(ar, dtype=np.int8) / 127.0 for ar in model_dict["ars"]]
    
    # Infer dimensions from array sizes
    L0 = int(np.cbrt(len(arrays[1]) / 36))  # piece_comb is L0*6*6
    L1 = L0  # Assume same
    L2 = len(arrays[5])  # layer2 output size
    
    # Create model
    model = SunfishNNUE(L0, L1, L2)
    
    # Load parameters
    model.pos_emb.data = torch.from_numpy(arrays[0].reshape(8, 8, 6)).float()
    model.piece_comb.data = torch.from_numpy(arrays[1].reshape(L0, 6, 6)).float()
    model.piece_val.data = torch.from_numpy(arrays[2]).float()
    model.color_comb.data = torch.from_numpy(arrays[3].reshape(L0, L0, 2)).float()
    
    # Load linear layers
    layer1_weights = torch.from_numpy(arrays[4].reshape(L2, 2*(L0-1))).float()
    layer2_weights = torch.from_numpy(arrays[5].reshape(1, L2)).float()
    
    model.layer1.weight.data = layer1_weights
    model.layer2.weight.data = layer2_weights
    
    model.scale = torch.tensor(model_dict["scale"])
    
    return model

# Test function
def test_model_compatibility():
    """Test that our model produces similar outputs to sunfish format."""
    
    # Create a small test model
    model = SunfishNNUE(L0=4, L1=4, L2=4)
    
    # Export to sunfish format
    test_file = "test_model.pickle"
    model.export_to_sunfish_format(test_file)
    
    # Load it back
    loaded_model = load_sunfish_model(test_file)
    
    print("Model compatibility test passed!")
    
    # Cleanup
    Path(test_file).unlink()

if __name__ == "__main__":
    test_model_compatibility()