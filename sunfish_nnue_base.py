#!/usr/bin/env pypy3
"""Base classes and constants from sunfish_nnue that can be imported without side effects."""

from collections import namedtuple

# Initial board setup
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

# Board constants
A1, H1, A8, H8 = 91, 98, 21, 28

# Move representation
Move = namedtuple("Move", "i j prom")

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N + N, N + W, N + E),
    "N": (N + N + E,
        E + N + E,
        E + S + E,
        S + S + E,
        S + S + W,
        W + S + W,
        W + N + W,
        N + N + W,
    ),
    "B": (N + E, S + E, S + W, N + W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N + E, S + E, S + W, N + W),
    "K": (N, E, S, W, N + E, S + E, S + W, N + W),
}

# Position class (simplified version without methods for imports)
Position = namedtuple("Position", "board score wf bf wc bc ep kp")