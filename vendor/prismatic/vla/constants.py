"""LIBERO constants for OpenVLA-OFT inference.

Minimal shim that provides the constants needed by the HF Hub modeling code.
Hardcoded to LIBERO values since that's the only checkpoint we use.
"""

from enum import Enum


class NormalizationType(str, Enum):
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"


# LIBERO constants
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2
