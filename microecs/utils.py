"""utils.py - common utilities used by microecs. Mostly logger and types used across the code."""
from loggez import make_logger
import numpy as np

PoolKey = int
EntityId = int
EntityData = dict[str, np.ndarray]
Shape = tuple[int, ...]

logger = make_logger("MICROECS")
