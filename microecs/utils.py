"""utils.py - common utilities used by microecs. Mostly logger and types used across the code."""
from loggez import make_logger

PoolKey = int
EntityId = int
Shape = tuple[int, ...]

logger = make_logger("MICROECS")

