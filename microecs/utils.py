"""clock.py - raylib-based clock allowing to 'drain' it such as to keep a steady FPS rate and consitent physics"""
from loggez import make_logger

EntityId = int
Shape = tuple[int, ...]

logger = make_logger("MICROECS")
