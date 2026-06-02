"""systems.py - Interfaces for various types of systems in ECS"""
from abc import ABC, abstractmethod

from .world import World

class TickSystem(ABC):
    """Generic tick-level system. Called inside the hot main engine loop on every tick"""
    @abstractmethod
    def on_tick(self, scene: World):
        """callback called on each tick"""
