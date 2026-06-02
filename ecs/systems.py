"""systems.py - Interfaces for various types of systems in ECS"""
from abc import ABC, abstractmethod

from .world import World

class TickSystem(ABC):
    @abstractmethod
    def on_tick(self, scene: World):
        pass
