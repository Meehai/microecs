"""component.py - submodule holind the base class for all components, which is just a dataclass wrapper"""
from dataclasses import dataclass

class Component:
    """Base for ECS components: subclassing auto-applies @dataclass(kw_only=True)."""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclass(kw_only=True)(cls)
