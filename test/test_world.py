"""Unit tests for ecs.World"""
from dataclasses import dataclass, field
import numpy as np
import pytest

from ecs import World


@dataclass(kw_only=True)
class HasPosition:
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


@dataclass(kw_only=True)
class HasVelocity:
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


def test_add_entity_rejects_field_from_an_unrequested_trait():
    """An entity declared with only HasPosition may not pass `velocity` (a field of the unrequested HasVelocity)."""
    world = World(traits=[HasPosition, HasVelocity])  # both traits known to the world

    with pytest.raises(AssertionError, match="velocity"):
        world.add_entity(
            traits=(HasPosition,),                          # entity declares HasPosition only
            position=np.array([1.0, 2.0], "float32"),       # required by HasPosition
            velocity=np.array([3.0, 4.0], "float32"),       # extra: belongs to HasVelocity, not requested
        )
