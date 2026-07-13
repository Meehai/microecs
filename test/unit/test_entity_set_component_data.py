"""Entity-facing spec for set_component_data — the thin wrapper over the SET_DATA command (microecs #25).

The command-buffer MECHANICS (append-time validation, projection, staging length, apply) live in
test_command_buffer.py, mirroring add/remove_component. Here we only pin the ENTITY API: that
entity.set_component_data(component, data) queues the right SET_DATA command and the change is deferred
until world.update().
"""
from dataclasses import field
import numpy as np

from microecs import World, Component
from microecs.command_buffer import CommandType


class HasA(Component):
    a: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


def test_set_component_data_queues_a_set_data_command():
    """Thin builder: the entity method queues exactly one SET_DATA command carrying (component, data)."""
    world = World([HasA])
    eid = world.add_entity((HasA,), a=np.array([1.0, 2.0], "float32"))
    world.update()

    assert world.get_entity(eid).set_component_data(HasA, {"a": np.array([9.0, 8.0], "float32")}) is None

    (cmd,) = world._command_buffer.data
    assert cmd.command_type == CommandType.SET_DATA
    assert cmd.entity_id == eid
    assert cmd.args["component"] is HasA
    np.testing.assert_array_equal(cmd.args["data"]["a"], [9.0, 8.0])


def test_set_component_data_is_deferred_then_applies():
    """Through the entity API the write is deferred: not visible until update(), then visible."""
    world = World([HasA])
    eid = world.add_entity((HasA,), a=np.array([1.0, 2.0], "float32"))
    world.update()

    world.get_entity(eid).set_component_data(HasA, {"a": np.array([9.0, 8.0], "float32")})

    np.testing.assert_array_equal(world.get_entity(eid).a, [1.0, 2.0])   # not yet
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).a, [9.0, 8.0])   # applied
