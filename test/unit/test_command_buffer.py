"""Unit tests for the CommandBuffer -- the eager validation gate (TDD for task 22).

The command buffer is a STAGING area, like git's index: every command is FULLY validated as it enters the
buffer (CommandBuffer.append), so only valid commands are ever staged and world.update() is a pure,
infallible apply -- it materializes the buffer, it does NOT re-validate or roll back (deliberately NOT atomic).

These are UNIT TESTS ON THE COMMAND BUFFER: they append raw Command objects to world._command_buffer and
assert on what it accepts/rejects + its length -- not on entity.add_component / world.update (those become
thin Command-builders over this gate). add_entity already validates eagerly (world.py:74); this brings
add/remove_component to the same bar. Validation has two halves:
  1. structural: dup-add / absent-remove judged against the PROJECTED set (committed + this tick's queued
     adds - queued removes). Valid churn (add->remove->add->remove) is accepted; a same-tick self-conflict
     (the same component added twice) is rejected at the SECOND append, before the poisoning command is staged.
  2. field data: dtype / shape / missing-required / bad-field-name checked at append (via world._validate_components),
     so no field-data error reaches commit.
The valid-churn tests guard the gate against OVER-rejection (update() replays interleaved add/remove); the reject
tests guard against UNDER-validation. Both halves are landed -- task 22 complete.
"""
from dataclasses import field
import numpy as np
import pytest

from microecs import World, Component
from microecs.utils import Command, CommandType


class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


def _add_cmd(entity_id, component, **data):
    """The exact ADD_COMPONENT command entity.add_component queues: the component plus its field data."""
    return Command(CommandType.ADD_COMPONENT, entity_id, args={"component": component, **data})


def _remove_cmd(entity_id, component):
    """The exact REMOVE_COMPONENT command entity.remove_component queues: args is the component itself."""
    return Command(CommandType.REMOVE_COMPONENT, entity_id, args=component)


def _assert_pool_ids_invariants(world: World):
    """The reverse id-map mirrors the pools exactly: no orphan/missing lists, one id per row, every id sits at the
    row it claims, and the union of all ids is precisely the live set (the command buffer is already committed)."""
    assert {id(p) for p in world._pool_ids} == {id(p) for p in world.pools.values()}    # no orphan / missing lists
    seen = set()
    for pool, ids in world._pool_ids.items():
        assert len(ids) == len(pool)                                                    # one id per row
        for ix, eid in enumerate(ids):
            assert world._eid_to_pool_ix[eid] == (pool, ix)                             # ids[ix] really sits at row ix
            seen.add(eid)
    assert seen == set(world.live_entities)                                            # exactly the live entities


# -- valid churn: accepted today (update() replays it) and must stay accepted once append validates --------------

def test_buffer_stages_add_then_remove_same_component():
    """add(V) then remove(V) staged in one tick commits back to the original archetype. The gate must accept this:
    the later remove has to see the queued add (the projected set), not just the committed snapshot."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    buf = world._command_buffer
    buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32")))
    buf.append(_remove_cmd(eid, HasVelocity))                    # accepted: V is in the projected set
    world.update()

    assert set(world.get_entity(eid).get_components()) == {HasPosition}
    _assert_pool_ids_invariants(world)


def test_buffer_stages_remove_then_add_same_component():
    """The mirror: remove(V) then add(V) in one tick commits back to {P, V} with the re-added value winning."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    buf = world._command_buffer
    buf.append(_remove_cmd(eid, HasVelocity))
    buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([5.0, 6.0], "float32")))   # re-add, same tick
    world.update()

    e = world.get_entity(eid)
    assert set(e.get_components()) == {HasPosition, HasVelocity}
    np.testing.assert_array_equal(e.velocity, [5.0, 6.0])       # the re-added value wins
    _assert_pool_ids_invariants(world)


def test_buffer_stages_add_remove_cycle():
    """A longer churn on one entity in a single tick stages cleanly and commits to the expected final archetype."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    buf = world._command_buffer
    buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32")))
    buf.append(_remove_cmd(eid, HasVelocity))
    buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([5.0, 6.0], "float32")))
    buf.append(_remove_cmd(eid, HasVelocity))
    world.update()

    assert set(world.get_entity(eid).get_components()) == {HasPosition}
    _assert_pool_ids_invariants(world)


# -- structural rejects: append must refuse the bad command and NOT stage it --------------------------------------

def test_buffer_rejects_add_of_committed_component():
    """Adding a component the entity already has (committed; empty buffer -> projected == committed) is refused at
    append, and nothing is staged."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    buf = world._command_buffer
    with pytest.raises(ValueError):
        buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([9.0, 9.0], "float32")))
    assert len(buf) == 0                                         # nothing staged
    world.update()                                              # pure no-op


def test_buffer_rejects_remove_of_absent_component():
    """Removing a component the entity does not have (committed) is refused at append, and nothing is staged."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    buf = world._command_buffer
    with pytest.raises(ValueError):
        buf.append(_remove_cmd(eid, HasVelocity))
    assert len(buf) == 0
    world.update()


def test_buffer_rejects_double_add_at_second_append():
    """Same-tick self-conflict is caught at APPEND, not deferred to commit: the second add of a component already
    staged is refused, so the poisoning command never enters the buffer and the first add stays staged."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    buf = world._command_buffer
    buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32")))     # 1st: staged
    with pytest.raises(ValueError):
        buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([5.0, 6.0], "float32")))  # 2nd: eager reject
    assert len(buf) == 1                                         # only the first add is staged


def test_buffer_rejects_double_remove_at_second_append():
    """Removing the same component twice in a tick: the second remove sees it already gone from the projected set
    and is refused at append -- no second, invalid REMOVE_COMPONENT is staged."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    buf = world._command_buffer
    buf.append(_remove_cmd(eid, HasVelocity))                    # 1st: staged
    with pytest.raises(ValueError):
        buf.append(_remove_cmd(eid, HasVelocity))                # 2nd: eager reject
    assert len(buf) == 1


# -- field-data rejects: dtype / shape / missing / bad-name checked at append (via world._validate_components) -----

def test_buffer_rejects_bad_field_name():
    """An extra field that doesn't belong to the added component is refused at append (a component's field set is
    static -- no world state), so it never reaches commit where it would strand a half-processed command (Bug A).
    The valid field is supplied too, to isolate the extra-name path from the missing-required one (its own test)."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    buf = world._command_buffer
    with pytest.raises(ValueError):
        buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32"),
                            WRONGNAME=np.array([3.0, 4.0], "float32")))
    assert len(buf) == 0
    world.update()


def test_buffer_rejects_wrong_shape():
    """Field-data validation is eager too (mirroring add_entity): a wrong-shaped value is refused at append, not
    deferred to commit's _check_components_against_pool."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    buf = world._command_buffer
    with pytest.raises((ValueError, TypeError)):
        buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([1.0, 2.0, 3.0], "float32")))  # (3,) != (2,)
    assert len(buf) == 0
    world.update()


def test_buffer_rejects_wrong_dtype():
    """Same for dtype: a float64 value where the field declares float32 is refused at append, not at commit."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    buf = world._command_buffer
    with pytest.raises((ValueError, TypeError)):
        buf.append(_add_cmd(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float64")))  # float64 != float32
    assert len(buf) == 0


def test_buffer_rejects_missing_required_field():
    """A component field with default=None must be supplied at add; omitting it is refused at append, not at commit
    (where today it surfaces as a KeyError deep in materialization)."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    buf = world._command_buffer
    with pytest.raises((KeyError, ValueError)):
        buf.append(_add_cmd(eid, HasVelocity))                   # velocity has default=None, not provided
    assert len(buf) == 0
    world.update()
