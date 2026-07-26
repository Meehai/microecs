"""Unit tests for the CommandBuffer -- the eager validation gate (TDD for task 178).

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
tests guard against UNDER-validation. Both halves are landed -- task 178 complete.
"""
from dataclasses import field
import numpy as np
import pytest

from microecs import World, Component
from microecs.command_buffer import Command, CommandType


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


# ==================================================================================================================
# The buffer holds STRUCTURE only — microecs #42 deleted the SET_DATA verb
# A data write moves no row and invalidates no query, so it is eager now (`pool.data[f][ix] = v` at the call; the
# spec lives in test_entity.py / test_entity_set_data.py). What is left is exactly the four verbs that RELOCATE a
# row, which is what deferral was ever for. Pinned from this side because a leftover SET_DATA branch in append or
# update() would be dead code that still validates, still scans the buffer and still clears the query cache.
# ==================================================================================================================

def test_buffer_verbs_are_the_four_structural_ones():
    """SET_DATA is gone: the buffer's whole alphabet is spawn / despawn / widen / narrow -- each one moves a row."""
    assert set(CommandType) == {CommandType.ADD_ENTITY, CommandType.REMOVE_ENTITY,
                                CommandType.ADD_COMPONENT, CommandType.REMOVE_COMPONENT}
    assert not hasattr(CommandType, "SET_DATA")


def test_data_write_appends_no_command():
    """The same proof from the buffer's side: both write idioms leave the staging area empty, so a data-only tick
    has nothing to commit and update() keeps the query cache (#36 fix 2, moot by construction)."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.get_entity(eid).position = np.array([9.0, 8.0], "float32")
    world.get_entity(eid).set_data(position=np.array([7.0, 6.0], "float32"))

    assert len(world._command_buffer) == 0
    np.testing.assert_array_equal(world.get_entity(eid).position, [7.0, 6.0])


# ==================================================================================================================
# ADD_ENTITY through the gate — task 23 subtasks 1 & 3
# world.add_entity is NOT yet a thin command-builder: it runs _validate_components + _defaults_for itself, and then
# append runs BOTH again on the same args (subtask 1: dead work, and it undercuts "append is the single gate").
# The tests below pin the two properties that make deleting that pre-validation safe:
#   * append alone fully validates a raw ADD_ENTITY (so nothing is lost by removing the earlier pass), and
#   * append alone fills defaults into the staged command (so update() never has to compute them).
# The bookkeeping half — a rejected spawn must not burn an id or leak a live_entities entry (subtask 2) — is in
# test_world.py, since that is world.add_entity's own state.
# ==================================================================================================================

class HasColorDefault(Component):   # optional field: a real default, filled when omitted
    color: np.ndarray = field(metadata={"shape": (3,), "dtype": "int32",
                                        "default": np.array([10, 20, 30], "int32")})


def _spawn_cmd(entity_id, components, **data):
    """The ADD_ENTITY command world.add_entity queues: the component list plus the field data."""
    return Command(CommandType.ADD_ENTITY, entity_id, args={"components": components, **data})


def _register_id(world, entity_id):
    """Mint an id the way add_entity does, so append's liveness gate passes without going through add_entity."""
    world.live_entities[entity_id] = None
    world._last_id = entity_id


@pytest.mark.parametrize("bad_data,reason", [
    ({"position": np.array([1.0, 2.0, 3.0], "float32")}, "wrong shape"),
    ({"position": np.array([1.0, 2.0], "float64")},      "wrong dtype"),
    ({"position": [1.0, 2.0]},                           "not an ndarray"),
    ({"bogus": np.array([1.0, 2.0], "float32")},         "unknown field"),
    ({},                                                 "missing required"),
], ids=["shape", "dtype", "not-ndarray", "unknown-field", "missing-required"])
def test_buffer_alone_fully_validates_a_raw_add_entity(bad_data, reason):
    """append is already a COMPLETE gate for ADD_ENTITY: every field-data error is refused here, with no help from
    world.add_entity's pre-validation. This is what makes subtask 1 (delete that pre-pass) a safe no-op."""
    world = World([HasPosition])
    _register_id(world, 0)

    with pytest.raises((ValueError, TypeError, KeyError)):
        world._command_buffer.append(_spawn_cmd(0, [HasPosition], **bad_data))
    assert len(world._command_buffer) == 0                       # nothing staged
    world.update()
    assert len(world.pools) == 0                                 # nothing materialized


def test_buffer_alone_fills_defaults_into_a_staged_add_entity():
    """append fills omitted defaulted fields into command.args, so a staged ADD_ENTITY carries a COMPLETE arg set
    and update() never computes defaults. (world.add_entity also pre-fills them -- the redundancy of subtask 1.)"""
    world = World([HasPosition, HasColorDefault])
    _register_id(world, 0)

    world._command_buffer.append(                                # color omitted; it has a default
        _spawn_cmd(0, [HasPosition, HasColorDefault], position=np.array([1.0, 2.0], "float32")))

    (cmd,) = world._command_buffer.data
    assert set(cmd.args) == {"components", "position", "color"}   # default filled AT APPEND, not at commit
    np.testing.assert_array_equal(cmd.args["color"], [10, 20, 30])


@pytest.mark.xfail(strict=True, reason="task 23 subtask 3: ADD_COMPONENT defers default-filling to commit")
def test_buffer_alone_fills_defaults_into_a_staged_add_component():
    """The asymmetry subtask 3 wants gone: ADD_ENTITY fills defaults at append (above) but ADD_COMPONENT defers to
    commit, so its staged args are incomplete. Both are correct today; pick one story -- fill at append for both,
    so a staged command always carries everything update() needs."""
    world = World([HasPosition, HasColorDefault])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.get_entity(eid).add_component(HasColorDefault)         # color omitted; it has a default

    (cmd,) = world._command_buffer.data
    assert set(cmd.args) == {"component", "color"}                # today: just {"component"}
    np.testing.assert_array_equal(cmd.args["color"], [10, 20, 30])
