"""Unit tests for the CommandBuffer -- the eager validation gate.

The command buffer is a STAGING area, like git's index: every command is FULLY validated as it enters the
buffer (CommandBuffer.append), so only valid commands are ever staged and world.update() is a pure,
infallible apply -- it materializes the buffer, it does NOT re-validate or roll back (deliberately NOT atomic).

These are UNIT TESTS ON THE COMMAND BUFFER: they append raw Command objects to world._command_buffer and
assert on what it accepts/rejects + its length -- not on entity.add_component / world.update (those become
thin Command-builders over this gate).

WHICH verbs append gates is a per-verb answer. ADD_ENTITY has exactly ONE producer -- World.add_entity
(world.py:82) -- and that producer validates and fills defaults itself, so append stages it VERBATIM; validating
again there would be pure duplicate work (20% of a spawn) and its tests belong at World.add_entity, in test_world.py.
The component verbs have no such single owner (Entity.add_component/remove_component do no validation of their
own), so append is their gate. What append does for EVERY verb is the liveness check.

For the component verbs, validation has two halves:
  1. structural: dup-add / absent-remove judged against the PROJECTED set (committed + this tick's queued
     adds - queued removes). Valid churn (add->remove->add->remove) is accepted; a same-tick self-conflict
     (the same component added twice) is rejected at the SECOND append, before the poisoning command is staged.
  2. field data: dtype / shape / missing-required / bad-field-name checked at append (via world._validate_components),
     so no field-data error reaches commit.
The valid-churn tests guard the gate against OVER-rejection (update() replays interleaved add/remove); the reject
tests guard against UNDER-validation.
"""
import numpy as np
import pytest

from microecs import World, Component, f32, i32, obj
from microecs.command_buffer import Command, CommandType

from formal import State, setup


class C1(Component):
    c1: np.ndarray = f32((2,), default=[0.0, 0.0])


class C2(Component):
    c2: np.ndarray = f32((2,), default=[0.0, 0.0])


class C3(Component):
    pass


class C4(Component):
    c4: np.ndarray = obj((1,))


class C5(Component):
    c5: np.ndarray = f32((2,))


class C6(Component):
    position: np.ndarray = f32((2,))


class C7(Component):   # optional field: a real default, filled when omitted
    color: np.ndarray = i32((3,), default=[10, 20, 30])


def _add_cmd(entity_id, component, **data):
    """The exact ADD_COMPONENT command entity.add_component queues: the component plus its field data."""
    return Command(CommandType.ADD_COMPONENT, entity_id, args={"component": component, **data})


def _remove_cmd(entity_id, component):
    """The exact REMOVE_COMPONENT command entity.remove_component queues: the component under its 'component' key."""
    return Command(CommandType.REMOVE_COMPONENT, entity_id, args={"component": component})


_COMPONENTS = [C1, C2, C3, C4, C5]


def addc(world, eid: int, component: Component, **data):
    """Append the exact ADD_COMPONENT command entity.add_component would build."""
    world._command_buffer.append(_add_cmd(eid, component, **data))


def rmc(world, eid: int, component: Component):
    """Append the exact REMOVE_COMPONENT command entity.remove_component would build."""
    world._command_buffer.append(_remove_cmd(eid, component))


# -- valid churn: accepted today (update() replays it) and must stay accepted once append validates --------------

def test_buffer_stages_add_then_remove_same_component():
    """addc then rmc in one tick commits back to the original archetype."""
    world = setup({0: [C1]}, _COMPONENTS).world
    addc(world, 0, C2, c2=np.array([3.0, 4.0], "float32"))
    rmc(world, 0, C2)
    world.update()
    assert State(world) == {0: [C1]}


def test_buffer_stages_remove_then_add_same_component():
    """rmc then addc in one tick commits back to {C1, C2} with the re-added value winning."""
    world = setup({0: [C1, C2]}, _COMPONENTS).world
    rmc(world, 0, C2)
    addc(world, 0, C2, c2=np.array([5.0, 6.0], "float32"))
    world.update()
    assert State(world) == {0: [C1, C2]}
    assert np.allclose(world.get_entity(0).c2, [5.0, 6.0])


def test_buffer_stages_add_remove_cycle():
    """A longer churn in one tick stages cleanly and commits to the expected final archetype."""
    world = setup({0: [C1]}, _COMPONENTS).world
    addc(world, 0, C2, c2=np.array([3.0, 4.0], "float32"))
    rmc(world, 0, C2)
    addc(world, 0, C2, c2=np.array([5.0, 6.0], "float32"))
    rmc(world, 0, C2)
    world.update()
    assert State(world) == {0: [C1]}


# -- structural rejects: append must refuse the bad command and NOT stage it --------------------------------------
# These tests poke the buffer directly: the partial staging area IS the thing under test. The formal `act`
# commits and would hide it.

def test_buffer_rejects_add_of_committed_component():
    """Adding a component the entity already has is refused at append, and nothing is staged."""
    world = setup({0: [C1, C2]}, _COMPONENTS).world
    with pytest.raises(ValueError):
        world._command_buffer.append(_add_cmd(0, C2, c2=np.array([9.0, 9.0], "float32")))
    assert len(world._command_buffer) == 0


def test_buffer_rejects_remove_of_absent_component():
    """Removing a component the entity does not have is refused at append, and nothing is staged."""
    world = setup({0: [C1]}, _COMPONENTS).world
    with pytest.raises(ValueError):
        world._command_buffer.append(_remove_cmd(0, C2))
    assert len(world._command_buffer) == 0


def test_buffer_rejects_removing_the_last_component():
    """The gate refuses the removal, so the entity cannot commit with an empty component set."""
    world = setup({0: [C1]}, _COMPONENTS).world
    with pytest.raises(ValueError, match="last component"):
        world._command_buffer.append(_remove_cmd(0, C1))
    assert len(world._command_buffer) == 0


def test_buffer_stages_a_remove_when_a_staged_add_survives():
    """Control: the projected set has C2 after the add, so removing the committed C1 is legal."""
    world = setup({0: [C1]}, _COMPONENTS).world
    addc(world, 0, C2)
    rmc(world, 0, C1)

    assert len(world._command_buffer) == 2
    world.update()
    assert State(world) == {0: [C2]}


def test_buffer_rejects_a_second_removal_that_empties_the_entity():
    """The first removal is staged; the second sees the projected set is empty and stages nothing."""
    world = setup({0: [C1, C3]}, _COMPONENTS).world
    rmc(world, 0, C1)
    with pytest.raises(ValueError, match="last component"):
        rmc(world, 0, C3)
    assert len(world._command_buffer) == 1


def test_buffer_rejects_double_add_at_second_append():
    """The second add of an already-staged component is refused; only the first add is staged."""
    world = setup({0: [C1]}, _COMPONENTS).world  # direct: the partial buffer is the assertion
    with pytest.raises(ValueError):
        world._command_buffer.append(_add_cmd(0, C2, c2=np.array([3.0, 4.0], "float32")))
        world._command_buffer.append(_add_cmd(0, C2, c2=np.array([5.0, 6.0], "float32")))
    assert len(world._command_buffer) == 1


def test_buffer_stages_one_remove_for_a_repeat():
    """Removing the same component twice in a tick stages one command and narrows the archetype once."""
    world = setup({0: [C1, C2]}, _COMPONENTS).world
    world._command_buffer.append(_remove_cmd(0, C2))
    world._command_buffer.append(_remove_cmd(0, C2))
    assert len(world._command_buffer) == 1
    world.update()
    assert State(world) == {0: [C1]}


# -- field-data rejects: dtype / shape / missing / bad-name checked at append (via world._validate_components) -----

def test_buffer_rejects_bad_field_name():
    """An extra field that doesn't belong to the component is refused at append."""
    world = setup({0: [C1]}, _COMPONENTS).world
    with pytest.raises(ValueError):
        world._command_buffer.append(_add_cmd(0, C2, c2=np.array([3.0, 4.0], "float32"),
                                              WRONGNAME=np.array([3.0, 4.0], "float32")))
    assert len(world._command_buffer) == 0


def test_buffer_rejects_wrong_shape():
    """A wrong-shaped value is refused at append, not deferred to commit."""
    world = setup({0: [C1]}, _COMPONENTS).world
    with pytest.raises((ValueError, TypeError)):
        world._command_buffer.append(_add_cmd(0, C2, c2=np.array([1.0, 2.0, 3.0], "float32")))
    assert len(world._command_buffer) == 0


def test_buffer_rejects_wrong_dtype():
    """A float64 value where the field declares float32 is refused at append."""
    world = setup({0: [C1]}, _COMPONENTS).world
    with pytest.raises((ValueError, TypeError)):
        world._command_buffer.append(_add_cmd(0, C2, c2=np.array([3.0, 4.0], "float64")))
    assert len(world._command_buffer) == 0


def test_buffer_rejects_missing_required_field():
    """A required field omitted at add is refused at append."""
    world = setup({0: [C1]}, _COMPONENTS).world
    with pytest.raises((KeyError, ValueError)):
        world._command_buffer.append(_add_cmd(0, C5))
    assert len(world._command_buffer) == 0


# ==================================================================================================================
# The buffer holds STRUCTURE only — there is no SET_DATA verb
# A data write moves no row and invalidates no query, so it is eager (`pool.data[f][ix] = v` at the call; the
# spec lives in test_entity.py / test_entity_set_data.py). What is left is exactly the four verbs that RELOCATE a
# row, which is what deferral is for. Pinned from this side because a leftover SET_DATA branch in append or
# update() would be dead code that still validates, still scans the buffer and still clears the query cache.
# ==================================================================================================================

def test_buffer_verbs_are_the_four_structural_ones():
    """No SET_DATA: the buffer's whole alphabet is spawn / despawn / widen / narrow -- each one moves a row."""
    assert set(CommandType) == {CommandType.ADD_ENTITY, CommandType.REMOVE_ENTITY,
                                CommandType.ADD_COMPONENT, CommandType.REMOVE_COMPONENT}
    assert not hasattr(CommandType, "SET_DATA")


def test_data_write_appends_no_command():
    """The same proof from the buffer's side: both write idioms leave the staging area empty, so a data-only tick
    has nothing to commit and update() keeps the query cache."""
    world = World([C6])
    eid = world.add_entity((C6,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.get_entity(eid).position = np.array([9.0, 8.0], "float32")
    world.get_entity(eid).set_data(position=np.array([7.0, 6.0], "float32"))

    assert len(world._command_buffer) == 0
    assert np.allclose(world.get_entity(eid).position, [7.0, 6.0])


# ==================================================================================================================
# ADD_ENTITY is a PASS-THROUGH
# World.add_entity owns spawn validation (it is the SOLE producer of an ADD_ENTITY command -- world.py:82 is the
# only construction site in the library), and append does nothing for the verb beyond the liveness check every verb
# gets. Validating again in append would be a superset-check of a check that just happened (~2.0 us/spawn: 34% of
# add_entity, 20% of a full spawn, on w5 churn -- the one workload microecs loses at every N). So the rejection tests
# -- shape, dtype, non-ndarray, unknown field, missing-required, duplicate components -- live in test_world.py,
# where the check that raises them lives. Testing them here would defend the buffer against commands its only
# caller cannot build.
#
# What stays here is the buffer's own half of the contract: it stages the spawn UNTOUCHED, and it refuses an id
# the world never registered. The "validated exactly once" guard is at the world layer --
# test_world.py::test_spawn_validates_exactly_once -- because that is where both passes would be. The bookkeeping
# half -- a rejected spawn must not burn an id or leak a live_entities entry -- is in test_world.py too, since that
# is world.add_entity's own state.
# ==================================================================================================================

def _spawn_cmd(entity_id, components, **data):
    """The ADD_ENTITY command world.add_entity queues: the component list plus the field data."""
    return Command(CommandType.ADD_ENTITY, entity_id, args={"components": components, **data})


def _register_id(world, entity_id):
    """Mint an id the way add_entity does, so append's liveness gate passes without going through add_entity."""
    world.live_entities[entity_id] = None
    world._last_id = entity_id


def test_buffer_rejects_a_spawn_for_an_unregistered_id():
    """The liveness check is the ONE thing append does for ADD_ENTITY, and with no validation there it is the
    only thing standing between a hand-built spawn command and update(). An id the world never minted is refused."""
    world = World([C6])                                 # no _register_id: id 0 is not live

    with pytest.raises(ValueError, match="not in live entities"):
        world._command_buffer.append(_spawn_cmd(0, [C6], position=np.array([1.0, 2.0], "float32")))
    assert len(world._command_buffer) == 0
    world.update()
    assert len(world.pools) == 0


@pytest.mark.xfail(strict=True, reason="ADD_COMPONENT defers default-filling to commit")
def test_buffer_alone_fills_defaults_into_a_staged_add_component():
    """An asymmetry to remove. The rule is 'the producer fills defaults': World.add_entity
    fills them for a spawn, so a staged ADD_ENTITY carries a complete arg set. ADD_COMPONENT has no such producer --
    Entity.add_component is a bare command-builder -- so its staged args are incomplete and _do_add_component
    computes them at commit. Both work; pick one story, so a staged command always carries what update() needs."""
    world = World([C6, C7])
    eid = world.add_entity((C6,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.get_entity(eid).add_component(C7)         # color omitted; it has a default

    (cmd,) = world._command_buffer.data
    assert set(cmd.args) == {"component", "color"}                # today: just {"component"}
    assert np.allclose(cmd.args["color"], [10, 20, 30])


# -- broadcastable data: ADD_COMPONENT converts array-likes like add_entity / set_data ------------------------------
# lists/tuples/scalars are accepted by add_component exactly as by add_entity and set_data (same contract: convert,
# None-as-data for object, extra fields rejected), and the converted values are what commit materializes.

def test_add_component_converts_array_like_data():
    """A tuple is converted to the field's dtype/shape at append, and the converted values hit the pool at commit."""
    world = setup({0: [C1]}, _COMPONENTS).world
    world._command_buffer.append(_add_cmd(0, C2, c2=(3.0, 4.0)))
    (cmd,) = world._command_buffer.data
    assert cmd.args["c2"].dtype == np.float32 and cmd.args["c2"].shape == (2,)

    world.update()
    c2 = world.get_entity(0).c2
    assert c2.dtype == np.float32 and c2.shape == (2,)
    assert np.allclose(c2, [3.0, 4.0])


def test_add_component_scalar_broadcast_and_omitted_default():
    """Scalar data is broadcast to the shape; omitted fields fall to the metadata default."""
    world = setup({0: [C1]}, _COMPONENTS).world
    addc(world, 0, C2)
    world.update()
    assert State(world) == {0: [C1, C2]}
    assert np.allclose(world.get_entity(0).c2, [0.0, 0.0])


def test_add_component_none_semantics_match_add_entity():
    """Key-present None is DATA for object fields and a caller bug (TypeError) for numeric ones."""
    world = setup({0: [C1]}, _COMPONENTS).world
    addc(world, 0, C4, c4=None)
    world.update()
    assert State(world) == {0: [C1, C4]}

    c4 = world.get_entity(0).c4
    assert c4.dtype == np.object_ and c4.shape == (1,) and c4[0] is None

    with pytest.raises(TypeError, match="c2"):
        addc(world, 0, C2, c2=None)
    assert State(world) == {0: [C1, C4]}
