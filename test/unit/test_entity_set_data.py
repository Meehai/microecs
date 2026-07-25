"""Entity-facing spec for `set_data(**fields)` — the ONLY way to write entity data (microecs #29).

Replaces `set_component_data(component, data)` (#24/#25). Two changes:
  * **kwargs, no component arg.** Field names are unique across components (world.field_to_component), so a
    name identifies its component. One call can therefore span SEVERAL components.
  * **the only write path.** `e.field = v` / `e.field[:] = v` / `e.field += v` all raise now — those live in
    test_entity.py. Everything that mutates an entity is buffered: set_data / add_component / remove_component.

Semantics carried over from #25: buffered (`SET_DATA` command, applied at `world.update()`), validated
EAGERLY at the call, schema-only (dtype+shape, not values), and all-or-nothing — a single bad field refuses
the whole call and stages NOTHING.

The command-buffer MECHANICS (append-time gate, projection, apply) live in test_command_buffer.py, driven by
raw Commands. Here we pin the ENTITY API on top of them.
"""
from dataclasses import field
import numpy as np
import pytest

from microecs import World, Component
from microecs.command_buffer import CommandType


class HasA(Component):
    a: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasB(Component):                       # a second component -> cross-component set_data in one call
    b: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasPair(Component):                    # two fields of the SAME component -> one atomic command
    x: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})
    y: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasLabel(Component):                   # object dtype: an arbitrary python object per entity
    label: np.ndarray = field(metadata={"shape": (1,), "dtype": "object", "default": None})


class HasScale(Component):                   # 0-d field: exactly one scalar per entity
    scale: np.ndarray = field(metadata={"shape": (), "dtype": "float32", "default": None})


def _world_a(a=(1.0, 2.0)):
    """A committed world with a single HasA entity (HasB known to the world, not on the entity)."""
    world = World([HasA, HasB])
    eid = world.add_entity((HasA,), a=np.array(a, "float32"))
    world.update()
    return world, eid


# --- the command it builds -----------------------------------------------------------------------------------------

def test_set_data_queues_a_set_data_command_resolving_the_component():
    """Thin builder: one SET_DATA command carrying {field: value} plus the component the FIELD NAME resolves to."""
    world, eid = _world_a()

    assert world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32")) is None   # succeeds -> returns None

    (cmd,) = world._command_buffer.data
    assert cmd.command_type == CommandType.SET_DATA
    assert cmd.entity_id == eid
    assert cmd.args["component"] is HasA                          # resolved from the field name, not passed in
    np.testing.assert_array_equal(cmd.args["a"], [9.0, 8.0])      # flat args, like ADD_COMPONENT


def test_set_data_multi_field_of_one_component_is_a_single_command():
    """Two fields of the same component -> ONE command holding both (the unit the append-gate validates)."""
    world = World([HasPair])
    eid = world.add_entity((HasPair,), x=np.array([1.0, 2.0], "float32"), y=np.array([3.0, 4.0], "float32"))
    world.update()

    world.get_entity(eid).set_data(x=np.array([9.0, 8.0], "float32"), y=np.array([7.0, 6.0], "float32"))

    (cmd,) = world._command_buffer.data                           # one command, not one per field
    assert cmd.args["component"] is HasPair
    assert set(cmd.args) == {"component", "x", "y"}


def test_set_data_empty_call_is_a_noop():
    """set_data() with no kwargs writes nothing and stages nothing (not an error)."""
    world, eid = _world_a()

    assert world.get_entity(eid).set_data() is None
    assert len(world._command_buffer) == 0


def test_set_data_no_longer_exposes_set_component_data():
    """The old name is gone: one write path means one method (#29 supersedes #24/#25's signature)."""
    world, eid = _world_a()

    assert not hasattr(world.get_entity(eid), "set_component_data")


# --- deferred, then applied ----------------------------------------------------------------------------------------

def test_set_data_is_deferred_then_applies():
    """Buffered like add/remove_component: invisible until update(), then visible."""
    world, eid = _world_a()

    world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32"))

    np.testing.assert_array_equal(world.get_entity(eid).a, [1.0, 2.0])   # not yet
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).a, [9.0, 8.0])   # applied


def test_set_data_spans_multiple_components_in_one_call():
    """The point of dropping the `component` arg: fields of DIFFERENT components in a single call."""
    world = World([HasA, HasB])
    eid = world.add_entity((HasA, HasB), a=np.array([1.0, 2.0], "float32"), b=np.array([3.0, 4.0], "float32"))
    world.update()

    world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32"), b=np.array([7.0, 6.0], "float32"))
    world.update()

    e = world.get_entity(eid)
    np.testing.assert_array_equal(e.a, [9.0, 8.0])
    np.testing.assert_array_equal(e.b, [7.0, 6.0])


def test_set_data_partial_multifield_leaves_the_other_field_alone():
    """Setting a SUBSET of a component's fields is legal; the untouched field keeps its value."""
    world = World([HasPair])
    eid = world.add_entity((HasPair,), x=np.array([1.0, 2.0], "float32"), y=np.array([3.0, 4.0], "float32"))
    world.update()

    world.get_entity(eid).set_data(x=np.array([9.0, 8.0], "float32"))      # y omitted
    world.update()

    np.testing.assert_array_equal(world.get_entity(eid).x, [9.0, 8.0])
    np.testing.assert_array_equal(world.get_entity(eid).y, [3.0, 4.0])     # untouched


def test_set_data_copies_the_value_it_is_given():
    """Numeric write lands in the pre-allocated pool row: mutating the source afterwards must not leak in."""
    world, eid = _world_a()

    src = np.array([5.0, 6.0], "float32")
    world.get_entity(eid).set_data(a=src)
    world.update()
    src[:] = [999.0, 999.0]                                               # mutate the source after the write

    np.testing.assert_array_equal(world.get_entity(eid).a, [5.0, 6.0])    # stored value is independent


def test_set_data_object_field_stores_the_reference():
    """An object-dtype field stores the exact python object -- numeric fields copy, objects don't."""
    world = World([HasLabel])
    eid = world.add_entity((HasLabel,), label=np.array([{"v": 0}], dtype=object))
    world.update()

    replacement = {"v": 42}
    world.get_entity(eid).set_data(label=np.array([replacement], dtype=object))
    world.update()

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool.label[ix, 0] is replacement                               # same reference swapped in


def test_set_data_writes_a_zero_dim_field():
    """A shape-() field is written like any other -- set_data is why the `[:]` idiom is no longer needed."""
    world = World([HasScale])
    eid = world.add_entity((HasScale,), scale=np.array(2.5, "float32"))
    world.update()

    world.get_entity(eid).set_data(scale=np.array(4.0, "float32"))
    world.update()

    e = world.get_entity(eid)
    assert e.scale.shape == ()
    np.testing.assert_array_equal(e.scale, 4.0)


# --- eager validation: raise at the CALL, stage nothing ------------------------------------------------------------
# Every rejection below must leave the buffer EMPTY and (after a following update()) the entity untouched --
# "nothing invalid ever stages", so update() stays an infallible apply (#22/#25).

def test_set_data_unknown_field_rejected():
    """A name that is no component's field cannot resolve -> raises, naming it; nothing staged."""
    world, eid = _world_a()

    with pytest.raises((KeyError, ValueError, AttributeError), match="bogus"):
        world.get_entity(eid).set_data(bogus=np.array([1.0, 2.0], "float32"))
    assert len(world._command_buffer) == 0


def test_set_data_field_of_a_component_the_entity_lacks_rejected():
    """`b` is a real field of the world, but this entity has no HasB -> refused at the call (projection gate)."""
    world, eid = _world_a()                                   # HasA only

    with pytest.raises((ValueError, KeyError)):
        world.get_entity(eid).set_data(b=np.array([1.0, 2.0], "float32"))
    assert len(world._command_buffer) == 0
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).a, [1.0, 2.0])    # untouched


def test_set_data_wrong_shape_rejected():
    """(3,) into a (2,) field -> refused at the call, nothing staged."""
    world, eid = _world_a()

    with pytest.raises((ValueError, TypeError)):
        world.get_entity(eid).set_data(a=np.array([1.0, 2.0, 3.0], "float32"))
    assert len(world._command_buffer) == 0


def test_set_data_wrong_dtype_rejected():
    """float64 into a float32 field -> refused at the call (no silent truncation), nothing staged."""
    world, eid = _world_a()

    with pytest.raises((ValueError, TypeError)):
        world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float64"))
    assert len(world._command_buffer) == 0


def test_set_data_non_array_value_rejected():
    """A python list is not an np.ndarray -> refused; microecs never coerces at the boundary."""
    world, eid = _world_a()

    with pytest.raises((TypeError, ValueError)):
        world.get_entity(eid).set_data(a=[9.0, 8.0])
    assert len(world._command_buffer) == 0


def test_set_data_nonfinite_accepted():
    """Schema, not values: NaN/Inf into a matching float field is accepted (finiteness is robosim's #167)."""
    world, eid = _world_a()

    world.get_entity(eid).set_data(a=np.array([np.nan, np.inf], "float32"))
    world.update()

    np.testing.assert_array_equal(world.get_entity(eid).a, np.array([np.nan, np.inf], "float32"))


# --- all-or-nothing across the WHOLE call --------------------------------------------------------------------------
# set_data takes N fields at once, so "atomic" now means atomic across the whole kwargs set -- including across
# components. Validate everything BEFORE staging anything; a per-field append loop would stage the good field and
# then raise on the bad one, leaving a half-transaction in the buffer that the next update() faithfully applies.

def test_set_data_one_bad_field_stages_nothing_same_component():
    """Good x + bad y (wrong shape) in one call -> whole call refused, nothing staged, nothing applied."""
    world = World([HasPair])
    eid = world.add_entity((HasPair,), x=np.array([1.0, 2.0], "float32"), y=np.array([3.0, 4.0], "float32"))
    world.update()

    with pytest.raises((ValueError, TypeError)):
        world.get_entity(eid).set_data(x=np.array([9.0, 8.0], "float32"),
                                       y=np.array([1.0, 2.0, 3.0], "float32"))        # y wrong shape
    assert len(world._command_buffer) == 0
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).x, [1.0, 2.0])                # x NOT written either


# The cross-component case is the one grouping alone cannot fix: `a` and `b` become two commands, so `a`'s is
# already staged when `b`'s is refused. Every reason `b` can be refused must unstage `a` -- parametrized because
# each reason raises from a different place (the resolve loop vs _validate_component's three checks vs the
# projection gate), and only the first is atomic for free.
_BAD_B = [
    ("wrong shape",       np.array([1.0, 2.0, 3.0], "float32")),
    ("wrong dtype",       np.array([1.0, 2.0], "float64")),
    ("not an ndarray",    [1.0, 2.0]),
    ("nan is fine, but",  np.array([np.nan], "float32")),      # right dtype, wrong shape -> still refused
]
@pytest.mark.parametrize("reason,bad_b", _BAD_B, ids=[r for r, _ in _BAD_B])
def test_set_data_one_bad_field_stages_nothing_across_components(reason, bad_b):
    """Whatever refuses `b`, the already-staged `a` command must be unstaged -- else it lands next update()."""
    world = World([HasA, HasB])
    eid = world.add_entity((HasA, HasB), a=np.array([1.0, 2.0], "float32"), b=np.array([3.0, 4.0], "float32"))
    world.update()

    with pytest.raises((ValueError, TypeError, KeyError)):
        world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32"), b=bad_b)
    assert len(world._command_buffer) == 0                                           # `a` unstaged, not just `b`
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).a, [1.0, 2.0])               # a NOT written
    np.testing.assert_array_equal(world.get_entity(eid).b, [3.0, 4.0])


def test_set_data_absent_component_beside_a_valid_field_stages_nothing():
    """`b` resolves but this entity has no HasB -> the projection gate refuses it; `a` must not slip through."""
    world, eid = _world_a()                                    # HasA only; HasB known to the world

    with pytest.raises((ValueError, KeyError)):
        world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32"),
                                       b=np.array([1.0, 2.0], "float32"))
    assert len(world._command_buffer) == 0
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).a, [1.0, 2.0])


def test_set_data_failure_unstages_only_its_own_commands():
    """The rollback must be a SLICE from where this call started -- not a buffer clear.

    A failing set_data is usually not the first thing staged this tick. Commands queued BEFORE the call (here an
    add_component and an unrelated set_data) are other callers' work and must survive; only what this call
    staged is removed. `buf.data.clear()` would pass every other test in this file and silently drop them."""
    world = World([HasA, HasB, HasPair])
    eid = world.add_entity((HasA, HasB), a=np.array([1.0, 2.0], "float32"), b=np.array([3.0, 4.0], "float32"))
    world.update()
    e = world.get_entity(eid)

    e.add_component(HasPair, x=np.array([1.0, 1.0], "float32"), y=np.array([2.0, 2.0], "float32"))  # staged
    e.set_data(a=np.array([7.0, 7.0], "float32"))                                                   # staged
    assert len(world._command_buffer) == 2

    with pytest.raises((ValueError, TypeError)):                    # a ok, b wrong shape -> a's cmd must go
        e.set_data(a=np.array([9.0, 8.0], "float32"), b=np.array([1.0, 2.0, 3.0], "float32"))

    assert len(world._command_buffer) == 2                          # the two earlier commands untouched
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).a, [7.0, 7.0])   # the EARLIER set_data still applied
    np.testing.assert_array_equal(world.get_entity(eid).x, [1.0, 1.0])   # the add_component still applied
    np.testing.assert_array_equal(world.get_entity(eid).b, [3.0, 4.0])   # the failed call changed nothing


def test_set_data_unknown_field_beside_a_valid_one_stages_nothing():
    """An unresolvable name must refuse the call before the valid sibling field is staged."""
    world, eid = _world_a()

    with pytest.raises((KeyError, ValueError, AttributeError)):
        world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32"), bogus=np.array([1.0], "float32"))
    assert len(world._command_buffer) == 0
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).a, [1.0, 2.0])


# --- live view: resolves the CURRENT row ---------------------------------------------------------------------------

def test_set_data_targets_the_correct_row_after_swap_remove():
    """Resolution is by id at apply time: after a swap-remove relocates rows, the write hits the moved entity."""
    world = World([HasA])
    a = world.add_entity((HasA,), a=np.array([0.0, 0.0], "float32"))      # idx 0
    b = world.add_entity((HasA,), a=np.array([1.0, 1.0], "float32"))      # idx 1
    c = world.add_entity((HasA,), a=np.array([2.0, 2.0], "float32"))      # idx 2 (tail)
    world.update()
    world.remove_entity(a)                                                # c swaps into slot 0
    world.update()

    world.get_entity(c).set_data(a=np.array([20.0, 20.0], "float32"))
    world.update()

    np.testing.assert_array_equal(world.get_entity(c).a, [20.0, 20.0])    # c got the write
    np.testing.assert_array_equal(world.get_entity(b).a, [1.0, 1.0])      # neighbour untouched


def test_set_data_on_a_field_added_this_tick_applies_after_the_migration():
    """add_component(B) then set_data(b=..) in one tick: buffer order holds, so the set lands post-migration."""
    world = World([HasA, HasB])
    eid = world.add_entity((HasA,), a=np.array([1.0, 2.0], "float32"))
    world.update()

    e = world.get_entity(eid)
    e.add_component(HasB, b=np.array([0.0, 0.0], "float32"))              # pending: A -> A+B pool
    e.set_data(b=np.array([5.0, 6.0], "float32"))                         # accepted against the projected set
    world.update()

    np.testing.assert_array_equal(world.get_entity(eid).b, [5.0, 6.0])
    np.testing.assert_array_equal(world.get_entity(eid).a, [1.0, 2.0])    # survived the migration


def test_set_data_after_pending_remove_rejected():
    """remove_component(B) then set_data(b=..): the projected set excludes B -> refused at the call."""
    world = World([HasA, HasB])
    eid = world.add_entity((HasA, HasB), a=np.array([1.0, 2.0], "float32"), b=np.array([3.0, 4.0], "float32"))
    world.update()

    e = world.get_entity(eid)
    e.remove_component(HasB)
    with pytest.raises((ValueError, KeyError)):
        e.set_data(b=np.array([5.0, 6.0], "float32"))
    assert len(world._command_buffer) == 1                                # only the remove staged


def test_set_data_through_a_stale_handle_rejects_eagerly():
    """A handle kept across remove_entity must refuse at the call, not queue a command that corrupts update()."""
    world, eid = _world_a()
    e = world.get_entity(eid)                                             # taken before the despawn
    world.remove_entity(eid)                                              # eid leaves live_entities -> stale

    with pytest.raises(ValueError):
        e.set_data(a=np.array([9.0, 8.0], "float32"))
