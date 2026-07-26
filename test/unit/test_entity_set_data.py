"""Entity-facing spec for `set_data(**fields)` — the multi-field TRANSACTION over the eager write (microecs #42).

`set_data` used to be the *only* entity write path, and it was buffered (#29). Both halves changed:

  * **eager** — it writes straight into the pool, like `e.field = v` and like `qr.field = v`. No `SET_DATA`
    command, no `world.update()` in between. Data writes land now; only STRUCTURE is staged.
  * **not the only path** — `e.field = v` is legal again (test_entity.py). What `set_data` still buys, and the
    only reason it is not a bare `for k, v in data.items(): setattr(self, k, v)` loop, is the **transaction**:
    N fields across N components, validated ALL FIRST, then written none-or-all.

That validate-first ordering is the whole spec of this file, and it is load-bearing in a way it was not under
#29: there is no buffer to roll back any more, so if validation ran per field as the loop wrote, a rejected call
would leave earlier fields already in the pool. Every rejection test below therefore asserts the pool is
**byte-identical**, not merely that "nothing was staged".

Carried over unchanged from #25: validation is schema-only (dtype + shape, never values), and one bad field
refuses the whole call. Kept from #29: the signature — field kwargs, no `component` arg (field names are unique
across components, so a name identifies its component). robosim's three call sites need no change.
"""
from dataclasses import field
import numpy as np
import pytest

from microecs import World, Component


class HasA(Component):
    a: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasB(Component):                       # a second component -> cross-component set_data in one call
    b: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasPair(Component):                    # two fields of the SAME component -> the intra-component transaction
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


def _world_ab():
    """A committed world with a single HasA+HasB entity: a=[1,2], b=[3,4]."""
    world = World([HasA, HasB])
    eid = world.add_entity((HasA, HasB), a=np.array([1.0, 2.0], "float32"), b=np.array([3.0, 4.0], "float32"))
    world.update()
    return world, eid


def _world_pair():
    """A committed world with a single HasPair entity: x=[1,2], y=[3,4]."""
    world = World([HasPair])
    eid = world.add_entity((HasPair,), x=np.array([1.0, 2.0], "float32"), y=np.array([3.0, 4.0], "float32"))
    world.update()
    return world, eid


def _row(world: World, entity_id: int, name: str) -> np.ndarray:
    """The pool row, read WITHOUT going through the Entity view -- so a test can prove the pool itself changed."""
    pool, ix = world._eid_to_pool_ix[entity_id]
    return pool.data[name][ix]


# --- eager: it writes the pool, and stages nothing -----------------------------------------------------------------

def test_set_data_writes_immediately_and_stages_nothing():
    """The headline of #42: the value is in the pool when the call returns. No command, no update() needed."""
    world, eid = _world_a()

    assert world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32")) is None    # succeeds -> returns None

    np.testing.assert_array_equal(_row(world, eid, "a"), [9.0, 8.0])                    # the pool itself
    assert len(world._command_buffer) == 0                                              # SET_DATA is gone


def test_set_data_is_readable_by_the_caller_that_wrote_it():
    """Read-your-own-write, the property #29 could not offer: the next read is the value just written."""
    world, eid = _world_a()
    e = world.get_entity(eid)

    e.set_data(a=np.array([9.0, 8.0], "float32"))

    np.testing.assert_array_equal(e.a, [9.0, 8.0])


def test_set_data_composes_with_a_later_write_to_the_same_field():
    """Two writes to one field in one tick both count, in call order -- no silent lost update (finding 5)."""
    world, eid = _world_a()
    e = world.get_entity(eid)

    e.set_data(a=np.array([5.0, 5.0], "float32"))
    e.set_data(a=e.a + np.float32(1.0))                    # reads its own write, adds to it
    e.a += np.float32(1.0)                                 # and the setattr idiom stacks on top

    np.testing.assert_array_equal(e.a, [7.0, 7.0])


def test_set_data_spans_multiple_components_in_one_call():
    """The point of dropping the `component` arg: fields of DIFFERENT components in a single call."""
    world, eid = _world_ab()

    world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32"), b=np.array([7.0, 6.0], "float32"))

    e = world.get_entity(eid)
    np.testing.assert_array_equal(e.a, [9.0, 8.0])
    np.testing.assert_array_equal(e.b, [7.0, 6.0])


def test_set_data_multi_field_of_one_component_writes_both():
    """Two fields of the same component in one call -- one transaction, both written."""
    world, eid = _world_pair()

    world.get_entity(eid).set_data(x=np.array([9.0, 8.0], "float32"), y=np.array([7.0, 6.0], "float32"))

    np.testing.assert_array_equal(_row(world, eid, "x"), [9.0, 8.0])
    np.testing.assert_array_equal(_row(world, eid, "y"), [7.0, 6.0])


def test_set_data_partial_multifield_leaves_the_other_field_alone():
    """Setting a SUBSET of a component's fields is legal; the untouched field keeps its value."""
    world, eid = _world_pair()

    world.get_entity(eid).set_data(x=np.array([9.0, 8.0], "float32"))      # y omitted

    np.testing.assert_array_equal(_row(world, eid, "x"), [9.0, 8.0])
    np.testing.assert_array_equal(_row(world, eid, "y"), [3.0, 4.0])       # untouched


def test_set_data_empty_call_is_a_noop():
    """set_data() with no kwargs writes nothing and stages nothing (not an error)."""
    world, eid = _world_a()

    assert world.get_entity(eid).set_data() is None
    assert len(world._command_buffer) == 0
    np.testing.assert_array_equal(_row(world, eid, "a"), [1.0, 2.0])


def test_set_data_no_longer_exposes_set_component_data():
    """The old name stays gone: #29 replaced set_component_data(component, data) and #42 keeps that signature."""
    world, eid = _world_a()

    assert not hasattr(world.get_entity(eid), "set_component_data")


def test_set_data_copies_the_value_it_is_given():
    """Numeric write lands in the pre-allocated pool row: mutating the source afterwards must not leak in (#39)."""
    world, eid = _world_a()

    src = np.array([5.0, 6.0], "float32")
    world.get_entity(eid).set_data(a=src)
    src[:] = [999.0, 999.0]                                               # mutate the source after the write

    np.testing.assert_array_equal(_row(world, eid, "a"), [5.0, 6.0])      # stored value is independent


def test_set_data_object_field_stores_the_reference():
    """An object-dtype field stores the exact python object -- numeric fields copy, objects don't."""
    world = World([HasLabel])
    eid = world.add_entity((HasLabel,), label=np.array([{"v": 0}], dtype=object))
    world.update()

    replacement = {"v": 42}
    world.get_entity(eid).set_data(label=np.array([replacement], dtype=object))

    assert _row(world, eid, "label")[0] is replacement                    # same reference swapped in


def test_set_data_writes_a_zero_dim_field():
    """A shape-() field is written like any other -- and it is the ONLY route for one (no `[:]` idiom exists)."""
    world = World([HasScale])
    eid = world.add_entity((HasScale,), scale=np.array(2.5, "float32"))
    world.update()

    world.get_entity(eid).set_data(scale=np.array(4.0, "float32"))

    e = world.get_entity(eid)
    assert e.scale.shape == ()
    np.testing.assert_array_equal(e.scale, 4.0)


def test_set_data_nonfinite_accepted():
    """Schema, not values: NaN/Inf into a matching float field is accepted (finiteness is robosim's #167)."""
    world, eid = _world_a()

    world.get_entity(eid).set_data(a=np.array([np.nan, np.inf], "float32"))

    np.testing.assert_array_equal(_row(world, eid, "a"), np.array([np.nan, np.inf], "float32"))


# --- what it accepts, what it refuses: `__setattr__`'s rule, which is numpy's ---------------------------------------
# set_data must not add a gate of its own: whatever `e.field = v` takes, `set_data(field=v)` takes. The full
# accept/reject boundary and its parity with `qr.field = v` are pinned in test_entity.py. Here: that set_data is not
# stricter, that a NAME it cannot place is still refused, and (below) that one bad field writes nothing at all.

_ACCEPTED = [
    ("list",          [9.0, 8.0],                       [9.0, 8.0]),
    ("python scalar", 5.0,                              [5.0, 5.0]),     # numpy fill, like `qr.a = 0.0`
    ("(1,) fill",     np.array([7.0], "float32"),       [7.0, 7.0]),
    ("float64 array", np.array([9.0, 8.0], "float64"),  [9.0, 8.0]),     # cast down, silently
]
@pytest.mark.parametrize("reason,value,expected", _ACCEPTED, ids=[r for r, _, _ in _ACCEPTED])
def test_set_data_accepts_whatever_numpy_accepts(reason, value, expected):
    """No ndarray required (dev's call, 2026-07-26): converted to the field's dtype, broadcast into its shape."""
    world, eid = _world_a()

    world.get_entity(eid).set_data(a=value)

    np.testing.assert_array_equal(_row(world, eid, "a"), expected)


def test_set_data_unknown_field_rejected():
    """A name that is no component's field cannot resolve -> raises, naming it; nothing written."""
    world, eid = _world_a()

    with pytest.raises((KeyError, ValueError, AttributeError), match="bogus"):
        world.get_entity(eid).set_data(bogus=np.array([1.0, 2.0], "float32"))
    np.testing.assert_array_equal(_row(world, eid, "a"), [1.0, 2.0])


def test_set_data_field_of_a_component_the_entity_lacks_rejected():
    """`b` is a real field of the world, but this entity has no HasB -> there is no column to write."""
    world, eid = _world_a()                                   # HasA only

    with pytest.raises((ValueError, KeyError, AttributeError)):
        world.get_entity(eid).set_data(b=np.array([1.0, 2.0], "float32"))
    np.testing.assert_array_equal(_row(world, eid, "a"), [1.0, 2.0])       # untouched


def test_set_data_not_broadcastable_rejected():
    """(3,) into a (2,) field: numpy itself refuses this one, so it stays a rejection -- and writes nothing."""
    world, eid = _world_a()

    with pytest.raises((ValueError, TypeError)):
        world.get_entity(eid).set_data(a=np.array([1.0, 2.0, 3.0], "float32"))
    np.testing.assert_array_equal(_row(world, eid, "a"), [1.0, 2.0])


# --- all-or-nothing across the WHOLE call -------------------------------------------------------------------------
# The one guarantee that has no buffer to fall back on now (#42): validate EVERY field, then write. A per-field
# "validate this one, write this one" loop passes every test above and fails every test below.

_BAD = [
    ("not broadcastable", np.array([1.0, 2.0, 3.0], "float32")),     # (3,) into (2,): numpy itself refuses
    ("too many dims",     np.ones((2, 2), "float32")),
    ("ragged list",       [[1, 2], [3]]),
    ("non-numeric str",   "abc"),                                    # not convertible to float32
    ("a dict",            {}),
]
_BAD_IDS = [r for r, _ in _BAD]


@pytest.mark.parametrize("reason,bad", _BAD, ids=_BAD_IDS)
def test_set_data_one_bad_field_writes_nothing_same_component(reason, bad):
    """Good x + bad y, same component: x must NOT be in the pool when the exception reaches the caller."""
    world, eid = _world_pair()

    with pytest.raises((ValueError, TypeError, KeyError)):
        world.get_entity(eid).set_data(x=np.array([9.0, 8.0], "float32"), y=bad)

    np.testing.assert_array_equal(_row(world, eid, "x"), [1.0, 2.0])       # x NOT written
    np.testing.assert_array_equal(_row(world, eid, "y"), [3.0, 4.0])


@pytest.mark.parametrize("reason,bad", _BAD, ids=_BAD_IDS)
def test_set_data_one_bad_field_writes_nothing_across_components(reason, bad):
    """Same across components -- the case grouping-by-component alone cannot fix: `a` is a different column, so
    a loop that finishes component A before validating component B has already dirtied the pool."""
    world, eid = _world_ab()

    with pytest.raises((ValueError, TypeError, KeyError)):
        world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32"), b=bad)

    np.testing.assert_array_equal(_row(world, eid, "a"), [1.0, 2.0])       # a NOT written
    np.testing.assert_array_equal(_row(world, eid, "b"), [3.0, 4.0])


@pytest.mark.parametrize("reason,bad", _BAD, ids=_BAD_IDS)
def test_set_data_bad_field_first_also_writes_nothing(reason, bad):
    """Order-independent: the bad field coming FIRST must not write the good one that follows either."""
    world, eid = _world_ab()

    with pytest.raises((ValueError, TypeError, KeyError)):
        world.get_entity(eid).set_data(b=bad, a=np.array([9.0, 8.0], "float32"))

    np.testing.assert_array_equal(_row(world, eid, "a"), [1.0, 2.0])
    np.testing.assert_array_equal(_row(world, eid, "b"), [3.0, 4.0])


def test_set_data_absent_component_beside_a_valid_field_writes_nothing():
    """`b` resolves to HasB but this entity has no such column -> `a` must not slip through first."""
    world, eid = _world_a()                                    # HasA only; HasB known to the world

    with pytest.raises((ValueError, KeyError, AttributeError)):
        world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32"), b=np.array([1.0, 2.0], "float32"))

    np.testing.assert_array_equal(_row(world, eid, "a"), [1.0, 2.0])


def test_set_data_unknown_field_beside_a_valid_one_writes_nothing():
    """An unresolvable name must refuse the call before the valid sibling field is written."""
    world, eid = _world_a()

    with pytest.raises((KeyError, ValueError, AttributeError)):
        world.get_entity(eid).set_data(a=np.array([9.0, 8.0], "float32"), bogus=np.array([1.0], "float32"))

    np.testing.assert_array_equal(_row(world, eid, "a"), [1.0, 2.0])


def test_set_data_failure_leaves_earlier_work_alone():
    """A failing set_data is usually not the first thing that happened this tick. Earlier writes (already in the
    pool) and earlier STAGED structural commands are other callers' work and must survive untouched -- the old
    implementation had a buffer slice to undo, this one must simply not touch anything."""
    world = World([HasA, HasB, HasPair])
    eid = world.add_entity((HasA, HasB), a=np.array([1.0, 2.0], "float32"), b=np.array([3.0, 4.0], "float32"))
    world.update()
    e = world.get_entity(eid)

    e.add_component(HasPair, x=np.array([1.0, 1.0], "float32"), y=np.array([2.0, 2.0], "float32"))  # staged
    e.set_data(a=np.array([7.0, 7.0], "float32"))                                                  # written
    assert len(world._command_buffer) == 1

    with pytest.raises((ValueError, TypeError)):                     # a ok, b wrong shape -> whole call refused
        e.set_data(a=np.array([9.0, 8.0], "float32"), b=np.array([1.0, 2.0, 3.0], "float32"))

    assert len(world._command_buffer) == 1                           # the staged migration untouched
    np.testing.assert_array_equal(e.a, [7.0, 7.0])                   # the EARLIER write still stands
    np.testing.assert_array_equal(e.b, [3.0, 4.0])                   # the failed call changed nothing
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).x, [1.0, 1.0])   # and the migration still applies
    np.testing.assert_array_equal(world.get_entity(eid).a, [7.0, 7.0])   # value carried across the pool move


# --- live view: the row must exist, and it must be the CURRENT one -------------------------------------------------

def test_set_data_targets_the_correct_row_after_swap_remove():
    """Resolution is by id: after a swap-remove relocates rows, the write hits the moved entity, not a neighbour."""
    world = World([HasA])
    a = world.add_entity((HasA,), a=np.array([0.0, 0.0], "float32"))      # idx 0
    b = world.add_entity((HasA,), a=np.array([1.0, 1.0], "float32"))      # idx 1
    c = world.add_entity((HasA,), a=np.array([2.0, 2.0], "float32"))      # idx 2 (tail)
    world.update()
    world.remove_entity(a)                                                # c swaps into slot 0
    world.update()

    world.get_entity(c).set_data(a=np.array([20.0, 20.0], "float32"))

    np.testing.assert_array_equal(world.get_entity(c).a, [20.0, 20.0])    # c got the write
    np.testing.assert_array_equal(world.get_entity(b).a, [1.0, 1.0])      # neighbour untouched


def test_set_data_before_the_spawn_is_committed_rejects():
    """No row yet -> nowhere to write, so it raises exactly like a read does (dev's call, 2026-07-26). Spawn data
    belongs in add_entity's kwargs; the entity is writable from the next update() on."""
    world = World([HasA])
    eid = world.add_entity((HasA,), a=np.array([1.0, 2.0], "float32"))     # not committed
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        e.set_data(a=np.array([9.0, 8.0], "float32"))

    world.update()
    np.testing.assert_array_equal(e.a, [1.0, 2.0])                        # the spawn value, unaffected
    e.set_data(a=np.array([9.0, 8.0], "float32"))                         # and now it works
    np.testing.assert_array_equal(e.a, [9.0, 8.0])


def test_set_data_on_a_field_added_this_tick_rejects_until_update():
    """add_component(B) is STAGED, so B's column does not exist yet: the write has nowhere to go. A write never
    consults the command buffer -- so this raises now and succeeds after update()."""
    world = World([HasA, HasB])
    eid = world.add_entity((HasA,), a=np.array([1.0, 2.0], "float32"))
    world.update()
    e = world.get_entity(eid)

    e.add_component(HasB, b=np.array([0.0, 0.0], "float32"))              # pending: A -> A+B pool
    with pytest.raises((ValueError, KeyError, AttributeError)):
        e.set_data(b=np.array([5.0, 6.0], "float32"))
    assert len(world._command_buffer) == 1                                # only the add_component

    world.update()
    e.set_data(b=np.array([5.0, 6.0], "float32"))
    np.testing.assert_array_equal(e.b, [5.0, 6.0])
    np.testing.assert_array_equal(e.a, [1.0, 2.0])                        # survived the migration


def test_set_data_on_a_field_pending_removal_lands_then_is_dropped():
    """Mirror case: remove_component(B) is staged but B's column is still live, so the write lands NOW -- and
    update() then drops the component and its data. (Under #29 this call was refused by the projection gate.)"""
    world, eid = _world_ab()
    e = world.get_entity(eid)

    e.remove_component(HasB)
    e.set_data(b=np.array([5.0, 6.0], "float32"))                         # accepted: the column exists
    np.testing.assert_array_equal(e.b, [5.0, 6.0])

    world.update()
    with pytest.raises(AttributeError):
        _ = e.b                                                           # gone with the component
    np.testing.assert_array_equal(e.a, [1.0, 2.0])


def test_set_data_through_a_stale_handle_rejects():
    """A handle kept across a committed remove_entity has no row: the write must refuse, not poke the entity that
    was swapped into that slot."""
    world = World([HasA])
    eid = world.add_entity((HasA,), a=np.array([1.0, 2.0], "float32"))
    other = world.add_entity((HasA,), a=np.array([3.0, 4.0], "float32"))  # swaps into eid's slot on removal
    world.update()
    e = world.get_entity(eid)
    world.remove_entity(eid)
    world.update()

    with pytest.raises((AttributeError, KeyError, ValueError)):
        e.set_data(a=np.array([9.0, 8.0], "float32"))

    np.testing.assert_array_equal(world.get_entity(other).a, [3.0, 4.0])  # neighbour untouched
