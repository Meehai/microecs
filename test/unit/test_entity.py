"""Unit tests for microecs.Entity -- the object-like view returned by world.get_entity(id).

Entity is a LIVE view of one row, not a snapshot: every attribute access re-resolves (pool, row) from the id,
so it stays correct across pool changes (swap-remove, archetype migration).

**One rule (#42): a STRUCTURAL change is buffered, a DATA write is EAGER.** Not "Entity is buffered,
QueryResult is eager" (#29) -- the line is drawn at what the write does, not at which object you hold:

  * `e.field = v`, `e.field[:] = v`, `e.field += v`, `e.set_data(f=v)`  -> land in the pool NOW
  * `e.add_component(...)`, `e.remove_component(...)`, `world.{add,remove}_entity(...)` -> at `world.update()`

Moving a row between pools invalidates iteration in flight, so structure must be staged. `pool.data[f][ix] = v`
moves nothing and invalidates nothing, so staging it bought nothing -- and it cost: a read-modify-write silently
kept only the last contribution (`damage(3); damage(4)` left 6), a read could not see its own write, and every
read paid `setflags(write=False)` (491 vs 227 ns/op) to police an idiom that is legal again.

A write goes where the row is NOW and consults the command buffer never (dev's call, 2026-07-26). Two corollaries
pinned below: a not-yet-committed spawn refuses writes exactly as it refuses reads, and a field whose component is
only *pending* has no column yet, so writing it raises.

`set_data`'s own spec (the multi-field transaction) lives in test_entity_set_data.py. Here we pin the view:
reads, the eager writes, what still raises, liveness across pool churn, and to_dict.

Issues these pin:
  (1) eager data write -- lands now, composes with the next read, stages nothing.
  (2) bare error       -- a bad field name must name the field and the valid set.
  (3) failed write     -- a rejected write leaves the pool byte-identical (no buffer left to fall back on).
  (4) internal attrs   -- Entity's own instance attrs must all be in the __setattr__ allowlist, or the class
                          cannot even be constructed once __setattr__ starts routing field writes.
"""
from dataclasses import field
import copy
import json
import pickle
import numpy as np
import pytest

from microecs import World, Component, Entity
from microecs.entity import _ENTITY_INTERNAL_ATTRS, ENTITY_RESERVED_NAMES


class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasHealth(Component):  # the accumulator field: read-modify-write must compose (#1, plan 2 finding 5)
    health: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "default": None})


class HasLabel(Component):  # object dtype: an arbitrary python object per entity (-> to_dict uses .item())
    label: np.ndarray = field(metadata={"shape": (1,), "dtype": "object", "default": None})


class HasScale(Component):  # 0-d array field: exactly one scalar per entity (shape ())
    scale: np.ndarray = field(metadata={"shape": (), "dtype": "float32", "default": None})


class HasPose(Component):   # (4, 4) field: the sliced read AND the sliced write (e.pose[0:3, 3]) must work
    pose: np.ndarray = field(metadata={"shape": (4, 4), "dtype": "float32", "default": None})


class HasSerial(Component):  # two fields; the 'serializable' extra-metadata drives to_dict's filter
    keep: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "serializable": True, "default": None})
    drop: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "serializable": False, "default": None})


def _world_with_one(position=(1.0, 2.0)):
    """A committed world holding a single HasPosition entity. Returns (world, entity_id)."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array(position, "float32"))
    world.update()
    return world, eid


def _row(world: World, entity_id: int, name: str) -> np.ndarray:
    """The pool row, read WITHOUT going through the Entity view -- so a test can prove the pool itself changed."""
    pool, ix = world._eid_to_pool_ix[entity_id]
    return pool.data[name][ix]


# --- construction: the __setattr__ allowlist must cover Entity's own attrs -----------------------------------------
# __setattr__ routes a field name to the pool and Entity's own attrs to super(); everything else raises. So
# __init__ itself goes through the guard. Any instance attr __init__ sets that is missing from the allowlist makes
# Entity unconstructable -> get_entity raises for every entity in the world (#29's defect 1: 95 failures). The
# allowlist must also feed ENTITY_RESERVED_NAMES, else a component field could be named like an internal attr and
# shadow it. #42 is likely to change WHICH attrs those are (a `_world` back-ref for validation) -- these tests do
# not care about the names, only that the set and the allowlist agree.

def test_entity_can_be_constructed_at_all():
    """get_entity must build the view: every attr __init__ assigns is in the __setattr__ allowlist."""
    world, eid = _world_with_one()

    e = world.get_entity(eid)                        # must not raise -- __init__ writes through __setattr__

    assert isinstance(e, Entity)


def test_entity_internal_attrs_allowlist_covers_every_instance_attr():
    """Whatever __init__ set on the instance must be exactly the allowlist -- a new attr must be added to it."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    assert set(vars(e)) == _ENTITY_INTERNAL_ATTRS    # no unlisted attr slipped in, no listed attr unset


def test_entity_reserved_names_include_every_internal_attr():
    """ENTITY_RESERVED_NAMES gates component field names, so it must cover the internal attrs too."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    assert set(vars(e)) <= ENTITY_RESERVED_NAMES     # else a component field could shadow an internal attr


# --- read ---------------------------------------------------------------------------------------------------------

def test_entity_reads_field_components_and_fields():
    """The view exposes the row by attribute, plus its current component types and field names."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    np.testing.assert_array_equal(e.position, [1.0, 2.0])
    assert set(e.get_components()) == {HasPosition}
    assert e.get_fields() == {"position"}
    assert e.entity_id == eid


def test_entity_read_hands_back_a_writable_view_of_the_pool():
    """#42 deletes the read-only guard: the row is a WRITABLE view into pool memory, not a frozen copy.

    This is the whole read tax (`isinstance` + `setflags(write=False)` on every field read, 491 vs 227 ns/op).
    It existed to police `e.field[:] = v`, which is legal again -- so the guard has nothing left to buy."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    row = e.position

    assert row.flags.writeable is True                                 # not frozen
    assert np.shares_memory(row, _row(world, eid, "position"))         # and it IS the pool's memory


def test_entity_unknown_field_read_raises_named_error():
    """Reading a field the entity's pool doesn't have raises AttributeError naming the field and the valid set."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError, match="velocity"):
        _ = e.velocity                                   # not a field of this (HasPosition-only) entity


def test_entity_sliced_read_still_works():
    """Sliced/fancy reads into a (4, 4) field -- the robosim translation-read idiom."""
    world = World([HasPose])
    pose = np.arange(16, dtype="float32").reshape(4, 4)
    eid = world.add_entity((HasPose,), pose=pose.copy())
    world.update()
    e = world.get_entity(eid)

    np.testing.assert_array_equal(e.pose[0:3, 3], pose[0:3, 3])
    np.testing.assert_array_equal(e.pose[0], pose[0])
    assert float(e.pose[1, 1]) == pose[1, 1]
    np.testing.assert_array_equal(e.pose.copy(), pose)


def test_entity_read_is_a_live_view_not_a_copy():
    """A bulk column write through a query is visible on the entity view -- same buffer, no stale copy."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    world.query(HasPosition).position = np.array([[7.0, 7.0]], "float32")   # the vectorised write path

    np.testing.assert_array_equal(e.position, [7.0, 7.0])


# --- eager data write: every idiom lands in the pool NOW: issue (1) ------------------------------------------------
# All four of these used to raise (#29). They are the pre-#29 idioms, back on purpose: a data write moves no row,
# so there is nothing to stage. Each test asserts the POOL changed (not just the view) and that the command buffer
# stayed empty -- there is no SET_DATA command any more.

def test_entity_attribute_write_lands_in_the_pool_immediately():
    """`e.field = v`: validate, then `pool.data[f][ix] = v`. No update() needed, nothing staged."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position = np.array([9.0, 8.0], "float32")

    np.testing.assert_array_equal(_row(world, eid, "position"), [9.0, 8.0])   # the pool itself
    assert len(world._command_buffer) == 0                                    # no command, no deferral


def test_entity_write_never_rebinds_the_pool_column():
    """A write goes INTO a row: the column must be the same array object, same shape, afterwards.

    `pool.data[f] = v` instead of `pool.data[f][ix] = v` is a one-character slip that type-checks and passes a
    naive round-trip test -- the writer reads its own value back. What it actually does is replace the whole
    (capacity, *shape) column with one row's worth of data: every other entity's value is gone and the next spawn
    dies with `IndexError: index N is out of bounds for axis 0 with size 2`. Pinned for both write paths."""
    world, eid = _world_with_one()
    other = world.add_entity((HasPosition,), position=np.array([3.0, 4.0], "float32"))
    world.update()
    pool, _ = world._eid_to_pool_ix[eid]
    column = pool.data["position"]

    world.get_entity(eid).position = np.array([9.0, 8.0], "float32")          # setattr path
    world.get_entity(eid).set_data(position=np.array([7.0, 6.0], "float32"))  # single-field set_data path

    assert pool.data["position"] is column                                   # not swapped out
    assert pool.data["position"].shape == (pool.capacity, 2)                 # still the full column
    np.testing.assert_array_equal(world.get_entity(other).position, [3.0, 4.0])   # neighbour's row intact
    world.add_entity((HasPosition,), position=np.array([5.0, 5.0], "float32"))
    world.update()                                                          # and the pool can still grow


def test_entity_attribute_write_copies_the_value_it_is_given():
    """The write lands in the pre-allocated row, so mutating the source afterwards must not leak in (#39)."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    src = np.array([5.0, 6.0], "float32")
    e.position = src
    src[:] = [999.0, 999.0]                                                  # mutate the source after the write

    np.testing.assert_array_equal(e.position, [5.0, 6.0])                    # stored value is independent


def test_entity_slice_write_lands_immediately():
    """`e.field[:] = v` -- the in-place idiom the read-only view used to forbid."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position[:] = np.array([9.0, 8.0], "float32")

    np.testing.assert_array_equal(_row(world, eid, "position"), [9.0, 8.0])
    assert len(world._command_buffer) == 0


def test_entity_element_write_lands_immediately():
    """`e.field[0] = v`: a single element, the other one untouched."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position[0] = 9.0

    np.testing.assert_array_equal(_row(world, eid, "position"), [9.0, 2.0])


def test_entity_inplace_add_lands_immediately():
    """`e.field += v`: numpy mutates the row in place, then __setattr__ writes the same buffer back. Both halves
    are now legal, so the op is a plain accumulate instead of a raise-after-dirtying."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position += np.array([10.0, 10.0], "float32")

    np.testing.assert_array_equal(_row(world, eid, "position"), [11.0, 12.0])


def test_entity_multidim_slice_write_lands_immediately():
    """`e.pose[0:3, 3] = t` -- robosim's set-translation idiom, on a (4, 4) field."""
    world = World([HasPose])
    eid = world.add_entity((HasPose,), pose=np.zeros((4, 4), "float32"))
    world.update()
    e = world.get_entity(eid)

    e.pose[0:3, 3] = np.array([1.0, 2.0, 3.0], "float32")

    np.testing.assert_array_equal(_row(world, eid, "pose")[0:3, 3], [1.0, 2.0, 3.0])
    assert _row(world, eid, "pose")[3, 3] == 0.0                             # nothing else touched


def test_entity_object_field_slice_write_swaps_the_reference():
    """An object-dtype row is a view too -- `e.label[0] = obj` swaps the stored reference, no copy."""
    world = World([HasLabel])
    eid = world.add_entity((HasLabel,), label=np.array([{"v": 0}], dtype=object))
    world.update()
    e = world.get_entity(eid)

    replacement = {"v": 42}
    e.label[0] = replacement

    assert _row(world, eid, "label")[0] is replacement                       # the object itself, not a copy


def test_entity_zero_dim_field_write_lands_immediately():
    """A shape-() field is written whole (`e.scale = np.float32 array`); there is no row to slice into."""
    world = World([HasScale])
    eid = world.add_entity((HasScale,), scale=np.array(2.5, "float32"))
    world.update()
    e = world.get_entity(eid)

    e.scale = np.array(4.0, "float32")

    np.testing.assert_array_equal(_row(world, eid, "scale"), 4.0)
    assert e.scale.shape == ()


# --- what the eager write buys: composability and read-your-own-write ---------------------------------------------
# These are the tests #29 could not pass. They are the reason #42 exists -- not the microseconds.

def test_entity_read_your_own_write_within_one_tick():
    """No update() in between: the next read sees the value just written."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position = np.array([9.0, 8.0], "float32")

    np.testing.assert_array_equal(e.position, [9.0, 8.0])       # under #29 this was still [1.0, 2.0]


def test_entity_read_modify_write_composes_within_one_tick():
    """Two independent read-modify-writes on one field in one tick BOTH count: damage(3); damage(4) -> 3, not 6.

    This is plan 2 finding 5 (silent lost updates) and #1's bounce-impulse accumulator. Under #29 the reads were
    eager and the writes buffered, so both calls read 10 and the second staged command overwrote the first."""
    world = World([HasHealth])
    eid = world.add_entity((HasHealth,), health=np.array([10.0], "float32"))
    world.update()
    e = world.get_entity(eid)

    def damage(amount):                                          # two callers, neither knows about the other
        e.health = e.health - np.float32(amount)

    damage(3.0)
    damage(4.0)

    np.testing.assert_array_equal(e.health, [3.0])               # 10 - 3 - 4, not 10 - 4


def test_entity_inplace_accumulator_composes_within_one_tick():
    """Same property through the `+=` idiom -- the shape an impulse accumulator actually takes."""
    world = World([HasHealth])
    eid = world.add_entity((HasHealth,), health=np.array([0.0], "float32"))
    world.update()
    e = world.get_entity(eid)

    for contribution in (1.0, 2.0, 3.0):
        e.health += np.float32(contribution)

    np.testing.assert_array_equal(e.health, [6.0])               # every contribution, not just the last


def test_entity_write_is_visible_through_a_query_taken_before_it():
    """A query held from before the write sees it: both point at the same pool memory, no re-query needed."""
    world, eid = _world_with_one()
    qr = world.query(HasPosition)                                # taken BEFORE the write

    world.get_entity(eid).position = np.array([9.0, 8.0], "float32")

    np.testing.assert_array_equal(qr.position[0], [9.0, 8.0])


def test_entity_write_is_visible_to_a_batch_read_modify_write():
    """The two APIs compose in one tick: an entity write then a batch write see each other's values in order."""
    world, eid = _world_with_one(position=(1.0, 1.0))
    e = world.get_entity(eid)

    e.position += np.array([1.0, 1.0], "float32")                # -> [2, 2]
    qr = world.query(HasPosition)
    qr.position = qr.position * 10.0                             # batch sees [2, 2] -> [20, 20]

    np.testing.assert_array_equal(e.position, [20.0, 20.0])      # and the entity view sees that


def test_data_only_tick_keeps_the_query_cache_alive():
    """A data write does not stage a command, so update() has nothing to commit and the query cache SURVIVES.

    Under #29 every set_data made the buffer non-empty, so update() dropped every cached query -- a data-only
    tick invalidated queries that structurally could not have changed (#36 fix 2, now moot by construction)."""
    world, eid = _world_with_one()
    qr = world.query(HasPosition)

    world.get_entity(eid).position = np.array([9.0, 8.0], "float32")
    world.update()

    assert world.query(HasPosition) is qr                        # same object: nothing was invalidated


def test_entity_write_does_not_disturb_staged_structural_commands():
    """Eager data + buffered structure coexist: the write lands now, the staged migration still applies at
    update(), and the freshly written value SURVIVES the pool move (the row is copied across)."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.add_component(HasVelocity, velocity=np.zeros(2, "float32"))     # staged: HasPosition -> HasPosition+HasVelocity
    e.position = np.array([9.0, 8.0], "float32")                     # eager, into the OLD pool
    assert len(world._command_buffer) == 1                           # the write added nothing to the buffer
    np.testing.assert_array_equal(e.position, [9.0, 8.0])

    world.update()

    np.testing.assert_array_equal(e.position, [9.0, 8.0])            # carried over by the migration
    np.testing.assert_array_equal(e.velocity, [0.0, 0.0])


# --- what a write ACCEPTS: numpy's rules, the same ones `qr.field = v` follows --------------------------------------
# Dev's call, 2026-07-26: the value does not have to be an ndarray. Anything numpy can turn into the field's dtype
# and broadcast into the field's shape is accepted -- python lists, tuples, scalars, a (1,) fill, a float64 array.
# That is not a loosening: the BATCH path has always worked this way (`qr.position = 0.0` fills every row; a float64
# column write truncates silently), so this makes the two write paths one rule instead of two. The parity test below
# is the real spec; the individual cases just document what that rule means. It costs the "no silent truncation"
# guarantee (#25) on this path -- `add_entity` / `add_component` remain STRICT, so declaring a row and updating one
# now differ. Deliberate, and the divergence is pinned in test_world.py.

_ACCEPTED = [
    ("list",            [9.0, 8.0],                          [9.0, 8.0]),
    ("tuple",           (9.0, 8.0),                          [9.0, 8.0]),
    ("int list",        [1, 2],                              [1.0, 2.0]),
    ("python scalar",   5.0,                                 [5.0, 5.0]),     # numpy fill, like `qr.position = 0.0`
    ("(1,) fill",       np.array([7.0], "float32"),          [7.0, 7.0]),
    ("float64 array",   np.array([9.0, 8.0], "float64"),     [9.0, 8.0]),     # cast down, silently
    ("exact match",     np.array([9.0, 8.0], "float32"),     [9.0, 8.0]),
]
_REJECTED = [
    ("(3,) too big",    np.array([1.0, 2.0, 3.0], "float32")),
    ("(2, 2) too big",  np.ones((2, 2), "float32")),
    ("ragged list",     [[1, 2], [3]]),
    ("non-numeric str", "abc"),
    ("a dict",          {}),
    ("a bare object",   object()),
]


@pytest.mark.parametrize("reason,value,expected", _ACCEPTED, ids=[r for r, _, _ in _ACCEPTED])
def test_entity_write_accepts_whatever_numpy_accepts(reason, value, expected):
    """No ndarray required: the value is converted to the field's dtype and broadcast into the field's shape."""
    world, eid = _world_with_one()

    world.get_entity(eid).position = value

    np.testing.assert_array_equal(_row(world, eid, "position"), expected)
    assert _row(world, eid, "position").dtype == "float32"           # the COLUMN's dtype never changes


_EVERY_VALUE = [*[(r, v) for r, v, _ in _ACCEPTED], *_REJECTED]      # accepted + rejected: the whole boundary
@pytest.mark.parametrize("reason,value", _EVERY_VALUE, ids=[r for r, _ in _EVERY_VALUE])
def test_entity_write_agrees_with_the_batch_write_path(reason, value):
    """THE rule, stated once: `e.field = v` accepts exactly what `qr.field = v` accepts, and writes the same thing.

    Pinned as a parity test rather than a hand-copied list of cases so the two paths cannot drift apart. Only
    inputs whose meaning is the same for one entity and for a column are in the table -- the (N, *e) positional
    form is inherently a batch concept and has no single-entity reading."""
    world_e, eid = _world_with_one()
    world_q, _ = _world_with_one()

    entity_err = qr_err = None
    try:
        world_e.get_entity(eid).position = value
    except Exception as ex:                                  # noqa: BLE001 -- the type is the thing under test
        entity_err = type(ex)
    try:
        world_q.query(HasPosition).position = value
    except Exception as ex:                                  # noqa: BLE001
        qr_err = type(ex)

    assert (entity_err is None) == (qr_err is None), f"entity: {entity_err}, batch: {qr_err}"
    if entity_err is None:                                   # both accepted -> same bytes in the pool
        np.testing.assert_array_equal(_row(world_e, eid, "position"), _row(world_q, eid, "position"))


@pytest.mark.parametrize("reason,value", _REJECTED, ids=[r for r, _ in _REJECTED])
def test_entity_write_rejected_value_leaves_the_pool_untouched(reason, value):
    """issue (3): what numpy refuses must raise BEFORE the row is touched -- there is no buffer to fall back on
    any more, so a check that runs after the write would hand the caller an exception over a dirty pool."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises((ValueError, TypeError)):
        e.position = value

    np.testing.assert_array_equal(_row(world, eid, "position"), [1.0, 2.0])


def test_entity_write_object_field_takes_a_bare_python_object():
    """The ergonomic win of following numpy: an object-dtype field takes the object itself, no np.array() wrapper."""
    world = World([HasLabel])
    eid = world.add_entity((HasLabel,), label=np.array([{"v": 0}], dtype=object))
    world.update()

    payload = {"v": 42}
    world.get_entity(eid).label = payload

    assert _row(world, eid, "label")[0] is payload                  # stored by reference, not copied


_UNKNOWN_WRITES = [("no component's field", "bogus"), ("a field this entity lacks", "velocity")]
@pytest.mark.parametrize("reason,name", _UNKNOWN_WRITES, ids=[r for r, _ in _UNKNOWN_WRITES])
def test_entity_write_unknown_field_raises_naming_the_valid_fields(reason, name):
    """issue (2): the error names the offending field AND this entity's fields -- plan 2 finding 15. It must
    never land as a silent dead instance attr on the view (which the next read would then happily return)."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError) as exc:
        setattr(e, name, np.array([3.0, 4.0], "float32"))

    assert name in str(exc.value) and "position" in str(exc.value)   # the bad name and the valid set
    assert name not in vars(e)                                       # not grafted onto the instance


def test_entity_write_cannot_add_a_new_internal_attr():
    """Only the known internal attrs route to super(): user code must not be able to graft state onto a view."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        e._my_cache = {"anything": 1}


_ENTITY_METHODS = sorted(n for n in vars(Entity) if not n.startswith("__"))   # set_data / get_fields / to_dict / ...
@pytest.mark.parametrize("method_name", _ENTITY_METHODS)
def test_entity_write_cannot_shadow_a_method(method_name):
    """A method name is not a field: assigning to it must raise, leaving the method intact and callable -- never
    silently replace the bound method with an array that blows up as 'not callable' at the next call site."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        setattr(e, method_name, np.array([0.0, 0.0], "float32"))
    assert callable(getattr(e, method_name))


def test_entity_zero_dim_field_accumulates_but_cannot_be_sliced():
    """A shape-() field reads back as a numpy SCALAR (its column is (N,), so a row is one element).

    `+=` works: the scalar is immutable, so numpy makes a NEW one and hands it to __setattr__, which accepts it
    like any other numpy-convertible value. `e.scale[...] = v` cannot work -- a scalar is not subscriptable -- but
    that is a property of the READ, not of the write path, and it is the same before and after #42."""
    world = World([HasScale])
    eid = world.add_entity((HasScale,), scale=np.array(2.5, "float32"))
    world.update()
    e = world.get_entity(eid)

    e.scale += np.float32(1.5)                               # rebind, not in-place -- and it composes
    e.scale += np.float32(1.0)
    np.testing.assert_array_equal(_row(world, eid, "scale"), 5.0)

    with pytest.raises((TypeError, ValueError)):
        e.scale[...] = 4.0                                   # numpy scalar: item assignment unsupported
    np.testing.assert_array_equal(_row(world, eid, "scale"), 5.0)


# --- not committed yet: reads and writes agree --------------------------------------------------------------------
# get_entity returns a handle before update() (the id is live from add_entity), but the row does not exist yet.
# Dev's call, 2026-07-26: a write raises exactly like a read, rather than patching the staged ADD_ENTITY command.
# The rule stays "a write goes where the row is" -- with no row, there is nowhere to go.

# Every public entry point that needs a row must fail the SAME way (plan 2 finding 1; landed untracked as the
# `_locate` follow-up to #42 -- do not read the old "#43" here, that number is a different task now). Four used
# to index `_eid_to_pool_ix` directly and raised a bare `KeyError: <id>` -- a number with no context, and
# reachable without any stale handle, because `add_entity` publishes the id in `world.live_entities` (a public
# dict) before the row exists. Routing `get_components` / `get_fields` through `_locate` fixed all four
# (`has_component` and `to_dict` call through them). Note the nesting that creates: `_locate`'s error message
# calls `get_components()`, which calls `_locate()` -- safe ONLY because `names=[]` can never fail the field
# check. If that message ever grows a call that can fail, this becomes infinite recursion.

_NEEDS_A_ROW = {
    "read":           lambda e: e.position,
    "write":          lambda e: setattr(e, "position", np.array([9.0, 8.0], "float32")),
    "set_data":       lambda e: e.set_data(position=np.array([9.0, 8.0], "float32")),
    "get_components": lambda e: e.get_components(),
    "get_fields":     lambda e: e.get_fields(),
    "has_component":  lambda e: e.has_component(HasPosition),
    "to_dict":        lambda e: e.to_dict(),
}


def _uncommitted_spawn():
    """A handle to an entity that is a live id with no row yet: add_entity, no update()."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    return world.get_entity(eid), eid


def _committed_despawn():
    """A handle kept across a COMMITTED remove_entity: the id is gone from live_entities and has no row."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()
    e = world.get_entity(eid)
    world.remove_entity(eid)
    world.update()
    return e, eid


@pytest.mark.parametrize("state", ["uncommitted spawn", "committed despawn"])
@pytest.mark.parametrize("entry", list(_NEEDS_A_ROW))
def test_entity_without_a_row_raises_attributeerror_everywhere(entry, state):
    """No row ⇒ `AttributeError` naming the entity, from every entry point and in both states that lack a row.

    `AttributeError` and not `KeyError` for two reasons: it is the protocol Python expects from attribute
    access (`hasattr`, `copy`, `pickle` all probe with it), and a bare `KeyError: 0` tells the caller nothing
    about which object refused or why."""
    e, eid = (_uncommitted_spawn if state == "uncommitted spawn" else _committed_despawn)()

    with pytest.raises(AttributeError) as exc:
        _NEEDS_A_ROW[entry](e)

    assert str(eid) in str(exc.value)          # names the entity, so the message is actionable


# The two states get the SAME message, on purpose (plan 2 finding 4, closed 2026-07-26). The proposal was to
# branch on `live_entities` and say "not committed yet" vs "removed"; the dev's call was one sentence that is
# true either way, with the `update()` advice made conditional. The old message asserted "not committed yet"
# as fact, which is a lie to a despawned handle -- that is what got fixed. Which of the two states you are in
# is something the caller already knows, so there is no behaviour left here to test beyond the parametrized
# case above; the rest is wording, and a test that pins wording is a test that breaks on a reword.


def test_entity_field_read_before_update_raises_attributeerror():
    """A field read on a not-yet-committed spawn raises AttributeError (not a raw KeyError on the id)."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))   # no update()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        _ = e.position


def test_entity_field_write_before_update_raises_attributeerror():
    """And so does a write: same error, same reason. Spawn data goes in add_entity's kwargs, not a pre-commit write."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))   # no update()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        e.position = np.array([9.0, 8.0], "float32")

    world.update()
    np.testing.assert_array_equal(e.position, [1.0, 2.0])            # the spawn value, unaffected by the refusal


def test_entity_write_after_commit_of_the_spawn_works():
    """The very next tick is fine -- the refusal above is about the missing row, not about the entity being new."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    e = world.get_entity(eid)
    world.update()

    e.position = np.array([9.0, 8.0], "float32")                     # same handle, now backed by a row

    np.testing.assert_array_equal(_row(world, eid, "position"), [9.0, 8.0])


# --- live view: resolves the CURRENT row, across swaps and migrations ---------------------------------------------

def test_entity_write_targets_correct_row_after_swap_remove():
    """Writes resolve by id, not index: after a swap-remove relocates rows, the write hits the moved entity's
    current row -- never a neighbour's."""
    world = World(components=[HasPosition, HasVelocity])
    a = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))  # idx 0
    b = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))  # idx 1
    c = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))  # idx 2 (tail)
    world.update()
    world.remove_entity(a)                               # c swaps into slot 0; b stays at slot 1
    world.update()

    world.get_entity(c).position = np.array([20.0, 20.0], "float32")

    np.testing.assert_array_equal(world.get_entity(c).position, [20.0, 20.0])       # c got the write
    np.testing.assert_array_equal(world.get_entity(b).position, [1.0, 1.0])         # neighbour untouched


def test_entity_view_is_live_across_archetype_migration():
    """A held Entity is a view, not a snapshot: after add_component migrates it to another pool, the SAME view
    still reads the (relocated) data, reports the new component set, and can write the newly-added field."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([5.0, 6.0], "float32"))
    world.update()

    e = world.get_entity(eid)                            # held BEFORE the migration
    assert set(e.get_components()) == {HasPosition}

    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([1.0, 1.0], "float32"))  # -> pos+vel pool
    world.update()

    np.testing.assert_array_equal(e.position, [5.0, 6.0])           # data survived the move, same view
    assert set(e.get_components()) == {HasPosition, HasVelocity}    # view sees the new archetype
    assert "velocity" in e.get_fields()

    e.velocity = np.array([7.0, 7.0], "float32")                    # write the newly-available field via old view
    np.testing.assert_array_equal(_row(world, eid, "velocity"), [7.0, 7.0])


def test_entity_write_to_a_field_added_this_tick_raises_until_update():
    """The other half of the rule: the component is only PENDING, so its column does not exist yet and the write
    has nowhere to land. A write never consults the command buffer (dev's call) -- so it raises, and the same
    write succeeds one update() later."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()
    e = world.get_entity(eid)

    e.add_component(HasVelocity, velocity=np.zeros(2, "float32"))    # staged, not applied
    with pytest.raises(AttributeError):
        e.velocity = np.array([5.0, 6.0], "float32")

    world.update()
    e.velocity = np.array([5.0, 6.0], "float32")                     # the column exists now
    np.testing.assert_array_equal(e.velocity, [5.0, 6.0])


def test_entity_write_to_a_field_pending_removal_lands_then_is_dropped():
    """Mirror case: the component is leaving but its column is still there, so the write lands NOW -- and then
    update() removes the component and the data goes with it. Consistent, and needs no buffer scan to explain."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity((HasPosition, HasVelocity), position=np.array([1.0, 2.0], "float32"),
                           velocity=np.array([3.0, 4.0], "float32"))
    world.update()
    e = world.get_entity(eid)

    e.remove_component(HasVelocity)                                  # staged
    e.velocity = np.array([5.0, 6.0], "float32")                     # accepted: the column is still live
    np.testing.assert_array_equal(e.velocity, [5.0, 6.0])

    world.update()

    with pytest.raises(AttributeError):
        _ = e.velocity                                               # gone with the component
    np.testing.assert_array_equal(e.position, [1.0, 2.0])            # the surviving field is intact


# --- stale handle across remove: every op must reject EAGERLY ------------------------------------------------------
# A handle held BEFORE remove_entity goes stale once the id is despawned. get_entity guards a *fresh* lookup (a dead
# id raises there), but a *stale* handle bypasses that path. Structural ops through it are rejected by the single
# gate in CommandBuffer.append (appending a command for a non-live id raises). A DATA write has no gate any more --
# the row is already gone from _eid_to_pool_ix, so resolution itself must refuse it.

def test_stale_handle_add_component_after_remove_rejects_eagerly():
    """A handle kept across remove_entity must reject add_component at the call, not queue a corrupting command."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)                            # valid handle, taken before despawn
    world.remove_entity(eid)                             # eid leaves live_entities -> handle is now stale

    with pytest.raises(ValueError):
        e.add_component(HasVelocity, velocity=np.array([1.0, 1.0], "float32"))   # valid comp -> only liveness can raise


def test_stale_handle_remove_component_after_remove_rejects_eagerly():
    """Same guard for the narrowing op: remove_component through a stale handle must reject eagerly."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity((HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()
    e = world.get_entity(eid)                            # valid handle, taken before despawn
    world.remove_entity(eid)                             # now stale

    with pytest.raises(ValueError):
        e.remove_component(HasVelocity)


def test_stale_handle_data_write_after_committed_remove_raises():
    """Once the despawn is committed the id has no row, so a write through the stale handle raises rather than
    poking whatever entity was swapped into that slot."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    other = world.add_entity((HasPosition,), position=np.array([3.0, 4.0], "float32"))   # swaps into eid's slot
    world.update()
    e = world.get_entity(eid)
    world.remove_entity(eid)
    world.update()

    with pytest.raises((AttributeError, KeyError, ValueError)):
        e.position = np.array([9.0, 8.0], "float32")

    np.testing.assert_array_equal(world.get_entity(other).position, [3.0, 4.0])   # the neighbour is untouched


def test_stale_handle_data_write_before_the_remove_commits_cannot_corrupt_a_neighbour():
    """While the despawn is only STAGED the row still exists, and #42 puts no gate on the data path -- so the
    write may well land in the doomed row. Either way the invariant is the same: it touches nothing but that row,
    so the entity that swap-removes into the slot still carries its own value afterwards."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    other = world.add_entity((HasPosition,), position=np.array([3.0, 4.0], "float32"))
    world.update()
    e = world.get_entity(eid)
    world.remove_entity(eid)                             # staged, not committed: eid still has a row

    try:
        e.position = np.array([9.0, 8.0], "float32")     # lands in the doomed row, or raises -- both legal
    except (AttributeError, KeyError, ValueError):
        pass

    world.update()
    np.testing.assert_array_equal(world.get_entity(other).position, [3.0, 4.0])   # no collateral damage


def test_stashed_row_view_across_update_is_the_documented_hole():
    """#42's one accepted loss, pinned so it stays visible instead of silent.

    A read hands back a writable view into pool memory, so a view STASHED across an `update()` writes into
    whatever row now sits at that index -- here B's, after B swap-removes into A's slot. This is the pre-#29
    status quo robosim ran on for months, and it is the same rule queries already have (#27): **do not hold a
    view across `update()`**. If a future guard ever makes this raise, delete this test -- the guarantee is back."""
    world = World([HasPosition])
    a = world.add_entity((HasPosition,), position=np.array([1.0, 1.0], "float32"))   # row 0
    b = world.add_entity((HasPosition,), position=np.array([2.0, 2.0], "float32"))   # row 1
    world.update()
    stashed = world.get_entity(a).position               # a writable view of row 0

    world.remove_entity(a)
    world.update()                                       # b's row is copied into slot 0
    stashed[:] = [9.0, 9.0]                              # still row 0 -- which is B's row now

    np.testing.assert_array_equal(world.get_entity(b).position, [9.0, 9.0])   # B was hit, and nothing warned


# --- serialization: entity.to_dict() ------------------------------------------------------------------------------
# to_dict() returns {"components": [class names], "data": {field: json-friendly value}}, read from the CURRENT row.
# Arrays -> .tolist(); object-dtype fields -> .item() (the raw python object). With serialization_field set, only
# fields whose metadata[serialization_field] is True are dumped.


def test_to_dict_dumps_components_and_all_fields():
    """No filter: every field of every component, values as plain lists, components by class name."""
    world = World([HasPosition, HasVelocity])
    eid = world.add_entity((HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    d = world.get_entity(eid).to_dict()

    assert set(d["components"]) == {"HasPosition", "HasVelocity"}
    assert d["data"] == {"position": [1.0, 2.0], "velocity": [3.0, 4.0]}


def test_to_dict_is_json_serializable():
    """The point of to_dict: the result round-trips through json -- no ndarray leaks into 'data'."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    restored = json.loads(json.dumps(world.get_entity(eid).to_dict()))
    assert restored["data"]["position"] == [1.0, 2.0]


def test_to_dict_object_dtype_uses_item_not_tolist():
    """An object-dtype field serializes via .item() -> the stored python object itself, not a [list]."""
    world = World([HasLabel])
    payload = {"hp": 7, "name": "goblin"}
    eid = world.add_entity((HasLabel,), label=np.array([payload], dtype=object))
    world.update()

    assert world.get_entity(eid).to_dict()["data"]["label"] == payload


def test_to_dict_filters_out_non_serializable_fields():
    """With serialization_field set, a field marked False is dropped; the True one survives."""
    world = World([HasSerial], extra_metadata=["serializable"])
    eid = world.add_entity((HasSerial,), keep=np.array([1.0], "float32"), drop=np.array([9.0], "float32"))
    world.update()

    d = world.get_entity(eid).to_dict(serialization_field="serializable")
    assert "keep" in d["data"] and "drop" not in d["data"]


def test_to_dict_without_filter_dumps_even_non_serializable_fields():
    """No serialization_field -> filter off, so even a serializable=False field is dumped."""
    world = World([HasSerial], extra_metadata=["serializable"])
    eid = world.add_entity((HasSerial,), keep=np.array([1.0], "float32"), drop=np.array([9.0], "float32"))
    world.update()

    d = world.get_entity(eid).to_dict()
    assert "keep" in d["data"] and "drop" in d["data"]


def test_to_dict_reflects_the_write_immediately():
    """to_dict reads the live row and writes are eager, so a dump taken right after a write already shows it --
    no update() in between (under #29 the dump showed the previous tick's value)."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position = np.array([7.0, 8.0], "float32")

    assert e.to_dict()["data"]["position"] == [7.0, 8.0]


# --- copy / pickle: an Entity is a handle, not a value (#45 C3) ----------------------------------------------------
# An Entity means "row ix of pool p RIGHT NOW". Its internals are a world-owned dict and the world's command buffer,
# so a shallow copy aliases the whole world and a deep copy clones it -- neither is what "a copy of this entity"
# should mean. to_dict() is. So both are refused, via __reduce__ ON THE CLASS.
#
# It has to be a class-level dunder, and the two failure modes it replaces are why:
#   * copy/deepcopy went into INFINITE RECURSION. copy._reconstruct builds the instance with cls.__new__(cls) -- no
#     __init__, so no _eid_to_pool_ix on it. Probing a missing internal hits __getattr__, which reads
#     self._eid_to_pool_ix, also missing -> __getattr__ again, forever. Pool and QueryResult each dodge this with a
#     self.__dict__.get(...) guard inside their own dunders; Entity cannot afford one (+45 ns on EVERY field touch,
#     #45), and __reduce__ costs nothing because a field touch never looks that name up.
#   * pickle.dumps SUCCEEDED (py3.12: object.__getstate__ just hands over the instance dict, nothing probes a
#     missing attr) -- quietly serializing _eid_to_pool_ix, i.e. every pool in the world, into "one entity". The
#     silent version is the worse bug of the two; TypeError is the fix for both.
# Regression note: the guard is only real if it is a method OF Entity. Nested one level too deep inside __setattr__
# it is a throwaway local -- copy still recursed, and every write paid to build the function object. Hence a
# behaviour assertion here rather than a hasattr() check.

@pytest.mark.parametrize("op", [copy.copy, copy.deepcopy, pickle.dumps], ids=["copy", "deepcopy", "pickle"])
def test_entity_cannot_be_copied_or_pickled(op):
    """copy / deepcopy / pickle raise TypeError -- not RecursionError, and not a silent clone of the whole world."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(TypeError, match="cannot be copied or pickled"):
        op(e)


def test_entity_deepcopy_refusal_propagates_out_of_a_container():
    """The refusal is not something a caller can route around by nesting the handle in a list/dict."""
    world, eid = _world_with_one()

    with pytest.raises(TypeError, match="cannot be copied or pickled"):
        copy.deepcopy({"entities": [world.get_entity(eid)]})


def test_entity_stays_usable_after_a_refused_copy():
    """__reduce__ raising must not leave the view damaged -- reads and eager writes still work on the same handle."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)
    with pytest.raises(TypeError):
        copy.copy(e)

    e.position = np.array([7.0, 8.0], "float32")     # the handle is untouched: it is still row ix of its pool

    np.testing.assert_array_equal(_row(world, eid, "position"), [7.0, 8.0])
