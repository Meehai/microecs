"""Unit tests for microecs.Entity -- the object-like view returned by world.get_entity(id).

Entity is a LIVE view, not a snapshot: every attribute access re-resolves (pool, row) from the id, so it
stays correct across pool changes (swap-remove, archetype migration).

**READS return the row. WRITES go through `set_data(**fields)` -- and nothing else (#29).** The three old
idioms are all gone:
  * `e.field = v`    -> raises (`__setattr__` only routes Entity's own internal attrs)
  * `e.field[:] = v` -> raises (`__getattr__` hands back a READ-ONLY view)
  * `e.field += v`   -> raises, and must NOT have already mutated the pool (the in-place op happens first)
Rationale: entity-level mutation is *always buffered* (applied at `world.update()`), like add/remove_component.
The eager idioms poked the pool immediately, so when a write landed depended on which idiom you reached for.

`set_data`'s own spec (validation, atomicity, deferral) lives in test_entity_set_data.py. Here we pin the view:
reads, the forbidden writes, liveness across pool churn, and to_dict.

Issues these pin:
  (1) one write path  -- every eager idiom raises, and a *failed* write leaves the pool untouched.
  (2) bare error      -- a bad field name must name the field and the valid set.
  (3) internal attrs  -- Entity's own instance attrs must all be in the __setattr__ allowlist, or the class
                         cannot even be constructed once __setattr__ starts refusing field writes.
"""
from dataclasses import field
import json
import numpy as np
import pytest

from microecs import World, Component, Entity
from microecs.entity import _ENTITY_INTERNAL_ATTRS, ENTITY_RESERVED_NAMES


class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasLabel(Component):  # object dtype: an arbitrary python object per entity (-> to_dict uses .item())
    label: np.ndarray = field(metadata={"shape": (1,), "dtype": "object", "default": None})


class HasScale(Component):  # 0-d array field: exactly one scalar per entity (shape ())
    scale: np.ndarray = field(metadata={"shape": (), "dtype": "float32", "default": None})


class HasPose(Component):   # (4, 4) field: the sliced-read case (e.pose[0:3, 3]) must keep working
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


# --- construction: the __setattr__ allowlist must cover Entity's own attrs -----------------------------------------
# __setattr__ now REFUSES everything except _ENTITY_INTERNAL_ATTRS, so __init__ itself goes through the guard.
# Any instance attr __init__ sets that is missing from the allowlist makes Entity unconstructable -> get_entity
# raises for every entity in the world. The allowlist must also feed ENTITY_RESERVED_NAMES, else a component
# field could be named like an internal attr and shadow it.

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
    assert e.get_fields() == ["position"]
    assert e.entity_id == eid


def test_entity_unknown_field_read_raises_named_error():
    """Reading a field the entity's pool doesn't have raises AttributeError naming the field and the valid set."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError, match="velocity"):
        _ = e.velocity                                   # not a field of this (HasPosition-only) entity


def test_entity_sliced_read_still_works():
    """Making the view read-only must not break READS -- including sliced/fancy reads into a (4, 4) field."""
    world = World([HasPose])
    pose = np.arange(16, dtype="float32").reshape(4, 4)
    eid = world.add_entity((HasPose,), pose=pose.copy())
    world.update()
    e = world.get_entity(eid)

    np.testing.assert_array_equal(e.pose[0:3, 3], pose[0:3, 3])     # the robosim translation-read idiom
    np.testing.assert_array_equal(e.pose[0], pose[0])
    assert float(e.pose[1, 1]) == pose[1, 1]
    np.testing.assert_array_equal(e.pose.copy(), pose)              # a copy of a read-only view is fine


def test_entity_read_is_a_live_view_not_a_copy():
    """Read-only must not mean detached: a bulk column write through a query is visible on the entity view."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    world.query(HasPosition).position = np.array([[7.0, 7.0]], "float32")   # the vectorised write path

    np.testing.assert_array_equal(e.position, [7.0, 7.0])           # same underlying buffer, no stale copy


# --- one write path: every eager idiom raises AND leaves the pool untouched: issue (1) -----------------------------

def test_entity_attribute_write_raises_and_routes_to_set_data():
    """`e.field = v` must raise (naming set_data) instead of poking the pool behind the buffer's back."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError, match="set_data"):
        e.position = np.array([9.0, 8.0], "float32")

    np.testing.assert_array_equal(world.get_entity(eid).position, [1.0, 2.0])   # unchanged
    assert len(world._command_buffer) == 0                                      # and nothing staged either


def test_entity_slice_write_raises_on_read_only_view():
    """`e.field[:] = v` -- the in-place idiom -- must raise: __getattr__ returns a read-only view."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(ValueError):                                  # numpy: assignment destination is read-only
        e.position[:] = np.array([9.0, 8.0], "float32")

    np.testing.assert_array_equal(world.get_entity(eid).position, [1.0, 2.0])   # pool untouched


def test_entity_element_write_raises_on_read_only_view():
    """Same for a single element: `e.field[0] = v` cannot sneak past the read-only view."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(ValueError):
        e.position[0] = 9.0

    np.testing.assert_array_equal(world.get_entity(eid).position, [1.0, 2.0])


def test_entity_inplace_add_raises_without_half_writing_the_pool():
    """`e.field += v` is the nastiest one: numpy mutates the row IN PLACE, then __setattr__ raises.

    With a writable view the caller sees an exception but the pool is ALREADY dirty (verified in
    test/manual/29-entity-set-data/probe.py: [1, 2] -> [11, 12] on a raising `+=`). A read-only view makes
    the in-place op raise FIRST, so a failed write changes nothing."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises((ValueError, AttributeError)):
        e.position += np.array([10.0, 10.0], "float32")

    np.testing.assert_array_equal(world.get_entity(eid).position, [1.0, 2.0])   # NOT [11.0, 12.0]


def test_entity_object_field_slice_write_raises():
    """An object-dtype row is a view too -- `e.label[0] = obj` must raise, not swap the reference in."""
    world = World([HasLabel])
    original = {"v": 0}
    eid = world.add_entity((HasLabel,), label=np.array([original], dtype=object))
    world.update()
    e = world.get_entity(eid)

    with pytest.raises(ValueError):
        e.label[0] = {"v": 42}

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool.label[ix, 0] is original                             # reference not swapped


def test_entity_zero_dim_field_write_raises():
    """A shape-() field comes back as a scalar (already immutable); the `= v` route must raise as well."""
    world = World([HasScale])
    eid = world.add_entity((HasScale,), scale=np.array(2.5, "float32"))
    world.update()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        e.scale = np.array(4.0, "float32")
    with pytest.raises((ValueError, TypeError)):
        e.scale[...] = 4.0

    np.testing.assert_array_equal(world.get_entity(eid).scale, 2.5)


def test_entity_unknown_field_write_raises():
    """Writing a name that is not a field raises too -- never a silent dead instance attr on the view."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        e.velocity = np.array([3.0, 4.0], "float32")     # velocity not in this pool
    assert "velocity" not in vars(e)                     # and it did not land as an instance attr


def test_entity_write_cannot_add_a_new_internal_attr():
    """Only the known internal attrs route to super(): user code must not be able to graft state onto a view."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        e._my_cache = {"anything": 1}


# --- live view: resolves the CURRENT row, across swaps and migrations ---------------------------------------------

def test_entity_set_data_targets_correct_row_after_swap_remove():
    """Writes resolve by id, not index: after a swap-remove relocates rows, the write hits the moved entity's
    current row -- never a neighbour's."""
    world = World(components=[HasPosition, HasVelocity])
    a = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))  # idx 0
    b = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))  # idx 1
    c = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))  # idx 2 (tail)
    world.update()
    world.remove_entity(a)                               # c swaps into slot 0; b stays at slot 1
    world.update()

    world.get_entity(c).set_data(position=np.array([20.0, 20.0], "float32"))
    world.update()

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

    e.set_data(velocity=np.array([7.0, 7.0], "float32"))            # write the newly-available field via old view
    world.update()
    np.testing.assert_array_equal(world.get_entity(eid).velocity, [7.0, 7.0])


# --- stale handle across remove: structural ops must reject EAGERLY -----------------------------------------------
# A handle held BEFORE remove_entity goes stale once the id is despawned. get_entity guards a *fresh* lookup (a dead
# id raises there), but a *stale* handle bypasses that path. add_component / remove_component through it must still
# reject at the CALL -- not silently queue a command that corrupts the pool inside update(). The guard now lives in
# CommandBuffer.append (the single gate every command passes through): appending a command for a non-live id raises
# ValueError, so the op is rejected eagerly and nothing is staged. Component passed is VALID, so the only thing that
# can raise is the liveness guard. Satisfiability proven in test/manual/add-component-on-entity/verify_fixes.py.

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


def test_to_dict_reflects_current_pool_value():
    """to_dict reads the live row, so a committed set_data shows up in the next dump (not a stale snapshot)."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.set_data(position=np.array([7.0, 8.0], "float32"))
    assert e.to_dict()["data"]["position"] == [1.0, 2.0]    # buffered: the dump still shows the old row
    world.update()
    assert e.to_dict()["data"]["position"] == [7.0, 8.0]    # applied


# --- bug: a field-write must not be able to shadow Entity's own methods --------------------------------------------
# `e.<method> = arr` should raise (a method name is not a field), not silently replace the bound method with an array
# (which then blows up at call time as 'not callable').

_ENTITY_METHODS = sorted(n for n in vars(Entity) if not n.startswith("__"))   # set_data / get_fields / to_dict / ...
@pytest.mark.parametrize("method_name", _ENTITY_METHODS)
def test_entity_write_cannot_shadow_a_method(method_name):
    """Assigning to one of Entity's own method names must raise, leaving the method intact and callable."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        setattr(e, method_name, np.array([0.0, 0.0], "float32"))
    assert callable(getattr(e, method_name))            # method still bound, not replaced by the array


# --- bug: field access on an uncommitted spawn must raise a clear AttributeError, not a raw KeyError ---------------
# get_entity returns a handle before update() (the Entity is built at add_entity), but the row isn't committed yet.
# A field read must raise AttributeError -- protocol-correct (hasattr/copy rely on it) and clearer than KeyError on
# the id.

def test_entity_field_read_before_update_raises_attributeerror():
    """A field read on a not-yet-committed spawn raises AttributeError (not a raw KeyError on the id)."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))   # no update()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        _ = e.position
