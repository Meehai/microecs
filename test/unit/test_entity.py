"""Unit tests for microecs.Entity -- the object-like view returned by world.get_entity(id).

Entity is a LIVE view, not a snapshot: every attribute access re-resolves (pool, row) from the id, so it
stays correct across pool changes (swap-remove, archetype migration). Reads come back as the entity's row;
writes (e.field = .. and e.field += ..) scatter straight into the pool buffer (eager, like set_entity_data).
A name that is not one of the entity's current fields raises AttributeError on both read and write.

Issues these pin:
  (1) silent lost write -- `e.field = x` must reach the pool, not set a dead instance attribute.
  (2) bare error      -- a bad field name must name the field and the valid set, on read AND write.
"""
from dataclasses import field
import json
import numpy as np
import pytest

from microecs import World, Component, Entity


class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasLabel(Component):  # object dtype: an arbitrary python object per entity (-> to_dict uses .item())
    label: np.ndarray = field(metadata={"shape": (1,), "dtype": "object"})


class HasSerial(Component):  # two fields; the 'serializable' extra-metadata drives to_dict's filter
    keep: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "serializable": True})
    drop: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "serializable": False})


def _world_with_one(position=(1.0, 2.0)):
    """A committed world holding a single HasPosition entity. Returns (world, entity_id)."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array(position, "float32"))
    world.update()
    return world, eid


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


# --- write-through: issue (1) -------------------------------------------------------------------------------------

def test_entity_attribute_write_persists_to_pool_and_query():
    """`e.field = value` must land in the pool buffer -- visible through both get_entity and a fresh query --
    not silently set a dead instance attribute on the view."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position = np.array([9.0, 8.0], "float32")

    np.testing.assert_array_equal(world.get_entity(eid).position, [9.0, 8.0])       # round-trips by id
    np.testing.assert_array_equal(world.query(HasPosition).position.numpy()[0], [9.0, 8.0])  # query sees it
    pool, ix = world._eid_to_pool_ix[eid]
    np.testing.assert_array_equal(pool.position[ix], [9.0, 8.0])                     # actual pool row overwritten


def test_entity_inplace_write_persists():
    """`e.field += value` mutates the pool row in place (the ufunc writes through the returned view)."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position += np.array([1.0, 1.0], "float32")

    np.testing.assert_array_equal(world.get_entity(eid).position, [2.0, 3.0])


def test_entity_write_is_eager_visible_without_update():
    """Unlike add/remove (command-buffered), an Entity write is a direct pool write: visible immediately, with
    nothing queued on the command buffer."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position = np.array([5.0, 5.0], "float32")

    assert world._command_buffer == []                                              # nothing queued
    np.testing.assert_array_equal(world.get_entity(eid).position, [5.0, 5.0])       # no second update() needed


def test_entity_write_copies_value_not_aliases():
    """Slot-assignment copies into the pre-allocated buffer: mutating the source afterwards must not leak in."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    src = np.array([5.0, 6.0], "float32")
    e.position = src
    src[:] = [999.0, 999.0]                                                         # mutate source after the write

    np.testing.assert_array_equal(world.get_entity(eid).position, [5.0, 6.0])       # stored value is independent


def test_entity_unknown_field_write_raises_named_error():
    """Writing a name that is not one of the entity's fields raises AttributeError -- never a silent dead attr."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError, match="velocity"):
        e.velocity = np.array([3.0, 4.0], "float32")     # velocity not in this pool


# --- live view: resolves the CURRENT row, across swaps and migrations ---------------------------------------------

def test_entity_write_targets_correct_row_after_swap_remove():
    """Writes resolve by id, not index: after a swap-remove relocates rows, writing through the view hits the
    moved entity's current row -- never a neighbour's."""
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

    world.add_component(eid, HasVelocity, velocity=np.array([1.0, 1.0], "float32"))  # -> pos+vel pool
    world.update()

    np.testing.assert_array_equal(e.position, [5.0, 6.0])           # data survived the move, same view
    assert set(e.get_components()) == {HasPosition, HasVelocity}    # view sees the new archetype
    assert "velocity" in e.get_fields()

    e.velocity = np.array([7.0, 7.0], "float32")                    # write the newly-available field via old view
    np.testing.assert_array_equal(world.get_entity(eid).velocity, [7.0, 7.0])


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
    """to_dict reads the live row, so a write through the view shows up in the next dump (not a stale snapshot)."""
    world, eid = _world_with_one()
    e = world.get_entity(eid)

    e.position = np.array([7.0, 8.0], "float32")
    assert e.to_dict()["data"]["position"] == [7.0, 8.0]


# --- bug: a field-write must not be able to shadow Entity's own methods --------------------------------------------
# `e.<method> = arr` should raise (a method name is not a field), not silently replace the bound method with an array
# (which then blows up at call time as 'not callable'). RED until __setattr__ stops routing method names to super().

_ENTITY_METHODS = sorted(n for n in vars(Entity) if not n.startswith("__"))   # get_components / get_fields / to_dict
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
# the id. RED until Entity.__getattr__ guards the missing row.

def test_entity_field_read_before_update_raises_attributeerror():
    """A field read on a not-yet-committed spawn raises AttributeError (not a raw KeyError on the id)."""
    world = World([HasPosition])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))   # no update()
    e = world.get_entity(eid)

    with pytest.raises(AttributeError):
        _ = e.position
