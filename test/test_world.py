"""Unit tests for ecs.World"""
from dataclasses import field
import numpy as np
import pytest

from ecs import World, Component


class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasRadius(Component):
    radius: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32"})


def test_add_entity_rejects_field_from_an_unrequested_component():
    """An entity declared with only HasPosition may not pass `velocity` (a field of the unrequested HasVelocity)."""
    world = World(components=[HasPosition, HasVelocity])  # both components known to the world

    with pytest.raises(AssertionError, match="velocity"):
        world.add_entity(
            components=(HasPosition,),                       # entity declares HasPosition only
            position=np.array([1.0, 2.0], "float32"),       # required by HasPosition
            velocity=np.array([3.0, 4.0], "float32"),       # extra: belongs to HasVelocity, not requested
        )


def test_fresh_world_has_no_pools():
    """A world creates pools lazily; before any add_entity there are none."""
    world = World(components=[HasPosition, HasVelocity])
    assert world.pools == {}


def test_add_one_entity_creates_exactly_one_pool_with_one_entity():
    """First add_entity creates a single pool, holding that single entity."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    assert len(world.pools) == 1
    pool = next(iter(world.pools.values()))
    assert len(pool) == 1


def test_entity_lands_in_the_pool_keyed_by_its_components():
    """The entity goes into the pool whose key is exactly the bitmask of its declared components."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    key = world._make_key((HasPosition,))
    assert key in world.pools                                   # a pool with that exact key exists
    assert world.pools[key] is world.query_and((HasPosition,))[0]  # and query_and finds the same pool


def test_added_entity_field_values_are_stored():
    """The component data we passed is readable back from the pool, unchanged."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.5, 2.5], "float32"))

    pool = world.query_and((HasPosition,))[0]
    np.testing.assert_array_equal(pool.position[0], np.array([1.5, 2.5], "float32"))


def test_same_archetype_entities_share_a_single_pool():
    """N entities with the same component set all land in one pool, in insertion order."""
    world = World(components=[HasPosition, HasVelocity])

    for i in range(3):
        world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32"))

    assert len(world.pools) == 1                                # still just one archetype
    pool = world.query_and((HasPosition,))[0]
    assert len(pool) == 3
    np.testing.assert_array_equal(pool.position, np.array([[0, 0], [1, 1], [2, 2]], "float32"))


def test_distinct_archetypes_get_distinct_pools():
    """Entities with different component sets are stored in separate pools."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))

    assert len(world.pools) == 2
    assert len(world.query_and((HasPosition,))[0]) >= 1
    pos_vel_pool = world.pools[world._make_key((HasPosition, HasVelocity))]
    assert len(pos_vel_pool) == 1


def test_component_order_does_not_create_a_second_pool():
    """(HasPosition, HasVelocity) and (HasVelocity, HasPosition) are the same archetype."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.add_entity(components=(HasVelocity, HasPosition),
                     position=np.array([5.0, 6.0], "float32"), velocity=np.array([7.0, 8.0], "float32"))

    assert len(world.pools) == 1                                # key is a bitmask, order-independent
    assert len(next(iter(world.pools.values()))) == 2


def test_query_and_returns_all_pools_that_are_supersets():
    """query_and((HasPosition,)) returns every pool containing HasPosition, not just the pos-only one."""
    world = World(components=[HasPosition, HasVelocity, HasRadius])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))

    assert len(world.query_and((HasPosition,))) == 2            # both pools contain HasPosition
    assert len(world.query_and((HasPosition, HasVelocity))) == 1  # only the richer pool has both


def test_query_and_is_empty_when_no_pool_has_the_component():
    """Querying a component that no existing pool carries returns no pools."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    assert world.query_and((HasVelocity,)) == []


def test_entities_are_conserved_across_pools():
    """Summing len over all pools equals the number of entities added."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(HasPosition,), position=np.array([3.0, 4.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([5.0, 6.0], "float32"), velocity=np.array([7.0, 8.0], "float32"))

    assert sum(len(pool) for pool in world.pools.values()) == 3


def test_remove_entity_leaves_empty_pool():
    """Removing the only entity empties its pool and leaves the id bookkeeping consistent (the `else` branch)."""
    world = World(components=[HasPosition])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    pool = world.query_and((HasPosition,))[0]
    assert len(pool) == 1

    world.remove_entity(eid)                                   # last entity out -> pool becomes empty

    assert len(pool) == 0                                      # pool is empty
    assert world._pool_ix_to_eid == {}                         # no dangling (pool, index) -> id mapping
    assert world._eid_to_pool_ix == {}                         # removed id is gone, not pointing at an empty slot


def test_remove_last_index_drops_only_that_entity():
    """Removing the last row (no swap happens) must drop that id, not resurrect it at a now-dead slot."""
    world = World(components=[HasPosition])

    keep = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))  # idx 0
    last = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))  # idx 1 (last)

    world.remove_entity(last)

    pool = world.query_and((HasPosition,))[0]
    assert len(pool) == 1
    assert last not in world._eid_to_pool_ix                   # removed id gone, not pointing at a dead slot
    assert world._eid_to_pool_ix == {keep: (pool, 0)}          # only the survivor remains, at its row
    assert world._pool_ix_to_eid == {(pool, 0): keep}          # reverse map agrees
    np.testing.assert_array_equal(pool.position[0], [1.0, 1.0])


def test_remove_middle_entity_repoints_swapped_id():
    """Removing a middle row swaps the tail into the gap: the tail's id re-points, the removed id vanishes."""
    world = World(components=[HasPosition])

    a = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))  # idx 0
    b = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))  # idx 1 (removed)
    c = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))  # idx 2 (tail)

    world.remove_entity(b)                                     # c slides from slot 2 into slot 1

    pool = world.query_and((HasPosition,))[0]
    assert len(pool) == 2
    assert b not in world._eid_to_pool_ix                      # removed id gone
    assert world._eid_to_pool_ix[a] == (pool, 0)               # a untouched
    assert world._eid_to_pool_ix[c] == (pool, 1)               # c re-pointed to the freed slot
    assert world._pool_ix_to_eid == {(pool, 0): a, (pool, 1): c}  # reverse map consistent
    np.testing.assert_array_equal(pool.position[1], [2.0, 2.0])   # c's data now sits at slot 1


def test_add_entity_returns_unique_ids():
    """Every add_entity hands back a distinct, monotonically increasing id."""
    world = World(components=[HasPosition])

    ids = [world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32")) for i in range(3)]

    assert len(set(ids)) == 3                                   # all distinct
    assert ids == sorted(ids)                                   # monotonic (distinct + sorted => strictly increasing)


def test_id_resolves_after_sibling_removed():
    """Swap-remove moves the tail row; the moved entity's id must still resolve to it, not to its new neighbour."""
    world = World(components=[HasPosition])

    a = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    b = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))

    world.remove_entity(a)                                      # swap-remove: b's row slides into a's old slot
    world.add_entity(components=(HasPosition,), position=np.array([3.0, 3.0], "float32"))  # c lands after b

    world.remove_entity(b)                                      # must drop b ([2,2]), not c, despite the earlier shuffle

    pool = world.query_and((HasPosition,))[0]
    assert len(pool) == 1
    np.testing.assert_array_equal(pool.position[0], [3.0, 3.0])  # only c remains


def test_add_component_moves_entity_and_preserves_fields():
    """add_component widens the archetype: entity leaves the old pool, old field intact, new field set."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_component(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32"))

    assert len(world.pools[world._make_key((HasPosition,))]) == 0           # left the position-only pool
    pos_vel = world.pools[world._make_key((HasPosition, HasVelocity))]
    assert len(pos_vel) == 1
    np.testing.assert_array_equal(pos_vel.position[0], [1.0, 2.0])          # carried-over value intact
    np.testing.assert_array_equal(pos_vel.velocity[0], [3.0, 4.0])          # new value set


def test_add_component_keeps_entity_id():
    """The id is the caller's stable handle (task 02): migrating via add_component must NOT change it."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_component(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32"))

    assert eid in world._eid_to_pool_ix                                     # original id still resolves
    pool, ix = world._eid_to_pool_ix[eid]
    assert pool is world.pools[world._make_key((HasPosition, HasVelocity))]  # now in the richer pool
    np.testing.assert_array_equal(pool.position[ix], [1.0, 2.0])            # carried-over field intact
    np.testing.assert_array_equal(pool.velocity[ix], [3.0, 4.0])            # new field set


def test_remove_component_narrows_archetype():
    """remove_component drops a field and moves the entity to the smaller pool."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.remove_component(eid, HasVelocity)

    assert len(world.pools[world._make_key((HasPosition, HasVelocity))]) == 0  # left the richer pool
    pos_only = world.pools[world._make_key((HasPosition,))]
    assert len(pos_only) == 1
    np.testing.assert_array_equal(pos_only.position[0], [1.0, 2.0])         # kept field survives
    assert not hasattr(pos_only, "velocity")                               # dropped field is gone


def test_add_component_only_needs_new_fields():
    """Caller supplies just the new component's field; existing fields carry over without being re-passed."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([5.0, 6.0], "float32"))
    world.add_component(eid, HasVelocity, velocity=np.array([7.0, 8.0], "float32"))  # position NOT re-passed

    pos_vel = world.pools[world._make_key((HasPosition, HasVelocity))]
    np.testing.assert_array_equal(pos_vel.position[0], [5.0, 6.0])          # carried over automatically
    np.testing.assert_array_equal(pos_vel.velocity[0], [7.0, 8.0])


def test_add_duplicate_component_raises():
    """Adding a component the entity already has is an error."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))

    with pytest.raises(AssertionError):
        world.add_component(eid, HasVelocity, velocity=np.array([9.0, 9.0], "float32"))


def test_add_unknown_component_is_rejected_and_keeps_entity():
    """Adding a component the world never registered must fail cleanly, leaving the entity untouched."""
    world = World(components=[HasPosition, HasVelocity])        # HasRadius is NOT registered with this world
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    with pytest.raises(AssertionError):                        # a clear error, not a raw KeyError
        world.add_component(eid, HasRadius, radius=np.array([5.0], "float32"))

    assert eid in world._eid_to_pool_ix                        # a rejected migration must not destroy the entity
    pool, ix = world._eid_to_pool_ix[eid]
    np.testing.assert_array_equal(pool.position[ix], [1.0, 2.0])  # original data still there

def test_remove_absent_component_raises():
    """Removing a component the entity does not have is an error."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    with pytest.raises(AssertionError):
        world.remove_component(eid, HasVelocity)


def test_remove_entity_by_id():
    """remove_entity(eid) drops exactly that entity; the rest are conserved and still correct."""
    world = World(components=[HasPosition])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    drop = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))

    world.remove_entity(drop)

    assert sum(len(pool) for pool in world.pools.values()) == 1            # counts conserved
    pool = world.query_and((HasPosition,))[0]
    np.testing.assert_array_equal(pool.position[0], [1.0, 1.0])            # the kept entity remains
