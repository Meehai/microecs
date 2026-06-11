"""Unit tests for ecs.World.

World is deferred (one mode): add_entity / remove_entity / add_component / remove_component queue a
command and return; nothing materializes until world.update() commits the buffer. So these tests call
world.update() after structural ops before asserting on pool state. Two things stay eager: entity ids
(minted and returned at call time) and field validation that runs synchronously in add_entity /
add_component (so "bad fields" / "unknown component" still raise at the call). Errors that are only
knowable while migrating (duplicate / missing component) raise at commit, inside update().
"""
from dataclasses import field
import random
import numpy as np
import pytest

from microecs import World, Component
from microecs.query_result import QueryResult
from microecs.entity import Entity, ENTITY_INTERNAL_ATTRS


class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasRadius(Component):
    radius: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32"})


class HasBox(Component):  # two fields, to exercise multi-field merge/ordering across migrations
    lo: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})
    hi: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasLabel(Component):  # object-dtype field: holds one arbitrary Python object per entity
    label: np.ndarray = field(metadata={"shape": (1,), "dtype": "object"})


class Frozen(Component):  # zero-field tag: a marker with no data, queried/filtered on as an archetype bit
    pass


class HasScale(Component):  # 0-d array field: exactly one scalar per entity (shape ())
    scale: np.ndarray = field(metadata={"shape": (), "dtype": "float32"})


def test_add_entity_rejects_field_from_an_unrequested_component():
    """An entity declared with only HasPosition may not pass `velocity` (a field of the unrequested HasVelocity).
    Validation is eager: the bad field crashes at the add_entity call, before any update()."""
    world = World(components=[HasPosition, HasVelocity])  # both components known to the world

    with pytest.raises(AssertionError, match="velocity"):
        world.add_entity(
            components=(HasPosition,),                  # entity declares HasPosition only
            position=np.array([1.0, 2.0], "float32"),   # required by HasPosition
            velocity=np.array([3.0, 4.0], "float32"),   # extra: belongs to HasVelocity, not requested
        )


def test_fresh_world_has_no_pools():
    """A world creates pools lazily; before any add_entity there are none."""
    world = World(components=[HasPosition, HasVelocity])
    assert world.pools == {}


def test_add_one_entity_creates_exactly_one_pool_with_one_entity():
    """First add_entity (after commit) creates a single pool, holding that single entity."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert len(world.pools) == 1
    pool = next(iter(world.pools.values()))
    assert len(pool) == 1


def test_entity_lands_in_the_pool_keyed_by_its_components():
    """The entity goes into the pool whose key is exactly the bitmask of its declared components."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    key = world._make_key((HasPosition,))
    assert key in world.pools                                   # a pool with that exact key exists
    assert world.pools[key] is world.query(HasPosition).pool_list[0]  # and query finds the same pool


def test_added_entity_field_values_are_stored():
    """The component data we passed is readable back from the pool, unchanged."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.5, 2.5], "float32"))
    world.update()

    pool = world.query(HasPosition).pool_list[0]
    np.testing.assert_array_equal(pool.position[0], np.array([1.5, 2.5], "float32"))


def test_same_archetype_entities_share_a_single_pool():
    """N entities with the same component set all land in one pool, in insertion order."""
    world = World(components=[HasPosition, HasVelocity])

    for i in range(3):
        world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32"))
    world.update()

    assert len(world.pools) == 1                                # still just one archetype
    pool = world.query(HasPosition).pool_list[0]
    assert len(pool) == 3
    np.testing.assert_array_equal(pool.position, np.array([[0, 0], [1, 1], [2, 2]], "float32"))


def test_distinct_archetypes_get_distinct_pools():
    """Entities with different component sets are stored in separate pools."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))
    world.update()

    assert len(world.pools) == 2
    assert len(world.query(HasPosition).pool_list[0]) >= 1
    pos_vel_pool = world.pools[world._make_key((HasPosition, HasVelocity))]
    assert len(pos_vel_pool) == 1


def test_component_order_does_not_create_a_second_pool():
    """(HasPosition, HasVelocity) and (HasVelocity, HasPosition) are the same archetype."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.add_entity(components=(HasVelocity, HasPosition),
                     position=np.array([5.0, 6.0], "float32"), velocity=np.array([7.0, 8.0], "float32"))
    world.update()

    assert len(world.pools) == 1                                # key is a bitmask, order-independent
    assert len(next(iter(world.pools.values()))) == 2


def test_query_returns_all_pools_that_are_supersets():
    """query(HasPosition) returns every pool containing HasPosition, not just the pos-only one."""
    world = World(components=[HasPosition, HasVelocity, HasRadius])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))
    world.update()

    assert len(world.query(HasPosition)) == 2            # both pools contain HasPosition
    assert len(world.query(HasPosition, HasVelocity)) == 1  # only the richer pool has both


def test_query_is_empty_when_no_pool_has_the_component():
    """Querying a component that no existing pool carries returns no pools."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert len(world.query(HasVelocity)) == 0


def test_query_exclude_drops_pools_that_carry_the_excluded_component():
    """exclude=[HasVelocity] keeps only pools that have HasPosition AND lack HasVelocity: the pos-only entity
    stays, the pos+vel one is filtered out. Without the exclude both would match."""
    world = World(components=[HasPosition, HasVelocity])
    a = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))
    world.update()

    assert len(world.query(HasPosition)) == 2                       # both pools have HasPosition
    narrowed = world.query(HasPosition, exclude=[HasVelocity])     # ...but only one lacks HasVelocity
    assert narrowed.entity_ids.tolist() == [a]


def test_query_exclude_tag_component():
    """The common case: a tag as the exclude filter. query(HasPosition, exclude=[Frozen]) skips every
    frozen entity and returns only the un-tagged ones."""
    world = World(components=[HasPosition, Frozen])
    moving = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(HasPosition, Frozen), position=np.array([3.0, 4.0], "float32"))
    world.update()

    qr = world.query(HasPosition, exclude=[Frozen])
    assert qr.entity_ids.tolist() == [moving]                         # the frozen one is excluded


def test_query_exclude_unmatched_component_is_a_noop():
    """Excluding a component that no matching pool carries removes nothing: the result equals the un-excluded
    query. exclude only ever subtracts, it can't invent matches."""
    world = World(components=[HasPosition, HasVelocity])
    a = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    b = world.add_entity(components=(HasPosition,), position=np.array([3.0, 4.0], "float32"))
    world.update()

    qr = world.query(HasPosition, exclude=[HasVelocity])          # no pos-only pool has velocity
    assert sorted(qr.entity_ids.tolist()) == sorted([a, b])


def test_query_exclude_multiple_components():
    """exclude is satisfied only by pools that carry NONE of the excluded bits. With four archetypes, excluding
    both HasVelocity and Frozen leaves the single pool that has neither."""
    world = World(components=[HasPosition, HasVelocity, Frozen])
    a = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([2.0, 2.0], "float32"), velocity=np.array([0.0, 0.0], "float32"))
    world.add_entity(components=(HasPosition, Frozen), position=np.array([3.0, 3.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity, Frozen),
                     position=np.array([4.0, 4.0], "float32"), velocity=np.array([0.0, 0.0], "float32"))
    world.update()

    qr = world.query(HasPosition, exclude=[HasVelocity, Frozen])
    assert qr.entity_ids.tolist() == [a]                             # only the pos-only pool survives both filters


def test_query_include_and_exclude_are_distinct_cache_entries():
    """The cache key is (include, exclude), so the same include with and without an exclude are independent
    entries: each re-query returns its OWN cached object, never the other and never the cache dict itself."""
    world = World(components=[HasPosition, HasVelocity])
    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))
    world.update()

    wide = world.query(HasPosition)                               # no exclude -> exclude_key 0
    narrow = world.query(HasPosition, exclude=[HasVelocity])     # same include, different exclude

    assert wide is not narrow                                        # distinct keys -> distinct entries
    assert len(wide) == 2 and len(narrow) == 1
    assert world.query(HasPosition) is wide                       # each key keeps returning its own result
    assert world.query(HasPosition, exclude=[HasVelocity]) is narrow


def test_query_exclude_none_matches_empty_exclude():
    """exclude=None is the default and means 'exclude nothing' -- identical key to exclude=[], so both hit the
    same cache entry and return the same object."""
    world = World(components=[HasPosition])
    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert world.query(HasPosition, exclude=None) is world.query(HasPosition, exclude=[])


def test_query_exclude_spans_multiple_pools():
    """exclude can leave more than one surviving archetype. With {Pos}, {Pos,Vel}, {Pos,Frozen}, excluding
    Frozen keeps the first two pools; a vectorised write through qr.position scatters back per pool and
    entity_ids stays aligned with the rows."""
    world = World(components=[HasPosition, HasVelocity, Frozen])
    a = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    b = world.add_entity(components=(HasPosition, HasVelocity),
                         position=np.array([2.0, 2.0], "float32"), velocity=np.array([0.0, 0.0], "float32"))
    world.add_entity(components=(HasPosition, Frozen), position=np.array([3.0, 3.0], "float32"))
    world.update()

    qr = world.query(HasPosition, exclude=[Frozen])
    assert len(qr.pool_list) == 2                              # two archetypes survive the Frozen exclusion
    assert sorted(qr.entity_ids.tolist()) == sorted([a, b])    # the Frozen entity is gone

    qr.position[:] = qr.position + 10.0                        # vectorised write across both surviving pools
    for eid, pos in zip(qr.entity_ids, qr.position):           # scattered back, id-aligned per pool
        np.testing.assert_array_equal(world.get_entity(int(eid)).position, pos)


def test_query_exclude_cache_invalidated_on_mutation():
    """The excluding query is cached, but a mutating update() drops it: after a Frozen entity is added and
    committed, re-querying with exclude=[Frozen] is a fresh object that still excludes it."""
    world = World(components=[HasPosition, Frozen])
    a = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    before = world.query(HasPosition, exclude=[Frozen])
    assert before.entity_ids.tolist() == [a]

    world.add_entity(components=(HasPosition, Frozen), position=np.array([2.0, 2.0], "float32"))
    world.update()                                             # mutating commit -> cache cleared
    after = world.query(HasPosition, exclude=[Frozen])

    assert after is not before                                 # fresh object, not the stale cached one
    assert after.entity_ids.tolist() == [a]                    # the new Frozen entity is still excluded


def test_query_exclude_unregistered_component_raises():
    """Excluding a component the world never registered is an error, surfaced by _make_key's assert -- the
    same guard that protects the include side. (HasRadius is not registered in this world.)"""
    world = World(components=[HasPosition])
    world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    with pytest.raises(AssertionError, match="not in"):
        world.query(HasPosition, exclude=[HasRadius])


def test_query_exclude_contradiction_is_empty():
    """include and exclude overlapping is a contradiction: a pool can't both have and lack HasPosition, so the
    result is empty -- not a crash. (archetype & exclude) != 0 for every otherwise-matching pool."""
    world = World(components=[HasPosition])
    world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    qr = world.query(HasPosition, exclude=[HasPosition])
    assert len(qr) == 0
    assert qr.entity_ids.tolist() == []


def test_entities_are_conserved_across_pools():
    """Summing len over all pools equals the number of entities added."""
    world = World(components=[HasPosition, HasVelocity])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(HasPosition,), position=np.array([3.0, 4.0], "float32"))
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([5.0, 6.0], "float32"), velocity=np.array([7.0, 8.0], "float32"))
    world.update()

    assert sum(len(pool) for pool in world.pools.values()) == 3


def test_remove_entity_leaves_empty_pool():
    """Removing the only entity empties its pool and leaves the id bookkeeping consistent (the `else` branch)."""
    world = World(components=[HasPosition])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()
    pool = world.query(HasPosition).pool_list[0]
    assert len(pool) == 1

    world.remove_entity(eid)                                   # last entity out -> pool becomes empty
    world.update()

    assert len(pool) == 0                                      # pool is empty
    assert len(world._pool_ids) == 0                           # no dangling pool_ids
    assert world._eid_to_pool_ix == {}                         # removed id is gone, not pointing at an empty slot


def test_empty_pool_is_reclaimed():
    """When the last entity leaves a pool, that archetype is fully dropped from the world (not leaked)."""
    world = World(components=[HasPosition, HasVelocity])
    pos_key = world._make_key((HasPosition,))

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()
    pos_pool = world.pools[pos_key]
    assert pos_key in world.pools                              # pool exists while it holds an entity

    world.add_component(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32"))  # empties the pos-only pool
    world.update()

    assert pos_key not in world.pools                          # reclaimed from the archetype registry
    assert pos_pool not in world.pool_to_components            # and released from the reverse map -> fully reclaimed


def test_remove_last_index_drops_only_that_entity():
    """Removing the last row (no swap happens) must drop that id, not resurrect it at a now-dead slot."""
    world = World(components=[HasPosition])

    keep = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))  # idx 0
    last = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))  # idx 1 (last)

    world.remove_entity(last)
    world.update()

    pool = world.query(HasPosition).pool_list[0]
    assert len(pool) == 1
    assert last not in world._eid_to_pool_ix                   # removed id gone, not pointing at a dead slot
    assert world._eid_to_pool_ix == {keep: (pool, 0)}          # only the survivor remains, at its row
    assert world._pool_ids[pool] == [keep]                     # reverse map agrees
    np.testing.assert_array_equal(pool.position[0], [1.0, 1.0])


def test_remove_middle_entity_repoints_swapped_id():
    """Removing a middle row swaps the tail into the gap: the tail's id re-points, the removed id vanishes."""
    world = World(components=[HasPosition])

    a = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))  # idx 0
    b = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))  # idx 1 (removed)
    c = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))  # idx 2 (tail)

    world.remove_entity(b)                                     # c slides from slot 2 into slot 1
    world.update()

    pool = world.query(HasPosition).pool_list[0]
    assert len(pool) == 2
    assert b not in world._eid_to_pool_ix                      # removed id gone
    assert world._eid_to_pool_ix[a] == (pool, 0)               # a untouched
    assert world._eid_to_pool_ix[c] == (pool, 1)               # c re-pointed to the freed slot
    assert world._pool_ids[pool] == [a, c]                     # reverse map consistent
    np.testing.assert_array_equal(pool.position[1], [2.0, 2.0])   # c's data now sits at slot 1


def test_add_entity_returns_unique_ids():
    """Every add_entity hands back a distinct, monotonically increasing id -- eagerly, before any update()."""
    world = World(components=[HasPosition])

    ids = [world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32")) for i in range(3)]

    assert len(set(ids)) == 3                                   # all distinct
    assert ids == sorted(ids)                                   # monotonic (distinct + sorted => strictly increasing)


def test_id_resolves_after_sibling_removed():
    """Swap-remove moves the tail row; the moved entity's id must still resolve to it, not to its new neighbour.
    The whole add/remove/add/remove sequence commits in order on a single update(), same as immediate mode would."""
    world = World(components=[HasPosition])

    a = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    b = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))

    world.remove_entity(a)                                      # swap-remove: b's row slides into a's old slot
    world.add_entity(components=(HasPosition,), position=np.array([3.0, 3.0], "float32"))  # c lands after b

    world.remove_entity(b)                                      # must drop b ([2,2]), not c, despite the earlier shuffle
    world.update()

    pool = world.query(HasPosition).pool_list[0]
    assert len(pool) == 1
    np.testing.assert_array_equal(pool.position[0], [3.0, 3.0])  # only c remains


def test_add_component_moves_entity_and_preserves_fields():
    """add_component widens the archetype: entity leaves the old pool, old field intact, new field set."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_component(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    assert world._make_key((HasPosition,)) not in world.pools               # position-only pool emptied -> reclaimed
    pos_vel = world.pools[world._make_key((HasPosition, HasVelocity))]
    assert len(pos_vel) == 1
    np.testing.assert_array_equal(pos_vel.position[0], [1.0, 2.0])          # carried-over value intact
    np.testing.assert_array_equal(pos_vel.velocity[0], [3.0, 4.0])          # new value set


def test_add_component_keeps_entity_id():
    """The id is the caller's stable handle: migrating via add_component must NOT change it."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_component(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32"))
    world.update()

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
    world.update()

    assert world._make_key((HasPosition, HasVelocity)) not in world.pools   # richer pool emptied -> reclaimed
    pos_only = world.pools[world._make_key((HasPosition,))]
    assert len(pos_only) == 1
    np.testing.assert_array_equal(pos_only.position[0], [1.0, 2.0])         # kept field survives
    assert not hasattr(pos_only, "velocity")                               # dropped field is gone


def test_add_component_only_needs_new_fields():
    """Caller supplies just the new component's field; existing fields carry over without being re-passed."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([5.0, 6.0], "float32"))
    world.add_component(eid, HasVelocity, velocity=np.array([7.0, 8.0], "float32"))  # position NOT re-passed
    world.update()

    pos_vel = world.pools[world._make_key((HasPosition, HasVelocity))]
    np.testing.assert_array_equal(pos_vel.position[0], [5.0, 6.0])          # carried over automatically
    np.testing.assert_array_equal(pos_vel.velocity[0], [7.0, 8.0])


def test_add_duplicate_component_raises():
    """Adding a component the entity already has is an error -- caught at commit, where the migration runs."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    world.add_component(eid, HasVelocity, velocity=np.array([9.0, 9.0], "float32"))  # queued
    with pytest.raises(AssertionError):
        world.update()                                                               # duplicate detected on commit


def test_add_unknown_component_raises():
    """Adding a component the world never registered is rejected eagerly, at the call (cheap synchronous check)."""
    world = World(components=[HasPosition, HasVelocity])        # HasRadius is NOT registered with this world
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    with pytest.raises(AssertionError):
        world.add_component(eid, HasRadius, radius=np.array([5.0], "float32"))


def test_remove_absent_component_raises():
    """Removing a component the entity does not have is an error -- caught at commit (the field check runs there)."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.remove_component(eid, HasVelocity)                   # queued
    with pytest.raises(AssertionError):
        world.update()                                         # missing-field detected on commit


def test_remove_entity_by_id():
    """remove_entity(eid) drops exactly that entity; the rest are conserved and still correct."""
    world = World(components=[HasPosition])

    world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    drop = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))

    world.remove_entity(drop)
    world.update()

    assert sum(len(pool) for pool in world.pools.values()) == 1            # counts conserved
    pool = world.query(HasPosition).pool_list[0]
    np.testing.assert_array_equal(pool.position[0], [1.0, 1.0])            # the kept entity remains


def test_add_component_to_middle_sibling_keeps_all_ids():
    """Migrating the middle of three siblings triggers a swap in the old pool; every id must still resolve to its row."""
    world = World(components=[HasPosition, HasVelocity])

    ids = [world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32")) for i in range(3)]
    world.add_component(ids[1], HasVelocity, velocity=np.array([9.0, 9.0], "float32"))  # middle one leaves the pool
    world.update()

    assert sum(len(pool) for pool in world.pools.values()) == 3            # nobody lost
    for eid, expected in zip(ids, ([0, 0], [1, 1], [2, 2])):
        pool, ix = world._eid_to_pool_ix[eid]
        np.testing.assert_array_equal(pool.position[ix], expected)         # each id still points at its own data


def test_add_then_remove_component_round_trips():
    """add_component then remove_component returns the entity to its original archetype, same id, data intact."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([5.0, 6.0], "float32"))
    world.add_component(eid, HasVelocity, velocity=np.array([7.0, 8.0], "float32"))
    world.remove_component(eid, HasVelocity)
    world.update()

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool is world.pools[world._make_key((HasPosition,))]            # back in the position-only pool
    assert world._make_key((HasPosition, HasVelocity)) not in world.pools   # richer pool emptied -> reclaimed
    np.testing.assert_array_equal(pool.position[ix], [5.0, 6.0])           # original data survived the round trip


def test_remove_component_lands_in_existing_pool_and_conserves_entities():
    """remove_component moving an entity into an already-populated smaller pool keeps both entities intact."""
    world = World(components=[HasPosition, HasVelocity])

    sibling = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    mover = world.add_entity(components=(HasPosition, HasVelocity),
                             position=np.array([2.0, 2.0], "float32"), velocity=np.array([3.0, 3.0], "float32"))

    world.remove_component(mover, HasVelocity)                             # mover joins sibling's pos-only pool
    world.update()

    assert sum(len(pool) for pool in world.pools.values()) == 2            # both conserved
    for eid, expected in ((sibling, [1.0, 1.0]), (mover, [2.0, 2.0])):
        pool, ix = world._eid_to_pool_ix[eid]
        assert pool is world.pools[world._make_key((HasPosition,))]
        np.testing.assert_array_equal(pool.position[ix], expected)


def test_add_component_reuses_existing_archetype_pool():
    """Reaching an archetype via add_component lands in the same pool a direct add_entity would (order-independent)."""
    world = World(components=[HasPosition, HasVelocity])

    direct = world.add_entity(components=(HasVelocity, HasPosition),
                              position=np.array([1.0, 1.0], "float32"), velocity=np.array([2.0, 2.0], "float32"))
    migrated = world.add_entity(components=(HasPosition,), position=np.array([3.0, 3.0], "float32"))
    world.add_component(migrated, HasVelocity, velocity=np.array([4.0, 4.0], "float32"))
    world.update()

    assert world._eid_to_pool_ix[direct][0] is world._eid_to_pool_ix[migrated][0]  # same pool object, no dup archetype
    assert len(world.pools[world._make_key((HasPosition, HasVelocity))]) == 2


def test_migrate_multi_field_component_preserves_all_fields():
    """A component with several fields migrates with every field intact and correctly named in the new pool."""
    world = World(components=[HasPosition, HasBox])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_component(eid, HasBox, lo=np.array([0.0, 0.0], "float32"), hi=np.array([4.0, 4.0], "float32"))
    world.update()

    pool, ix = world._eid_to_pool_ix[eid]
    np.testing.assert_array_equal(pool.position[ix], [1.0, 2.0])           # carried-over field
    np.testing.assert_array_equal(pool.lo[ix], [0.0, 0.0])                 # both new fields land, by name
    np.testing.assert_array_equal(pool.hi[ix], [4.0, 4.0])


# --- eager id tracking -------------------------------------------------------------------------------------------
# A structural op on an entity that is NOT currently live fails at the CALL (clear AssertionError), not later inside
# update() as a cryptic KeyError. "Live" = committed or pending-spawn this tick, minus pending-despawn; World keeps
# this in live_entities. add_entity adds the new id; remove_entity removes it; add/remove_component just validate.


def test_operate_on_uncommitted_spawn_same_tick():
    """Boundary that must keep working: an id minted this tick is a valid handle BEFORE commit.
    add_component on a not-yet-committed spawn lands correctly after a single update() (pending spawn == live)."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))  # queued, not committed
    world.add_component(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32"))              # operate pre-commit
    world.update()

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool is world.pools[world._make_key((HasPosition, HasVelocity))]
    np.testing.assert_array_equal(pool.position[ix], [1.0, 2.0])
    np.testing.assert_array_equal(pool.velocity[ix], [3.0, 4.0])


def test_remove_entity_twice_fails_on_second_call():
    """Removing the same id twice in a tick: the second call targets an already-removed entity -> reject eagerly."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    world.remove_entity(eid)                                   # eid now logically gone (pending despawn)
    with pytest.raises(AssertionError):
        world.remove_entity(eid)                               # 2nd call must fail at the call site


def test_add_component_after_remove_entity_fails():
    """A system removes an entity; a later system tries to widen it the same tick -> reject eagerly."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.remove_entity(eid)
    with pytest.raises(AssertionError):
        world.add_component(eid, HasVelocity, velocity=np.array([3.0, 4.0], "float32"))


def test_remove_component_after_remove_entity_fails():
    """A system removes an entity; a later system tries to narrow it the same tick -> reject eagerly."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    world.remove_entity(eid)
    with pytest.raises(AssertionError):
        world.remove_component(eid, HasVelocity)


def test_remove_unknown_entity_id_fails():
    """An id the world never handed out is not live -> remove_entity must reject it at the call, not at commit."""
    world = World(components=[HasPosition])
    with pytest.raises(AssertionError):
        world.remove_entity(123)


def test_spawn_into_archetype_reclaimed_by_earlier_despawn_same_tick():
    """Despawn the last entity of an archetype, then spawn a new one of the SAME archetype, same tick.
    The despawn reclaims the pool at commit; the newcomer must still land in a live, queryable pool, not orphaned."""
    world = World(components=[HasPosition])
    old = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))
    world.update()

    world.remove_entity(old)                                       # queued first: empties -> reclaims the pos pool
    new = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    assert old not in world._eid_to_pool_ix
    assert sum(len(pool) for pool in world.pools.values()) == 1    # exactly the newcomer, and it is visible
    assert world._make_key((HasPosition,)) in world.pools          # its pool is live / registered (not orphaned)
    pool, ix = world._eid_to_pool_ix[new]
    np.testing.assert_array_equal(pool.position[ix], [1.0, 1.0])


# --- object-dtype components -------------------------------------------------------------------------------------
# A component field may declare dtype "object": its storage holds arbitrary Python objects (dicts, callbacks, handles)
# by reference rather than numeric data. Everything else (pools, migrations, swap-remove) must treat it like any field.


def test_world_accepts_object_dtype_component():
    """Construction validation allows dtype 'object'; the world records it for the field."""
    world = World(components=[HasLabel])
    assert world.component_to_dtypes[HasLabel] == ["object"]


def test_object_component_stores_and_reads_back_the_same_object():
    """The exact Python object passed in is readable back from the pool -- by identity, not just by value."""
    world = World(components=[HasLabel])

    payload = {"name": "drone-7", "tags": ["a", "b"]}
    eid = world.add_entity(components=(HasLabel,), label=np.array([payload], dtype=object))
    world.update()

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool.label.dtype == object
    assert pool.label[ix, 0] is payload                            # same reference, not a copy


def test_object_component_survives_migration():
    """add_component carries an object field over to the wider pool with its reference intact."""
    world = World(components=[HasPosition, HasLabel])

    obj = object()
    eid = world.add_entity(components=(HasLabel,), label=np.array([obj], dtype=object))
    world.add_component(eid, HasPosition, position=np.array([1.0, 2.0], "float32"))
    world.update()

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool is world.pools[world._make_key((HasPosition, HasLabel))]
    assert pool.label[ix, 0] is obj                                # object preserved across the archetype move
    np.testing.assert_array_equal(pool.position[ix], [1.0, 2.0])   # sibling numeric field set as usual


def test_distinct_objects_per_entity_survive_swap_remove():
    """Each entity keeps its own object; removing one swaps the tail in, and the survivor's object is unchanged."""
    world = World(components=[HasLabel])

    first, second = {"id": 1}, {"id": 2}
    a = world.add_entity(components=(HasLabel,), label=np.array([first], dtype=object))   # idx 0
    b = world.add_entity(components=(HasLabel,), label=np.array([second], dtype=object))  # idx 1 (tail)
    world.update()

    world.remove_entity(a)                                         # b's row swaps into slot 0
    world.update()

    pool, ix = world._eid_to_pool_ix[b]
    assert len(pool) == 1
    assert pool.label[ix, 0] is second                            # the right object followed the right id


# --- get_entity: read one entity's data + components by id -------------------------------------------------------
# get_entity(eid) is a READ accessor: returns an Entity view (entity.field, entity.get_components()) for the entity at
# its current row, resolved by id (not index). It must NOT mutate id bookkeeping -- the id has to keep resolving and
# the entity stays usable after.


def test_get_entity_returns_field_data_and_components():
    """The happy path: get_entity hands back the entity's field values plus its component list."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    entity = world.get_entity(eid)
    np.testing.assert_array_equal(entity.position, [1.0, 2.0])
    assert set(entity.get_components()) == {HasPosition}


def test_get_entity_returns_all_fields_of_a_multi_component_entity():
    """Every field of a multi-component archetype comes back, keyed by field name."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    entity = world.get_entity(eid)
    np.testing.assert_array_equal(entity.position, [1.0, 2.0])
    np.testing.assert_array_equal(entity.velocity, [3.0, 4.0])
    assert set(entity.get_components()) == {HasPosition, HasVelocity}


def test_get_entity_is_read_only_and_id_still_resolves():
    """get_entity must NOT consume the entity: the id keeps resolving, the call is repeatable, removal still works."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.get_entity(eid)

    assert eid in world._eid_to_pool_ix                            # lookup intact (a read may not delete the mapping)
    assert eid in world.live_entities
    world.get_entity(eid)                                          # repeatable -> not consumed by the first read

    world.remove_entity(eid)                                       # normal lifecycle still works afterwards
    assert eid not in world.live_entities                          # eagerly evicted from the live cache at the call
    world.update()
    assert eid not in world._eid_to_pool_ix


def test_get_entity_reads_current_row_after_sibling_swap_remove():
    """After a swap-remove relocates rows, get_entity(id) still returns each id's own data, not a neighbour's."""
    world = World(components=[HasPosition])
    a = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))  # idx 0
    b = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))  # idx 1
    c = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))  # idx 2 (tail)
    world.update()

    world.remove_entity(a)                                         # c swaps into slot 0; b stays at slot 1
    world.update()

    entity_b = world.get_entity(b)
    entity_c = world.get_entity(c)
    np.testing.assert_array_equal(entity_b.position, [1.0, 1.0])  # b unmoved
    np.testing.assert_array_equal(entity_c.position, [2.0, 2.0])  # c followed its id into the freed slot


def test_get_entity_unknown_id_raises():
    """An id the world never handed out has no data -> raise, not return an empty/garbage result."""
    world = World(components=[HasPosition])
    with pytest.raises((AssertionError, KeyError)):                # ideally a clear AssertionError, like the other ops
        world.get_entity(123)


# --- set_entity_data: write one entity's single field by id ------------------------------------------------------
# set_entity_data(eid, field, value) is the WRITE counterpart to get_entity: it resolves an id to its current
# (pool, row) and writes one field straight into the pool buffer. It is EAGER (no command buffer, no update()) and
# non-vectorised -- "use rarely". The contract: resolve by id (not index), touch only the named field, leave id
# bookkeeping untouched.


def test_set_entity_data_writes_field_value_by_id():
    """The happy path: set a field by id, read the new value back (via get_entity and via the pool)."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.set_entity_data(eid, "position", np.array([9.0, 8.0], "float32"))

    np.testing.assert_array_equal(world.get_entity(eid).position, [9.0, 8.0])
    pool, ix = world._eid_to_pool_ix[eid]
    np.testing.assert_array_equal(pool.position[ix], [9.0, 8.0])   # the actual pool row was overwritten


def test_set_entity_data_is_eager_visible_without_update():
    """Unlike add/remove (command-buffered), set_entity_data writes immediately: the value is there before any
    further update() -- it is a direct pool write, not a queued command."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()                                                 # commits the spawn only

    world.set_entity_data(eid, "position", np.array([5.0, 5.0], "float32"))

    np.testing.assert_array_equal(world.get_entity(eid).position, [5.0, 5.0])  # no second update() needed
    assert world._command_buffer == []                             # nothing was queued


def test_set_entity_data_only_updates_the_named_field():
    """Writing one field of a multi-field entity leaves the entity's other fields untouched."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    world.set_entity_data(eid, "velocity", np.array([7.0, 7.0], "float32"))

    entity = world.get_entity(eid)
    np.testing.assert_array_equal(entity.velocity, [7.0, 7.0])  # the targeted field changed
    np.testing.assert_array_equal(entity.position, [1.0, 2.0])  # the sibling field is untouched


def test_set_entity_data_targets_correct_row_after_swap_remove():
    """The core safety property: writes resolve by id, not by index. After a swap-remove relocates rows, setting
    by id must hit the moved entity's CURRENT row -- never a neighbour's."""
    world = World(components=[HasPosition])
    a = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))  # idx 0
    b = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))  # idx 1
    c = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))  # idx 2 (tail)
    world.update()

    world.remove_entity(a)                                         # c swaps into slot 0; b stays at slot 1
    world.update()

    world.set_entity_data(c, "position", np.array([20.0, 20.0], "float32"))   # by id, after the relocation

    np.testing.assert_array_equal(world.get_entity(c).position, [20.0, 20.0])  # c got the write
    np.testing.assert_array_equal(world.get_entity(b).position, [1.0, 1.0])    # b (its neighbour) untouched


def test_set_entity_data_is_visible_through_query_view():
    """The write lands in the live pool buffer, so a vectorised query view reads the new value -- set_entity_data
    and query see the same storage."""
    world = World(components=[HasPosition])
    a = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))
    world.update()

    world.set_entity_data(a, "position", np.array([9.0, 9.0], "float32"))

    qr = world.query(HasPosition)
    row = {int(eid): pos for eid, pos in zip(qr.entity_ids, qr.position.numpy())}
    np.testing.assert_array_equal(row[a], [9.0, 9.0])              # the query view reflects the single-entity write


def test_set_entity_data_copies_numeric_value_not_aliases():
    """numpy slot-assignment copies into the pre-allocated buffer: mutating the source array afterwards must NOT
    leak into the stored row."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    src = np.array([5.0, 6.0], "float32")
    world.set_entity_data(eid, "position", src)
    src[:] = [999.0, 999.0]                                        # mutate the source after the call

    np.testing.assert_array_equal(world.get_entity(eid).position, [5.0, 6.0])  # stored value is independent


def test_set_entity_data_on_object_field_replaces_reference():
    """For an object-dtype field the write stores the exact Python object by reference (not a copy)."""
    world = World(components=[HasLabel])
    eid = world.add_entity(components=(HasLabel,), label=np.array([{"v": 0}], dtype=object))
    world.update()

    replacement = {"v": 42}
    world.set_entity_data(eid, "label", np.array([replacement], dtype=object))

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool.label[ix, 0] is replacement                        # same reference, swapped in


def test_set_entity_data_zero_dim_scalar_field():
    """A shape-() scalar field can be written by id and reads back as a 0-d scalar."""
    world = World(components=[HasScale])
    eid = world.add_entity(components=(HasScale,), scale=np.array(2.5, "float32"))
    world.update()

    world.set_entity_data(eid, "scale", np.array(4.0, "float32"))

    entity = world.get_entity(eid)
    assert entity.scale.shape == ()
    np.testing.assert_array_equal(entity.scale, 4.0)


def test_set_entity_data_unknown_id_raises():
    """An id the world never handed out cannot resolve to a row -> clear AssertionError, no silent write."""
    world = World(components=[HasPosition])
    world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    with pytest.raises(AssertionError, match="not found"):
        world.set_entity_data(123, "position", np.array([0.0, 0.0], "float32"))


def test_set_entity_data_on_uncommitted_spawn_raises():
    """set_entity_data needs the entity already committed to a pool: a pending spawn (live id, no row yet) raises
    -- unlike add_component, which works on the same-tick spawn. The message points at the missing update()."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))  # queued, not committed

    with pytest.raises(AssertionError, match="not found"):
        world.set_entity_data(eid, "position", np.array([0.0, 0.0], "float32"))


def test_set_entity_data_unknown_field_name_raises():
    """No field-name validation: writing a field the entity's pool doesn't have surfaces a KeyError (documented
    footgun -- the caller is trusted to name a real field)."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    with pytest.raises(KeyError):
        world.set_entity_data(eid, "velocity", np.array([0.0, 0.0], "float32"))  # velocity not in this pool


# --- _pool_ids randomized churn: the reverse id-map must mirror the pools through every popswap ------------------

_CHURN_COMPONENTS = {HasPosition: ("position", (2,)), HasVelocity: ("velocity", (2,)), HasRadius: ("radius", (1,))}


def _rand_fields(comp, rng: random.Random) -> dict:
    """Random field-data kwargs for one churn component (one (shape,) float32 field, name unique per component)."""
    name, shape = _CHURN_COMPONENTS[comp]
    return {name: np.array([rng.random() for _ in range(shape[0])], "float32")}


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


def test_pool_ids_stay_aligned_through_random_churn():
    """500 seeded random ops (add / remove / add_component / remove_component) interleaved across archetypes. After
    every commit the reverse id-map mirrors the pools AND each id's field data round-trips through get_entity -- so
    no swap ever hands an id its neighbour's row."""
    rng = random.Random(1234)
    world = World(components=list(_CHURN_COMPONENTS))
    shadow: dict[int, dict] = {}   # eid -> {field_name: data} we believe the world holds

    for _ in range(500):
        live = list(world.live_entities)
        roll = rng.random()
        if roll < 0.45 or not live:                                  # add a new entity (random archetype)
            comps = rng.sample(list(_CHURN_COMPONENTS), rng.randint(1, 3))
            data = {}
            for c in comps:
                data.update(_rand_fields(c, rng))
            eid = world.add_entity(components=tuple(comps), **{k: v.copy() for k, v in data.items()})
            shadow[eid] = {k: v.copy() for k, v in data.items()}
        elif roll < 0.70:                                            # remove an entity (forces a popswap)
            eid = rng.choice(live)
            world.remove_entity(eid)
            shadow.pop(eid)
        elif roll < 0.85:                                            # grow an entity's archetype
            eid = rng.choice(live)
            missing = [c for c in _CHURN_COMPONENTS if _CHURN_COMPONENTS[c][0] not in shadow[eid]]
            if missing:
                c = rng.choice(missing)
                d = _rand_fields(c, rng)
                world.add_component(eid, c, **{k: v.copy() for k, v in d.items()})
                shadow[eid].update({k: v.copy() for k, v in d.items()})
        else:                                                        # shrink it (never below one component)
            eid = rng.choice(live)
            have = [c for c in _CHURN_COMPONENTS if _CHURN_COMPONENTS[c][0] in shadow[eid]]
            if len(have) > 1:
                c = rng.choice(have)
                world.remove_component(eid, c)
                shadow[eid].pop(_CHURN_COMPONENTS[c][0])
        world.update()

        _assert_pool_ids_invariants(world)
        for eid, fields in shadow.items():
            entity = world.get_entity(eid)
            for name, value in fields.items():
                np.testing.assert_array_equal(getattr(entity, name), value)  # each id keeps its OWN data thru swaps

    assert len(world.live_entities) > 0                            # sanity: the churn left a populated world


# --- QueryResult.entity_ids: a flat (N,) integer array, pool-by-pool aligned with the qr.field parts -----------

def test_query_result_entity_ids_is_flat_and_aligned_across_pools():
    """qr.entity_ids is a flat (N,) integer array covering every matched entity across archetypes, in the same
    pool-by-pool order as qr.position -- so zip(qr.entity_ids, qr.position) pairs each id with its own row."""
    world = World(components=[HasPosition, HasVelocity])
    a = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))
    b = world.add_entity(components=(HasPosition, HasVelocity),
                         position=np.array([1.0, 1.0], "float32"), velocity=np.array([9.0, 9.0], "float32"))
    c = world.add_entity(components=(HasPosition, HasVelocity),
                         position=np.array([2.0, 2.0], "float32"), velocity=np.array([8.0, 8.0], "float32"))
    world.update()

    qr = world.query(HasPosition)                           # matches both archetypes -> two pools

    assert isinstance(qr.entity_ids, np.ndarray)
    assert np.issubdtype(qr.entity_ids.dtype, np.integer)
    assert qr.entity_ids.shape == (len(qr),)                       # flat, one entry per entity
    assert set(qr.entity_ids.tolist()) == {a, b, c}                # exactly the matched ids
    for eid, pos in zip(qr.entity_ids, qr.position):               # id <-> row alignment, across pools
        np.testing.assert_array_equal(world.get_entity(int(eid)).position, pos)


def test_query_result_entity_ids_supports_flat_array_ops():
    """entity_ids is a real ndarray, not a _Field: entity-axis indexing and fancy ops that _Field rejects --
    qr.entity_ids[i], slicing, np.isin -- all work, because ids are materialized by World, not a per-pool view."""
    world = World(components=[HasPosition])
    ids = [world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32")) for i in range(4)]
    world.update()

    qr = world.query(HasPosition)

    assert int(qr.entity_ids[0]) in ids                            # entity-axis index -> allowed (unlike _Field)
    assert qr.entity_ids[1:3].shape == (2,)                        # slicing the entity axis -> allowed
    assert np.isin(qr.entity_ids, ids[:2]).sum() == 2              # fancy / set ops -> allowed


def test_query_result_entity_ids_empty_query_is_empty_flat_array():
    """A query that matches no pool yields an empty flat (0,) id array, mirroring an empty field -- not a crash."""
    world = World(components=[HasPosition, HasVelocity])
    world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    qr = world.query(HasVelocity)                           # nothing has velocity

    assert len(qr) == 0
    assert qr.entity_ids.shape == (0,)


def test_query_result_entity_ids_track_rows_after_swap_remove():
    """After a swap-remove relocates rows, qr.entity_ids still aligns with qr.position: each surviving id pairs
    with its own (moved) data, never a neighbour's."""
    world = World(components=[HasPosition])
    a = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    b = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))
    c = world.add_entity(components=(HasPosition,), position=np.array([3.0, 3.0], "float32"))
    world.update()
    world.remove_entity(b)                                         # swap: c slides into b's slot
    world.update()

    qr = world.query(HasPosition)

    assert set(qr.entity_ids.tolist()) == {a, c}
    for eid, pos in zip(qr.entity_ids, qr.position):
        np.testing.assert_array_equal(world.get_entity(int(eid)).position, pos)


def test_query_cache_returns_same_object_between_updates():
    """Two query calls for the same components, with no mutating update between, return the SAME QueryResult
    object -- the second is served from the cache, not rebuilt."""
    world = World(components=[HasPosition])
    world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    first = world.query(HasPosition)
    second = world.query(HasPosition)

    assert first is second


def test_noop_update_keeps_cache():
    """An update() that commits nothing (empty command buffer) changes no pool, so the cache survives: a re-query
    returns the same object cached before that update()."""
    world = World(components=[HasPosition])
    world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    cached = world.query(HasPosition)
    world.update()                                  # empty buffer -> no structural change
    again = world.query(HasPosition)

    assert again is cached


def test_mutating_update_invalidates_cache():
    """An update() that commits a structural change drops the cache: the next query is a fresh object whose len
    and entity_ids reflect the new entity, not the stale cached result."""
    world = World(components=[HasPosition])
    a = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    before = world.query(HasPosition)
    assert len(before) == 1

    b = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()                                  # mutating commit -> cache cleared
    after = world.query(HasPosition)

    assert after is not before
    assert len(after) == 2
    assert set(after.entity_ids.tolist()) == {a, b}


def test_cache_keyed_per_query():
    """The cache is keyed by the query, so different component sets get independent entries and never collide:
    query (HasPosition,) and query (HasVelocity,) are distinct objects, and each key keeps returning its own."""
    world = World(components=[HasPosition, HasVelocity])
    world.add_entity(components=(HasPosition, HasVelocity),
                     position=np.array([0.0, 0.0], "float32"), velocity=np.array([1.0, 1.0], "float32"))
    world.update()

    pos = world.query(HasPosition)
    vel = world.query(HasVelocity)

    assert pos is not vel                           # distinct queries -> distinct cache entries
    assert world.query(HasPosition) is pos   # each key returns its own cached result
    assert world.query(HasVelocity) is vel


def test_new_archetype_appears_after_invalidation():
    """Spawning the first entity of a brand-new archetype and committing it must show up in a re-query: the new
    pool is not masked by a stale cached result. (HasPosition,) matches both the position-only pool and the new
    position+velocity pool."""
    world = World(components=[HasPosition, HasVelocity])
    world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    before = world.query(HasPosition)
    assert len(before) == 1                         # only the position-only entity so far

    world.add_entity(components=(HasPosition, HasVelocity),                       # brand-new archetype
                     position=np.array([1.0, 1.0], "float32"), velocity=np.array([2.0, 2.0], "float32"))
    world.update()
    after = world.query(HasPosition)

    assert len(after) == 2                           # the new pool is visible after invalidation


def test_no_stale_views_across_realloc():
    """Growing a pool past its capacity makes Pool._realloc swap in a NEW backing array, so a cached query taken
    before the growth must not be reused -- its views point at the old, freed array. After the committing
    update() the re-query is a fresh object whose views write through to the live (reallocated) pool."""
    world = World(components=[HasPosition])
    for i in range(100):                            # fill exactly to INITIAL_CAPACITY (100)
        world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32"))
    world.update()

    stale = world.query(HasPosition)         # views into the capacity-100 backing array
    assert len(stale) == 100

    world.add_entity(components=(HasPosition,), position=np.array([999, 999], "float32"))  # forces _realloc(200)
    world.update()                                  # mutating commit -> cache cleared
    fresh = world.query(HasPosition)

    assert fresh is not stale                        # the pre-realloc result is not reused
    assert len(fresh) == 101                          # sees the grown pool

    fresh.position[:] = fresh.position + 1            # must land in the LIVE (reallocated) array
    pool = next(iter(world.pools.values()))
    np.testing.assert_array_equal(pool.position[0], [1.0, 1.0])           # entity i=0 -> +1
    np.testing.assert_array_equal(pool.position[100], [1000.0, 1000.0])   # the 101st entity -> +1


_QUERYRESULT_RESERVED = sorted(vars(QueryResult([], {}, {}, np.array([], "int64"))))
@pytest.mark.parametrize("reserved", _QUERYRESULT_RESERVED)
def test_world_rejects_component_field_named_like_a_queryresult_attribute(reserved):
    """A component whose field is named like a QueryResult attribute must be rejected at world creation, rather
    than be silently shadowed when queried."""
    bad = type("Bad", (Component,), {"__annotations__": {reserved: np.ndarray},
                                      reserved: field(metadata={"shape": (2,), "dtype": "float32"})})
    with pytest.raises((AssertionError, ValueError)):
        World(components=[bad])


# An Entity (world.get_entity(id)) exposes the row by attribute, so a component field named like one of Entity's
# own members would be shadowed: e.entity_id would return the id (not the field), e.get_components a bound method.
# These must be rejected at world creation. Derived programmatically (not hardcoded) so new attrs/methods are picked
# up automatically. Entity isn't slotted, so its collidable surface is split in two: instance-attr names live in the
# module constant ENTITY_INTERNAL_ATTRS, public methods live in the class dict -- union covers both (mirrors the impl).
_ENTITY_RESERVED = sorted(ENTITY_INTERNAL_ATTRS | {n for n in vars(Entity) if not n.startswith("__")})
@pytest.mark.parametrize("reserved", _ENTITY_RESERVED)
def test_world_rejects_component_field_named_like_an_entity_attribute(reserved):
    """A component whose field is named like an Entity attribute/method must be rejected at world creation,
    rather than be silently shadowed when read/written through get_entity."""
    bad = type("Bad", (Component,), {"__annotations__": {reserved: np.ndarray},
                                      reserved: field(metadata={"shape": (2,), "dtype": "float32"})})
    with pytest.raises((AssertionError, ValueError)):
        World(components=[bad])


def test_extra_metadata_required_strictly_on_every_field():
    """extra_metadata makes named metadata keys mandatory on every field, checked strictly (==).

    A field's metadata must equal EXACTLY {shape, dtype, *extra_metadata}:
      - a plain world (no extras) wants exactly {shape, dtype}
      - a world(extra=["serializable"]) wants exactly {shape, dtype, serializable}
    so each component is valid in exactly ONE of the two worlds. 2 components x 2 worlds = 4 cases,
    of which 2 raise: a required key missing, OR an undeclared extra key present.
    """
    class Plain(Component):    # field carries only the always-required keys
        a: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})

    class Serial(Component):   # same field, plus the extra "serializable" key
        b: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "serializable": True})

    World([Plain])                                          # ok: {shape,dtype} == {shape,dtype}
    World([Serial], extra_metadata=["serializable"])  # ok: {shape,dtype,ser} == {shape,dtype,ser}

    with pytest.raises(AssertionError):                     # missing the required "serializable"
        World([Plain], extra_metadata=["serializable"])
    with pytest.raises(AssertionError):                     # carries "serializable" the world never declared
        World([Serial])


def test_tag_component_is_valid_and_queryable():
    """A field-less component is a valid 'tag': it registers with empty field/shape/dtype maps, needs no data
    on add, lands in its own (pure-tag) pool, and is usable both as a query filter and a tag-only query."""
    world = World([HasPosition, Frozen])

    assert world.component_to_field_names[Frozen] == []     # registers with empty per-field maps
    assert world.component_to_shapes[Frozen] == []
    assert world.component_to_dtypes[Frozen] == []

    tagged = world.add_entity((HasPosition, Frozen), position=np.array([1.0, 2.0], "float32"))
    plain = world.add_entity((HasPosition,), position=np.array([3.0, 4.0], "float32"))
    pure = world.add_entity((Frozen,))                      # pure tag: no data at all
    world.update()

    qr = world.query(HasPosition, Frozen)             # tag as a filter: only the tagged entity, position exposed
    assert qr.entity_ids.tolist() == [tagged]
    np.testing.assert_array_equal(qr.position.numpy(), [[1.0, 2.0]])

    qr_tag = world.query(Frozen)                     # tag-only query spans {Pos,Frozen} + pure {Frozen} pools
    assert sorted(qr_tag.entity_ids.tolist()) == sorted([tagged, pure])
    assert qr_tag.fields == []                             # a tag exposes no fields
    assert len(qr_tag) == 2
    assert plain not in qr_tag.entity_ids.tolist()          # the untagged entity is excluded


def test_tag_component_add_remove_migrates():
    """Adding/removing a tag migrates the entity between archetypes and round-trips its data + id."""
    world = World([HasPosition, Frozen])
    eid = world.add_entity((HasPosition,), position=np.array([5.0, 6.0], "float32"))
    world.update()
    assert world.query(Frozen).entity_ids.tolist() == []

    world.add_component(eid, Frozen)                        # tag carries no data
    world.update()
    assert world.query(Frozen).entity_ids.tolist() == [eid]
    entity = world.get_entity(eid)
    np.testing.assert_array_equal(entity.position, [5.0, 6.0])  # data preserved across the migration
    assert Frozen in entity.get_components()

    world.remove_component(eid, Frozen)
    world.update()
    assert world.query(Frozen).entity_ids.tolist() == []
    entity = world.get_entity(eid)
    np.testing.assert_array_equal(entity.position, [5.0, 6.0])  # id stable, data still there


def test_zero_dim_array_field_roundtrips():
    """A field with shape () (a 0-d / scalar array) is valid: it stores, queries as (N,) and round-trips per entity."""
    world = World([HasScale])
    e0 = world.add_entity((HasScale,), scale=np.array(2.5, "float32"))
    world.add_entity((HasScale,), scale=np.array(4.0, "float32"))
    world.update()

    qr = world.query(HasScale)
    np.testing.assert_array_equal(qr.scale.numpy(), [2.5, 4.0])  # (N,) contiguous view over the 0-d field

    entity = world.get_entity(e0)
    assert entity.scale.shape == ()                            # still a 0-d scalar per entity
    np.testing.assert_array_equal(entity.scale, 2.5)
