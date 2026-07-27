"""Unit tests for ecs.World.

World is deferred (one mode): add_entity / remove_entity / add_component / remove_component queue a
command and return; nothing materializes until world.update() commits the buffer. So these tests call
world.update() after structural ops before asserting on pool state.

Validation is EAGER and the command buffer is a STAGING area (like git's index): every mutator fully
validates at the call and queues only valid commands, so world.update() is a pure, infallible apply --
it materializes the buffer, it does NOT re-validate or roll back (deliberately NOT atomic). A rejected
op raises at the call and leaves the buffer untouched.

"Fully validated" has two halves. Structural: duplicate-add / absent-remove are judged against the
PROJECTED component set (committed, plus this tick's queued adds, minus queued removes) -- so same-tick
sequences like add->remove->add->remove are legal, and a same-tick self-conflict (add the same component
twice) is rejected at the SECOND call, before the poisoning command reaches the buffer. Field data:
dtype / shape / missing-required are checked at the call too, mirroring add_entity (world.py:69 already
runs _check_components_against_pool eagerly), so no field-data error reaches commit either.

add_entity is the template: it is fully eager (world.py:78) and, since task 44, it is the SOLE validator of
a spawn -- CommandBuffer.append used to repeat the same pass and no longer does. add_component /
remove_component have no such single owner (Entity queues a bare command), so for them append IS the gate.
Those buffer-level unit tests live in test_command_buffer.py; the eager-id-tracking tests below exercise the
same staging model through the Entity/World API.
"""
from dataclasses import field
import os
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

import microecs
from microecs import World, Component
from microecs.command_buffer import CommandType
from microecs.query_result import QUERY_RESULT_RESERVED_NAMES
from microecs.entity import ENTITY_RESERVED_NAMES
from microecs.pool import POOL_RESERVED_NAMES


class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasRadius(Component):
    radius: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "default": None})


class HasBox(Component):  # two fields, to exercise multi-field merge/ordering across migrations
    lo: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})
    hi: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasLabel(Component):  # object-dtype field: holds one arbitrary Python object per entity
    label: np.ndarray = field(metadata={"shape": (1,), "dtype": "object", "default": None})


class Frozen(Component):  # zero-field tag: a marker with no data, queried/filtered on as an archetype bit
    pass


class HasScale(Component):  # 0-d array field: exactly one scalar per entity (shape ())
    scale: np.ndarray = field(metadata={"shape": (), "dtype": "float32", "default": None})


class HasColorDefault(Component):  # optional field: metadata carries a real default, filled when omitted
    color: np.ndarray = field(metadata={"shape": (3,), "dtype": "int32", "default": np.array([10, 20, 30], "int32")})


def test_add_entity_rejects_field_from_an_unrequested_component():
    """An entity declared with only HasPosition may not pass `velocity` (a field of the unrequested HasVelocity).
    Validation is eager: the bad field crashes at the add_entity call, before any update()."""
    world = World(components=[HasPosition, HasVelocity])  # both components known to the world

    with pytest.raises(ValueError, match="velocity"):
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
    """Excluding a component the world never registered is an error, surfaced by _make_key's raise -- the
    same guard that protects the include side. A query is user input, so it must survive `python -O`:
    ValueError, not AssertionError. (HasRadius is not registered in this world.)"""
    world = World(components=[HasPosition])
    world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    with pytest.raises(ValueError, match="not in"):
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

    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([3.0, 4.0], "float32"))  # empties the pos-only pool
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
    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([3.0, 4.0], "float32"))
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
    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([3.0, 4.0], "float32"))
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
    world.get_entity(eid).remove_component(HasVelocity)
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
    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([7.0, 8.0], "float32"))  # position NOT re-passed
    world.update()

    pos_vel = world.pools[world._make_key((HasPosition, HasVelocity))]
    np.testing.assert_array_equal(pos_vel.position[0], [5.0, 6.0])          # carried over automatically
    np.testing.assert_array_equal(pos_vel.velocity[0], [7.0, 8.0])


def test_add_unknown_component_raises():
    """Adding a component the world never registered is rejected eagerly, at the call (cheap synchronous check)."""
    world = World(components=[HasPosition, HasVelocity])        # HasRadius is NOT registered with this world
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    with pytest.raises(KeyError):
        world.get_entity(eid).add_component(HasRadius, radius=np.array([5.0], "float32"))


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
    world.get_entity(ids[1]).add_component(HasVelocity, velocity=np.array([9.0, 9.0], "float32"))  # middle one leaves the pool
    world.update()

    assert sum(len(pool) for pool in world.pools.values()) == 3            # nobody lost
    for eid, expected in zip(ids, ([0, 0], [1, 1], [2, 2])):
        pool, ix = world._eid_to_pool_ix[eid]
        np.testing.assert_array_equal(pool.position[ix], expected)         # each id still points at its own data


def test_add_then_remove_component_round_trips():
    """add_component then remove_component returns the entity to its original archetype, same id, data intact."""
    world = World(components=[HasPosition, HasVelocity])

    eid = world.add_entity(components=(HasPosition,), position=np.array([5.0, 6.0], "float32"))
    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([7.0, 8.0], "float32"))
    world.get_entity(eid).remove_component(HasVelocity)
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

    world.get_entity(mover).remove_component(HasVelocity)                             # mover joins sibling's pos-only pool
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
    world.get_entity(migrated).add_component(HasVelocity, velocity=np.array([4.0, 4.0], "float32"))
    world.update()

    assert world._eid_to_pool_ix[direct][0] is world._eid_to_pool_ix[migrated][0]  # same pool object, no dup archetype
    assert len(world.pools[world._make_key((HasPosition, HasVelocity))]) == 2


def test_migrate_multi_field_component_preserves_all_fields():
    """A component with several fields migrates with every field intact and correctly named in the new pool."""
    world = World(components=[HasPosition, HasBox])

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.get_entity(eid).add_component(HasBox, lo=np.array([0.0, 0.0], "float32"), hi=np.array([4.0, 4.0], "float32"))
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
    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([3.0, 4.0], "float32"))              # operate pre-commit
    world.update()

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool is world.pools[world._make_key((HasPosition, HasVelocity))]
    np.testing.assert_array_equal(pool.position[ix], [1.0, 2.0])
    np.testing.assert_array_equal(pool.velocity[ix], [3.0, 4.0])


# --- task 23 subtask 2: a REJECTED add_entity must not leave bookkeeping behind ------------------------------------
# add_entity bumps _last_id and inserts live_entities[id]=None BEFORE the append. That ordering is safe because
# add_entity validates FIRST -- a bad spawn raises before any bookkeeping happens. Task 44 settled which side owns
# that validation, and it chose this one, so the ordering is now load-bearing rather than incidental: append does
# nothing for ADD_ENTITY but the liveness check, which add_entity has just satisfied by construction. Move the
# validation out of add_entity without also moving the two mutations after the append, and a rejected spawn burns
# an id and leaks a dangling live_entities entry -- `world.get_entity(that_id)` hands back an entity that will
# never exist. These two are that tripwire.

def test_rejected_add_entity_burns_no_id():
    """A refused spawn must not consume an entity id: the next successful add_entity gets the id it would have."""
    world = World(components=[HasPosition])

    with pytest.raises((ValueError, TypeError, KeyError)):
        world.add_entity((HasPosition,), position=np.array([1.0, 2.0, 3.0], "float32"))   # (3,) into a (2,) field

    assert world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32")) == 0   # id 0, not 1


def test_rejected_add_entity_leaves_no_live_entity_and_nothing_staged():
    """A refused spawn must leave live_entities and the buffer exactly as they were -- no dangling handle."""
    world = World(components=[HasPosition])

    with pytest.raises((ValueError, TypeError, KeyError)):
        world.add_entity((HasPosition,), position=np.array([1.0, 2.0, 3.0], "float32"))

    assert list(world.live_entities) == []                  # no id registered -> get_entity(0) raises, correctly
    assert len(world._command_buffer) == 0
    world.update()
    assert len(world.pools) == 0                            # and nothing materialized


# --- task 43 subtask 2 (landed): remove_entity is idempotent WITHIN A TICK ------------------------------------------
# A kill is decided by several systems in one tick (damage, TTL, out-of-bounds), so an already-dead id is a NO-OP
# REQUEST, not a programming error -- it used to raise, and every app wrote the same kill-set helper to avoid it.
# "No-op" means staging NOTHING: two REMOVE_ENTITY commands for one id would blow up at commit on
# _eid_to_pool_ix.pop (world.py:169), so the guard sits in World.remove_entity, BEFORE the append -- the buffer
# can only raise or stage, it has no way to say "do nothing".
#
# The no-op stops at the tick boundary, and that boundary is the POINT. Within a tick, system order is arbitrary,
# so two systems killing the same entity is a race, not a bug. Across an update() the order is explicit, so a dead
# id you are still holding is a STALE REFERENCE, and swallowing it would hide the bug. The library answers "was it
# killed this tick?" from `CommandBuffer.removed_this_tick`, which clears with the buffer -- no id arithmetic, so
# an id never spawned and an id long dead are the same answer: raise.


def test_remove_entity_twice_is_a_noop():
    """Two systems kill the same entity in one tick: the second remove returns silently and stages nothing."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    world.remove_entity(eid)
    world.remove_entity(eid)                                   # must not raise
    assert len(world._command_buffer) == 1                     # and must not stage a second despawn

    world.update()                                             # a doubled command would KeyError here
    assert eid not in world.live_entities and eid not in world._eid_to_pool_ix
    assert len(world.pools) == 0


def test_remove_entity_after_the_despawn_committed_raises():
    """The boundary. One tick later the same call is no longer a race -- the id is a stale reference, and the
    caller has to hear about it. This is the half that separates "two systems agreed" from "you kept a dead id"."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    world.remove_entity(eid)
    world.update()                                             # despawn committed; the tick is over

    with pytest.raises(ValueError):
        world.remove_entity(eid)
    assert len(world._command_buffer) == 0                     # the refusal staged nothing


def test_removed_this_tick_is_dropped_by_update():
    """The mechanism behind the boundary, pinned directly: the set lives and dies with the buffer. If it ever
    outlived an update(), every stale remove would silently become a no-op -- and nothing else would notice."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    world.remove_entity(eid)
    assert world._command_buffer.removed_this_tick == {eid}
    world.update()
    assert world._command_buffer.removed_this_tick == set()


def test_removed_this_tick_agrees_with_the_commands_it_summarizes():
    """It is derived state, maintained by hand in two places (append + clear). Pin it against its own source, so
    a REMOVE_ENTITY that stops updating the set is caught here rather than as a mystery no-op."""
    world = World(components=[HasPosition])
    ids = [world.add_entity(components=(HasPosition,), position=np.array([float(i), 1.0], "float32"))
           for i in range(3)]
    world.update()

    world.remove_entity(ids[0])
    world.remove_entity(ids[2])
    staged = {cmd.entity_id for cmd in world._command_buffer if cmd.command_type == CommandType.REMOVE_ENTITY}
    assert world._command_buffer.removed_this_tick == staged == {ids[0], ids[2]}


@pytest.mark.parametrize("bad_id", [999, -1, -5])
def test_remove_entity_of_an_id_the_world_never_handed_out_raises(bad_id):
    """Never-spawned and long-dead are the SAME answer now: not live, no kill staged this tick -> raise. Negative
    ids are covered by the same rule rather than by a separate bounds check, which is why the rule is one line."""
    world = World(components=[HasPosition])                    # fresh: _last_id == -1
    with pytest.raises(ValueError):
        world.remove_entity(bad_id)

    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()
    world.remove_entity(eid)                                   # a real kill staged alongside must not confuse it
    with pytest.raises(ValueError):
        world.remove_entity(bad_id)
    assert len(world._command_buffer) == 1                     # the refusal staged nothing of its own


def test_remove_entity_accepts_the_numpy_ids_a_query_hands_out():
    """qr.entity_ids is int64. The no-op is a set lookup, so it must not care which integer type it is given --
    otherwise `kill_all(world, qr.entity_ids[mask])` raises on the duplicate kill it is supposed to absorb."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    world.remove_entity(world.query(HasPosition).entity_ids[0])   # np.int64
    world.remove_entity(eid)                                      # python int, same entity -> still a no-op
    assert len(world._command_buffer) == 1
    world.update()
    assert len(world.pools) == 0


def test_remove_entity_of_an_uncommitted_spawn_is_removed_once_and_then_a_noop():
    """The no-op must not eat a LEGAL despawn: killing an entity spawned this tick is add-then-remove in one
    buffer, and only the repeat is dropped."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))

    world.remove_entity(eid)                                   # never committed -> ADD + REMOVE staged together
    world.remove_entity(eid)                                   # the repeat is the no-op
    assert len(world._command_buffer) == 2

    world.update()
    assert len(world.pools) == 0 and len(world.live_entities) == 0


def test_add_component_after_remove_entity_fails():
    """A system removes an entity; a later system tries to widen it the same tick -> reject eagerly."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.remove_entity(eid)
    with pytest.raises(ValueError):
        world.get_entity(eid).add_component(HasVelocity, velocity=np.array([3.0, 4.0], "float32"))


def test_remove_component_after_remove_entity_fails():
    """A system removes an entity; a later system tries to narrow it the same tick -> reject eagerly."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    world.remove_entity(eid)
    with pytest.raises(ValueError):
        world.get_entity(eid).remove_component(HasVelocity)


def test_remove_unknown_entity_id_fails():
    """An id the world never handed out is not live -> remove_entity must reject it at the call, not at commit."""
    world = World(components=[HasPosition])
    with pytest.raises(ValueError):
        world.remove_entity(123)


def test_spawn_into_archetype_reclaimed_by_earlier_despawn_same_tick():
    """Despawn the last entity of an archetype, then spawn a new one of the SAME archetype, same tick.
    The newcomer must land in a live, queryable pool, not an orphaned one.

    #49 changed HOW this holds, and for the better. Reclamation used to happen inside `_pop_from_pool`, mid-commit:
    the despawn deleted the pool and the spawn built a brand-new one behind it. It is now a single sweep at the END
    of update(), so the emptied pool is still there when the spawn arrives and gets REUSED -- see the identity
    assertion below. Same observable result, one less Pool allocation."""
    world = World(components=[HasPosition])
    old = world.add_entity(components=(HasPosition,), position=np.array([2.0, 2.0], "float32"))
    world.update()
    pool_before = world.pools[world._make_key((HasPosition,))]

    world.remove_entity(old)                                       # queued first: empties the pos pool
    new = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    assert old not in world._eid_to_pool_ix
    assert sum(len(pool) for pool in world.pools.values()) == 1    # exactly the newcomer, and it is visible
    assert world._make_key((HasPosition,)) in world.pools          # its pool is live / registered (not orphaned)
    pool, ix = world._eid_to_pool_ix[new]
    np.testing.assert_array_equal(pool.position[ix], [1.0, 1.0])
    assert pool is pool_before                                     # reused, not blinked: no realloc, no new Pool


def test_an_archetype_emptied_and_refilled_in_one_tick_is_never_torn_down():
    """The same property stated as the thing it fixes: a bullet archetype that momentarily hits zero mid-tick used
    to cost a full Pool teardown + rebuild (INITIAL_CAPACITY rows per field, per blink -- the space-shooter run
    measured 747 pool builds in a chaos session). With reclamation deferred to the end of update(), a pool that is
    refilled before the sweep runs is never torn down at all."""
    world = World(components=[HasPosition])
    ids = [world.add_entity((HasPosition,), position=np.array([float(i), 0.0], "float32")) for i in range(3)]
    world.update()
    pool_before = world.pools[world._make_key((HasPosition,))]

    for _ in range(5):                                             # drain to zero and refill, 5 ticks running
        for eid in ids:
            world.remove_entity(eid)
        ids = [world.add_entity((HasPosition,), position=np.array([float(i), 1.0], "float32")) for i in range(3)]
        world.update()
        assert world.pools[world._make_key((HasPosition,))] is pool_before

    assert len(world.pools) == 1 and len(pool_before) == 3
    _assert_pool_ids_invariants(world)


def test_an_archetype_left_empty_at_the_end_of_a_tick_is_reclaimed():
    """The other side of the sweep: if nothing refills it, the pool and BOTH its bookkeeping entries go away.
    A pool left in `pool_to_components` or `_pool_ids` would be a leak the query loop still walks every tick."""
    world = World(components=[HasPosition, HasVelocity])
    a = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity((HasPosition, HasVelocity), position=np.array([3.0, 4.0], "float32"),
                     velocity=np.array([5.0, 6.0], "float32"))
    world.update()
    assert len(world.pools) == 2

    world.remove_entity(a)
    world.update()

    assert len(world.pools) == len(world.pool_to_components) == len(world._pool_ids) == 1
    assert world._make_key((HasPosition,)) not in world.pools      # the emptied one is gone, not just empty
    _assert_pool_ids_invariants(world)


# --- fully-eager staging (task 178, landed) ----------------------------------------------------------------------
# The command buffer is a STAGING area, like git's index: every op is FULLY validated at the call, so only valid
# commands ever enter it and update() is a pure, infallible apply -- it materializes, it does NOT re-validate or
# roll back (NOT atomic). Two halves, both eager:
#   1. structural: dup-add / absent-remove judged against the PROJECTED set (committed + queued adds - queued
#      removes). Same-tick sequences (add->remove->add->remove) are legal; a same-tick self-conflict (add the same
#      component twice) is rejected at the SECOND call -- the poisoning command never enters the buffer.
#   2. field data: dtype / shape / missing-required checked at the call too, mirroring add_entity. No field-data
#      error reaches commit.
# add_entity is the template -- it is already fully eager.


def test_add_then_remove_same_component_same_tick_ok():
    """A queued add must be visible to a later remove the same tick: add(V) then remove(V) is a legal no-op pair,
    committing back to the original archetype. Broken today: remove checks the committed set, can't see the queued
    add, and raises 'does not exist'."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([3.0, 4.0], "float32"))
    world.get_entity(eid).remove_component(HasVelocity)           # must see the queued add -> legal no-op
    world.update()

    assert set(world.get_entity(eid).get_components()) == {HasPosition}
    _assert_pool_ids_invariants(world)


def test_remove_then_add_same_component_same_tick_ok():
    """The mirror: remove(V) then add(V) in one tick is a legal no-op. Broken today: the re-add checks the committed
    set, still sees V, and raises 'already in components'."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    world.get_entity(eid).remove_component(HasVelocity)
    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([5.0, 6.0], "float32"))  # re-add, same tick
    world.update()

    assert set(world.get_entity(eid).get_components()) == {HasPosition, HasVelocity}
    np.testing.assert_array_equal(world.get_entity(eid).velocity, [5.0, 6.0])   # the re-added value wins
    _assert_pool_ids_invariants(world)


def test_add_remove_cycle_same_tick_ok():
    """A longer churn on one entity in a single tick stages cleanly: add/remove/add/remove each validate against the
    running projected set and commit to the expected final archetype."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    e = world.get_entity(eid)
    e.add_component(HasVelocity, velocity=np.array([3.0, 4.0], "float32"))
    e.remove_component(HasVelocity)
    e.add_component(HasVelocity, velocity=np.array([5.0, 6.0], "float32"))
    e.remove_component(HasVelocity)
    world.update()

    assert set(world.get_entity(eid).get_components()) == {HasPosition}
    _assert_pool_ids_invariants(world)


def test_double_add_raises_eagerly_at_second_call():
    """Same-tick self-conflict is caught at the CALL, not deferred to commit: the second add of a component already
    staged raises immediately and does NOT enter the buffer -- so update() never sees the poisoning command."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([3.0, 4.0], "float32"))   # 1st: stages
    with pytest.raises(ValueError):
        world.get_entity(eid).add_component(HasVelocity, velocity=np.array([5.0, 6.0], "float32"))  # 2nd: eager reject
    assert len(world._command_buffer) == 1                       # only the first add is staged


def test_remove_staged_component_twice_raises_eagerly():
    """Removing the same component twice in a tick: the second remove sees it already gone from the projected set
    and rejects at the call -- it never stages a second, invalid REMOVE_COMPONENT."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    world.get_entity(eid).remove_component(HasVelocity)          # 1st: stages the removal
    with pytest.raises(ValueError):
        world.get_entity(eid).remove_component(HasVelocity)      # 2nd: already gone from projected -> eager reject
    assert len(world._command_buffer) == 1


def test_add_component_wrong_shape_raises_eagerly():
    """Field-data validation is eager too (mirroring add_entity): a wrong-shaped value is rejected at the call and
    never staged -- today it slips past the name-only check and detonates in commit's _check_components_against_pool."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    with pytest.raises((ValueError, TypeError)):
        world.get_entity(eid).add_component(HasVelocity, velocity=np.array([1.0, 2.0, 3.0], "float32"))  # (3,) != (2,)
    assert len(world._command_buffer) == 0                       # nothing staged
    world.update()                                               # pure no-op, no crash


def test_add_component_wrong_dtype_raises_eagerly():
    """Same for dtype: a float64 value where the field declares float32 is caught at the call, not at commit."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    with pytest.raises((ValueError, TypeError)):
        world.get_entity(eid).add_component(HasVelocity, velocity=np.array([3.0, 4.0], "float64"))  # float64 != float32
    assert len(world._command_buffer) == 0


def test_add_component_missing_required_field_raises_eagerly():
    """A component field with default=None must be supplied at add; omitting it is rejected at the call, not at
    commit (where today it surfaces as a KeyError deep in materialization)."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    with pytest.raises((KeyError, ValueError)):
        world.get_entity(eid).add_component(HasVelocity)         # velocity has default=None, not provided
    assert len(world._command_buffer) == 0
    world.update()


# --- object-dtype components -------------------------------------------------------------------------------------
# A component field may declare dtype "object": its storage holds arbitrary Python objects (dicts, callbacks, handles)
# by reference rather than numeric data. Everything else (pools, migrations, swap-remove) must treat it like any field.


def test_world_accepts_object_dtype_component():
    """Construction validation allows dtype 'object'; the world records it for the field."""
    world = World(components=[HasLabel])
    assert world.component_to_dtypes[HasLabel] == ["object"]


def test_world_rejects_str_dtype_component():
    """`str` is no longer an allowed dtype. numpy strings are fixed-width (<U n) and a pre-allocated pool
    can't size the width ahead of the data -- writes silently truncate. A field declaring dtype 'str' must
    fail loud at World construction (where the bad field lives), not corrupt data later. Python strings now
    live in dtype='object' (see test_object_field_holds_python_string_and_compares_by_equality)."""
    class HasName(Component):
        name: np.ndarray = field(metadata={"shape": (1,), "dtype": "str", "default": None})

    with pytest.raises(TypeError, match="str not a string or not in"):
        World(components=[HasName])


def test_world_rejects_non_dataclass_component():
    """A component that isn't a dataclass has no fields to lay out -- rejected with the TypeError that names it."""
    class NotADataclass:                                    # not a Component subclass -> no @dataclass applied
        pass

    with pytest.raises(TypeError, match="NotADataclass"):
        World(components=[NotADataclass])


@pytest.mark.parametrize("bad", ["serializable", ("serializable",), {"serializable"}])
def test_world_rejects_non_list_extra_metadata(bad):
    """extra_metadata is a ctor arg -- user input -- so a wrong type is a real `raise` (task 34), not an assert.
    Measured with the assert stripped by -O: a bare str is splatted into single characters and construction dies
    later with `expected_meta = {'i','e','s','z',...}` -- the right rejection for an unreadable reason."""
    with pytest.raises(TypeError):
        World(components=[HasPosition], extra_metadata=bad)


def _two_components_sharing_a_field_name() -> tuple[type[Component], type[Component]]:
    """Two components declaring the SAME field name -- illegal: rejected at World construction (task 23)."""
    class HasFoo(Component):
        payload: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "default": None})

    class HasBar(Component):
        payload: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "default": None})  # same name!

    return HasFoo, HasBar


def test_world_rejects_duplicate_field_name_across_components():
    """Two components may not declare the SAME field name, rejected at construction -- earliest and loudest.

    A pool merges fields by name and query() sums field names across components, so a clash would silently
    alias two components' data. It is also what makes `entity.set_data(**fields)` well-defined (#29): the
    field name alone must identify its component, via world.field_to_component.

    This used to be caught only late, inside update()'s field-dict merge ("Duplicate keys") -- reachable
    because World() accepted the clash. With the construction guard that path is now unreachable through the
    public API, so the merge check in _do_add_component is pure defence-in-depth and has no test of its own."""
    HasFoo, HasBar = _two_components_sharing_a_field_name()

    with pytest.raises(ValueError, match="payload"):
        World(components=[HasFoo, HasBar])


# --- reserved field names: a field named like an internal attr must be rejected at construction --------------------
# A field name that collides with QueryResult's own attrs can never be caught downstream -- QueryResult.__setattr__
# checks the allow-list BEFORE the data-field branch, so `qr.entity_ids = v` replaces the id array instead of
# scattering, and len()/repr()/id lookups then read the written value. World._check_components (world.py:264) is the
# one gate: ENTITY_RESERVED_NAMES | POOL_RESERVED_NAMES | QUERY_RESULT_RESERVED_NAMES, each set being that class's
# instance attrs plus its class dict. Both task-31 gaps are closed -- Pool's names are in the union, and every check
# is a real `raise`, so `python -O` no longer erases them (pinned below).

def _component_with_field(name: str) -> type[Component]:
    """A one-field Component whose field is named `name` (a dataclass field name can't be dynamic in a class body)."""
    return type("HasClash", (Component,), {
        "__annotations__": {name: np.ndarray},
        name: field(metadata={"shape": (1,), "dtype": "float32", "default": None}),
    })


# A field name colliding with a class's own attribute is rejected with ValueError (world.py:277) -- the guard is a
# real raise, so it survives `python -O` (task 31 gap 2, landed).
_REJECTED = ValueError


@pytest.mark.parametrize("name", sorted(QUERY_RESULT_RESERVED_NAMES - POOL_RESERVED_NAMES))
def test_world_rejects_field_name_colliding_with_queryresult_internals(name):
    """A field may not be named like one of QueryResult's own attrs -- rejected at World construction, the earliest
    place that sees every component's fields (and the only place: the collision is invisible to Pool and wins over
    the data branch in QueryResult.__setattr__)."""
    with pytest.raises(_REJECTED, match=name):
        World(components=[_component_with_field(name)])


@pytest.mark.parametrize("name", sorted(POOL_RESERVED_NAMES - QUERY_RESULT_RESERVED_NAMES))
def test_world_rejects_field_name_colliding_with_pool_reserved_names(name):
    """Same rule for POOL's own names -- attrs (size/data/...) AND methods (add_entity/pop_entity/...), since
    Pool.__getattr__ only runs after normal lookup fails. These used to slip past World and blow up much later,
    when the first entity of that archetype commits and Pool rejects at construction -- the late, quiet failure
    this construction-time check exists to prevent. One union, one gate."""
    with pytest.raises(_REJECTED, match=name):
        World(components=[_component_with_field(name)])


def test_world_reserved_name_guard_survives_python_dash_o():
    """The guard must survive `python -O`, which strips asserts. A component definition is user input, so it is
    `raise` territory, not `assert` (asserts are for our own bugs). Before task 31 this was measured broken: under
    -O, World() accepted a field named `entity_ids` and the first write through a QueryResult silently replaced the
    id array. Run in a subprocess because -O is decided at interpreter start."""
    script = ("import numpy as np\n"
              "from dataclasses import field\n"
              "from microecs import World, Component\n"
              "C = type('HasClash', (Component,), {'__annotations__': {'entity_ids': np.ndarray},\n"
              "     'entity_ids': field(metadata={'shape': (1,), 'dtype': 'float32', 'default': None})})\n"
              "World(components=[C])\n")
    env = {**os.environ, "PYTHONPATH": str(Path(microecs.__file__).parent.parent)}

    res = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True, env=env, check=False)

    assert res.returncode != 0, f"-O accepted a field named 'entity_ids'\nstdout: {res.stdout}\nstderr: {res.stderr}"


# --- task 35: World's own kwarg names are a fourth collidable surface ----------------------------------------------
# The union above covers the three CLASSES' attrs+methods. It misses the keyword-PARAMETER names of the World
# methods that take field data as **kwargs, so `**kwargs` shadows the parameter. These four pass World() and only
# die at the first add_entity, with an internal "got multiple values for argument" TypeError that names a parameter
# instead of the offending field -- the late, cryptic failure this gate exists to prevent. `data` and `entity_id`
# are already rejected, but only incidentally (they happen to be Pool / Entity attrs).

_WORLD_KWARG_NAMES = ["check_extra", "component", "components", "strict"]
@pytest.mark.xfail(strict=True, reason="task 35: World's internal kwarg names not reserved yet")
@pytest.mark.parametrize("name", _WORLD_KWARG_NAMES)
def test_world_rejects_field_name_colliding_with_its_own_kwargs(name):
    """A field named like one of World's own **kwargs-taking parameters must be refused at construction."""
    with pytest.raises(_REJECTED, match=name):
        World(components=[_component_with_field(name)])


@pytest.mark.parametrize("name", _WORLD_KWARG_NAMES)
def test_world_kwarg_name_clash_is_currently_a_confusing_late_typeerror(name):
    """Pins the CURRENT bad behaviour so task 35 has a measured before/after: World() accepts the component and
    add_entity dies with a TypeError about a *parameter*. Delete this test when the guard lands (the xfail above
    flips green and construction never reaches add_entity)."""
    world = World(components=[_component_with_field(name)])          # accepted today -- that is the bug

    # "...for argument 'component'" vs "...for keyword argument 'strict'" depending on the shadowed parameter
    with pytest.raises(TypeError, match="got multiple values for"):
        world.add_entity((world.component_types.copy().pop(),), **{name: np.array([1.0], "float32")})


def test_object_field_holds_python_string_and_compares_by_equality():
    """The sanctioned replacement for the removed str dtype: store a python string in a dtype='object'
    field. The full string is kept (no <U width cap, no truncation) and string `==` on the pool column
    works -- element-wise and vectorised."""
    class HasKind(Component):  # e.g. a 'component_kind' tag stored as a string-in-object
        kind: np.ndarray = field(metadata={"shape": (1,), "dtype": "object", "default": None})

    world = World(components=[HasKind])
    a = world.add_entity(components=(HasKind,), kind=np.array(["enemy"], dtype=object))             # idx 0
    b = world.add_entity(components=(HasKind,), kind=np.array(["player_one_long_name"], dtype=object))  # idx 1
    world.update()

    pool, ia = world._eid_to_pool_ix[a]
    _, ib = world._eid_to_pool_ix[b]
    assert pool.kind.dtype == object
    assert pool.kind[ia, 0] == "enemy"                                  # exact string, not truncated to "e"
    assert pool.kind[ib, 0] == "player_one_long_name"                   # full length kept, no <U cap
    np.testing.assert_array_equal(pool.kind[:, 0] == "enemy", [True, False])  # vectorised string equality


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
    world.get_entity(eid).add_component(HasPosition, position=np.array([1.0, 2.0], "float32"))
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
    with pytest.raises(ValueError):                               # a clear ValueError, like the other ops
        world.get_entity(123)


# --- lazy Entity allocation + cache (task 15, part A) ------------------------------------------------------------
# live_entities does two jobs: a liveness registry (id present == live) and an Entity cache. The Entity object is
# NOT built at add_entity (a pure-vectorised sim that never calls get_entity pays zero Entity objects); it is built
# on the FIRST get_entity and cached (same object thereafter), and evicted at remove_entity. These pin that.


def test_add_entity_allocates_no_entity_object():
    """add_entity registers the id as live but stores no Entity -- the slot is None until someone asks for it."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert eid in world.live_entities                              # the id is live (registry job)
    assert world.live_entities[eid] is None                       # but no Entity was built (lazy: pay only on demand)


def test_get_entity_builds_once_then_caches_same_object():
    """First get_entity builds the Entity and stores it; later calls hand back the SAME object (stable identity)."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert world.live_entities[eid] is None                       # nothing built yet
    first = world.get_entity(eid)
    assert world.live_entities[eid] is first                      # cached on first request
    assert world.get_entity(eid) is first                         # repeat call -> same object, not a rebuild


def test_remove_entity_evicts_the_cached_entity():
    """remove_entity drops the id from live_entities, taking the cached Entity with it -- no stale view lingers."""
    world = World(components=[HasPosition])
    eid = world.add_entity(components=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()
    world.get_entity(eid)                                         # populate the cache

    world.remove_entity(eid)                                      # eager eviction from the registry+cache
    assert eid not in world.live_entities
    with pytest.raises(ValueError):                             # a removed id no longer resolves
        world.get_entity(eid)


def test_get_entity_resolves_each_id_to_its_own_object():
    """Two ids never share one cached Entity: each builds its own, keyed by id."""
    world = World(components=[HasPosition])
    a = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))
    b = world.add_entity(components=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    ea, eb = world.get_entity(a), world.get_entity(b)
    assert ea is not eb and ea.entity_id == a and eb.entity_id == b


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
                world.get_entity(eid).add_component(c, **{k: v.copy() for k, v in d.items()})
                shadow[eid].update({k: v.copy() for k, v in d.items()})
        else:                                                        # shrink it (never below one component)
            eid = rng.choice(live)
            have = [c for c in _CHURN_COMPONENTS if _CHURN_COMPONENTS[c][0] in shadow[eid]]
            if len(have) > 1:
                c = rng.choice(have)
                world.get_entity(eid).remove_component(c)
                shadow[eid].pop(_CHURN_COMPONENTS[c][0])
        world.update()

        _assert_pool_ids_invariants(world)
        for eid, fields in shadow.items():
            entity = world.get_entity(eid)
            for name, value in fields.items():
                np.testing.assert_array_equal(getattr(entity, name), value)  # each id keeps its OWN data thru swaps

    assert len(world.live_entities) > 0                            # sanity: the churn left a populated world


# --- #49 item 1: the two despawn paths must not drift ---------------------------------------------------------------
# Dropping the discarded copy added `_remove_from_pool` beside `_pop_from_pool`. The refactor kept the duplication
# small on purpose -- hoisting empty-pool reclamation into a single sweep at the end of update() took the biggest
# shared block out of both -- but they still carry the same pop-swap and id-repointing, and differ only in whether
# the row is copied out first. update() sends REMOVE_ENTITY to the no-copy one and component migration to the
# copying one, so a fix applied to one and not the other silently desyncs half the churn paths. Nothing else
# compares them directly, so this does.


def _world_shape(world: World) -> dict:
    """Everything about the world's structure that a despawn can move, in a comparable form (pools are keyed by
    archetype rather than by object identity, since the two paths build different Pool instances)."""
    by_key = {world._make_key(cs): p for p, cs in world.pool_to_components.items()}
    return {
        "live": sorted(world.live_entities),
        "pool_keys": sorted(world.pools),
        "ids_per_pool": {k: list(world._pool_ids[p]) for k, p in by_key.items()},
        "rows": {(k, f): p.data[f][0:len(p)].tolist() for k, p in by_key.items() for f in p.fields},
        "eid_rows": {eid: (world._make_key(world.pool_to_components[p]), ix)
                     for eid, (p, ix) in world._eid_to_pool_ix.items()},
    }


@pytest.mark.parametrize("victims", [(0,), (4,), (2,), (0, 1, 2), (4, 3, 2), (2, 0, 4), (0, 1, 2, 3, 4)],
                         ids=["first", "last", "middle", "front-run", "back-run", "scattered", "drain-pool"])
def test_the_two_despawn_paths_leave_identical_world_state(victims):
    """Same removals, once through each path -- head, tail, middle, and a full drain that reclaims the pool.
    Any divergence in the pop-swap, the id repointing, or the empty-pool cleanup shows up as a state mismatch."""
    def _run(path_name):
        world = World(components=[HasPosition, HasVelocity])
        ids = [world.add_entity((HasPosition, HasVelocity),
                                position=np.array([float(i), float(i)], "float32"),
                                velocity=np.array([float(-i), float(i)], "float32")) for i in range(5)]
        world.add_entity((HasPosition,), position=np.array([99.0, 99.0], "float32"))   # a 2nd pool, untouched
        world.update()
        for v in victims:
            getattr(world, path_name)(ids[v])
            del world.live_entities[ids[v]]        # what World.remove_entity does around the pool call
        return _world_shape(world)

    assert _run("_remove_from_pool") == _run("_pop_from_pool")


def test_the_last_despawn_in_a_world_leaves_no_bookkeeping_behind():
    """Drain the world entirely through the no-copy path: every map must end empty. Neither despawn path reclaims
    pools any more -- the end-of-update sweep does -- so this pins that the sweep covers the no-copy path too, and
    that nothing is left dangling in `pool_to_components` / `_pool_ids` / `_eid_to_pool_ix`."""
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()
    assert len(world.pools) == 1

    world.remove_entity(eid)
    world.update()

    assert world.pools == {} and world.pool_to_components == {} and world._pool_ids == {}
    assert world._eid_to_pool_ix == {} and world.live_entities == {}


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
    """entity_ids is a real ndarray, not a Field: entity-axis indexing and fancy ops that Field rejects --
    qr.entity_ids[i], slicing, np.isin -- all work, because ids are materialized by World, not a per-pool view."""
    world = World(components=[HasPosition])
    ids = [world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32")) for i in range(4)]
    world.update()

    qr = world.query(HasPosition)

    assert int(qr.entity_ids[0]) in ids                            # entity-axis index -> allowed (unlike Field)
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


# Derived from the library's own constant, like _ENTITY_RESERVED below -- NOT from vars(<a throwaway instance>).
# An instance dict only sees attrs __init__ assigns, so the moment an attr becomes a property (as `entity_ids` did
# when it went lazy) it silently drops out of the parametrize and stops being covered here. The constant is built
# from _QR_INTERNAL_ATTRS | vars(QueryResult) -- the class dict -- so it keeps up.
_QUERYRESULT_RESERVED = sorted(QUERY_RESULT_RESERVED_NAMES)
@pytest.mark.parametrize("reserved", _QUERYRESULT_RESERVED)
def test_world_rejects_component_field_named_like_a_queryresult_attribute(reserved):
    """A component whose field is named like a QueryResult attribute must be rejected at world creation, rather
    than be silently shadowed when queried."""
    bad = type("Bad", (Component,), {"__annotations__": {reserved: np.ndarray},
                                      reserved: field(metadata={"shape": (2,), "dtype": "float32", "default": None})})
    with pytest.raises(_REJECTED):
        World(components=[bad])


# An Entity (world.get_entity(id)) exposes the row by attribute, so a component field named like one of Entity's
# own members would be shadowed: e.entity_id would return the id (not the field), e.get_components a bound method.
# These must be rejected at world creation. Entity isn't slotted, so its collidable surface is split in two:
# instance-attr names in the private _ENTITY_INTERNAL_ATTRS, public methods in the class dict. ENTITY_RESERVED_NAMES
# is the union of both, derived at import (not hardcoded), so new attrs/methods are picked up automatically.
_ENTITY_RESERVED = sorted(ENTITY_RESERVED_NAMES)
@pytest.mark.parametrize("reserved", _ENTITY_RESERVED)
def test_world_rejects_component_field_named_like_an_entity_attribute(reserved):
    """A component whose field is named like an Entity attribute/method must be rejected at world creation,
    rather than be silently shadowed when read/written through get_entity."""
    bad = type("Bad", (Component,), {"__annotations__": {reserved: np.ndarray},
                                      reserved: field(metadata={"shape": (2,), "dtype": "float32", "default": None})})
    with pytest.raises(_REJECTED):
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
        a: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})

    class Serial(Component):   # same field, plus the extra "serializable" key
        b: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "serializable": True, "default": None})

    World([Plain])                                          # ok: {shape,dtype} == {shape,dtype}
    World([Serial], extra_metadata=["serializable"])  # ok: {shape,dtype,ser} == {shape,dtype,ser}

    with pytest.raises(ValueError):                         # missing the required "serializable"
        World([Plain], extra_metadata=["serializable"])
    with pytest.raises(ValueError):                         # carries "serializable" the world never declared
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

    world.get_entity(eid).add_component(Frozen)                        # tag carries no data
    world.update()
    assert world.query(Frozen).entity_ids.tolist() == [eid]
    entity = world.get_entity(eid)
    np.testing.assert_array_equal(entity.position, [5.0, 6.0])  # data preserved across the migration
    assert Frozen in entity.get_components()

    world.get_entity(eid).remove_component(Frozen)
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


def test_add_entity_wrong_dtype_crashes_eagerly():
    """A field declared float32 must reject an int32 array *at the add_entity call* (see task 170).

    Field-name validation is eager (bad name crashes at the call) and, since task 170, dtype is too:
    _check_components_against_pool now validates dtype against component metadata, so a wrong dtype
    crashes at the call instead of slipping through to pool.add_entity at world.update()."""
    world = World(components=[HasRadius])  # HasRadius.radius is shape (1,) dtype float32

    # name + shape are correct here, so the only thing that can crash at the call is the dtype mismatch
    with pytest.raises(TypeError):
        world.add_entity((HasRadius,), radius=np.zeros((1,), "int32"))  # wrong dtype must crash here


def test_add_entity_wrong_shape_crashes_eagerly():
    """A field declared shape (1,) must reject a (2,) array *at the add_entity call* (see task 170).

    Companion to test_add_entity_wrong_dtype_crashes_eagerly: shape is the other half of the same guard.
    Since task 170, _check_components_against_pool validates shape against component metadata, so a wrong
    shape crashes at the call instead of slipping through to pool.add_entity at world.update()."""
    world = World(components=[HasRadius])  # HasRadius.radius is shape (1,) dtype float32

    # name + dtype are correct here, so the only thing that can crash at the call is the shape mismatch
    with pytest.raises(ValueError):
        world.add_entity((HasRadius,), radius=np.zeros((2,), "float32"))  # wrong shape must crash here


def test_add_entity_missing_required_field_crashes_eagerly():
    """A field with default=None is REQUIRED at spawn: omitting it raises at the add_entity call, not at commit
    (where it would surface as a KeyError deep in pool materialization, mid-loop and un-attributable)."""
    world = World(components=[HasPosition])  # HasPosition.position has default=None

    with pytest.raises(KeyError):
        world.add_entity((HasPosition,))     # position not supplied and has no default

    assert list(world.live_entities) == []
    assert len(world._command_buffer) == 0


def test_add_entity_rejects_a_non_ndarray_field():
    """Field values are np.ndarray, full stop -- a list that merely looks array-like is refused at the call. Pools
    are typed columns, so a list would be silently coerced (or worse, stored as object) at commit."""
    world = World(components=[HasPosition])

    with pytest.raises(TypeError):
        world.add_entity((HasPosition,), position=[1.0, 2.0])   # a list, not an ndarray

    assert list(world.live_entities) == []
    assert len(world._command_buffer) == 0


# --- task 44 (landed): the spawn path validates ONCE ----------------------------------------------------------------
# add_entity ran _validate_components + _defaults_for, then CommandBuffer.append ran BOTH again on the same args --
# a superset-check of the check that had just happened, whose _defaults_for could only return {}. ~2.0 us/spawn:
# 34% of add_entity, 20% of a full spawn, 16% of a spawn+despawn pair, landing on w5 churn (the one workload
# microecs loses at every N). Fixed by giving the verb ONE owner: World.add_entity validates, and append -- whose
# only ADD_ENTITY producer is world.py:82 -- stages it verbatim.
#
# These two are the regression guard, and they catch BOTH directions: a re-added pass in append (the bug that was
# removed) and a second pass anywhere else on the spawn path. Everything else about the change is invisible, which
# is exactly why a plain behaviour test cannot protect it -- the counter is the test.


def _count_calls(world, method_name):
    """Wrap a bound method on this world instance with a call counter. The buffer reaches the world through
    `self.world`, i.e. the same instance -- so a second pass from either file lands in the same list."""
    calls = []
    original = getattr(world, method_name)

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    setattr(world, method_name, counted)
    return calls


def test_spawn_validates_exactly_once():
    """One spawn, one validation pass. The whole point of task 44."""
    world = World([HasPosition])
    calls = _count_calls(world, "_validate_components")

    world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert len(calls) == 1, f"spawn validated {len(calls)}x -- the duplicate pass is back"


def test_spawn_computes_defaults_exactly_once():
    """Same for the defaults pass, on the archetype that actually has one to fill -- and the default still lands.
    The second pass ran on the ALREADY-merged kwargs, so it could only ever return {}: pure cost, no effect."""
    world = World([HasColorDefault])
    calls = _count_calls(world, "_defaults_for")

    world.add_entity((HasColorDefault,))                        # color omitted -> filled once, by add_entity
    world.update()

    assert len(calls) == 1, f"defaults computed {len(calls)}x -- the duplicate pass is back"
    pool = world.query(HasColorDefault).pool_list[0]
    np.testing.assert_array_equal(pool.color[0], np.array([10, 20, 30], "int32"))


def test_rejected_spawn_validates_exactly_once_too():
    """A REJECTED spawn must not validate twice either: it raises on the first pass and stops there -- no retry,
    no second opinion. (Pins that the fix did not just move the duplicate behind the happy path.)"""
    world = World([HasPosition])
    calls = _count_calls(world, "_validate_components")

    with pytest.raises((ValueError, TypeError, KeyError)):
        world.add_entity((HasPosition,), position=np.array([1.0, 2.0, 3.0], "float32"))   # (3,) into a (2,) field

    assert len(calls) == 1


# --- task 43 subtask 1 (landed): duplicate components at spawn ------------------------------------------------------
# _validate_components de-duped into a set for its checks, then _get_entity_pool iterated the LIST -- so a dup
# built pool.fields == ['position', 'position']: every add/remove/realloc did that column twice, forever, and
# to_dict() reported the malformed component list back out. The archetype key is a bitmask, so the FIRST caller's
# argument list decided the pool's shape: later well-formed spawns of the same archetype landed in the bad pool.
# Closed by one `len(cs) != len(components)` raise (world.py:232), beside the empty/unknown checks.


def test_add_entity_rejects_duplicate_components():
    """`add_entity([Pos, Pos, Vel])` is a malformed archetype -- rejected at the call, like the empty one."""
    world = World(components=[HasPosition, HasVelocity])

    with pytest.raises(ValueError, match="Duplicate components"):
        world.add_entity((HasPosition, HasPosition, HasVelocity),
                         position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))


def test_rejected_duplicate_component_spawn_leaves_nothing_behind():
    """The refusal must be complete: no id burnt, no dangling live_entities handle, nothing staged or built."""
    world = World(components=[HasPosition, HasVelocity])

    with pytest.raises(ValueError):
        world.add_entity((HasPosition, HasPosition), position=np.array([1.0, 2.0], "float32"))

    assert list(world.live_entities) == []
    assert len(world._command_buffer) == 0
    world.update()
    assert len(world.pools) == 0

    # and the clean spawn that follows gets id 0 and a single-column pool
    eid = world.add_entity((HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.update()
    assert eid == 0
    assert world._eid_to_pool_ix[eid][0].fields == ["position"]


def test_query_with_a_repeated_component_is_harmless():
    """Control for the two above: a dup in a QUERY is de-duped by the bitmask key, and must stay legal -- the new
    spawn-side raise must not leak into the read side."""
    world = World(components=[HasPosition, HasVelocity])
    world.add_entity((HasPosition, HasVelocity),
                     position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    assert len(world.query(HasPosition, HasPosition).entity_ids) == 1
    np.testing.assert_array_equal(world.query(HasPosition, HasPosition).position.numpy(), [[1.0, 2.0]])


# --- default metadata (task 171): omitted fields fall back to the component's declared default ---

def test_add_entity_fills_default_when_field_omitted():
    """Omitting a field whose metadata declares a (non-None) default fills it with that default."""
    world = World([HasColorDefault])

    world.add_entity((HasColorDefault,))  # color omitted -> default [10, 20, 30]
    world.update()

    pool = world.query(HasColorDefault).pool_list[0]
    np.testing.assert_array_equal(pool.color[0], np.array([10, 20, 30], "int32"))


def test_add_entity_explicit_value_overrides_default():
    """An explicit value for a defaulted field wins over the default."""
    world = World([HasColorDefault])

    world.add_entity((HasColorDefault,), color=np.array([1, 2, 3], "int32"))
    world.update()

    pool = world.query(HasColorDefault).pool_list[0]
    np.testing.assert_array_equal(pool.color[0], np.array([1, 2, 3], "int32"))


def test_default_and_explicit_coexist_per_row():
    """Same pool: an omitted field takes the default while an explicit one keeps its own value, per row."""
    world = World([HasColorDefault])

    world.add_entity((HasColorDefault,))                                       # row 0: default
    world.add_entity((HasColorDefault,), color=np.array([1, 2, 3], "int32"))   # row 1: explicit
    world.update()

    pool = world.query(HasColorDefault).pool_list[0]
    np.testing.assert_array_equal(pool.color[0], np.array([10, 20, 30], "int32"))
    np.testing.assert_array_equal(pool.color[1], np.array([1, 2, 3], "int32"))


def test_component_default_wrong_dtype_rejected():
    """A default whose dtype mismatches the declared dtype is rejected (see task 171).

    Filling a default runs it through the same dtype check as an explicit value, so an int32 default for
    a float32 field raises TypeError -- at World() construction if validated there, else when the default
    is filled at the add_entity call. (Assumes the same raise-TypeError convention as the explicit-value
    dtype check; tighten if the construction-time guard chooses a different exception.)"""
    class BadDtypeDefault(Component):
        x: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "default": np.zeros((1,), "int32")})

    with pytest.raises(TypeError):
        world = World([BadDtypeDefault])
        world.add_entity((BadDtypeDefault,))  # omit x -> fill int32 default -> dtype mismatch


def test_component_default_wrong_shape_rejected():
    """A default whose shape mismatches the declared shape is rejected (see task 171).

    Companion to test_component_default_wrong_dtype_rejected: a (2,) default for a (1,) field raises
    ValueError -- at World() construction if validated there, else when the default is filled at the
    add_entity call."""
    class BadShapeDefault(Component):
        x: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "default": np.zeros((2,), "float32")})

    with pytest.raises(ValueError):
        world = World([BadShapeDefault])
        world.add_entity((BadShapeDefault,))  # omit x -> fill (2,) default -> shape mismatch


def test_add_component_fills_default_when_field_omitted():
    """add_component fills a defaulted field that's omitted, exactly like add_entity (see task 171)."""
    world = World([HasPosition, HasColorDefault])
    eid = world.add_entity((HasPosition,), position=np.array([1, 2], "float32"))
    world.update()

    world.get_entity(eid).add_component(HasColorDefault)  # color omitted -> default [10, 20, 30]
    world.update()

    pool = world.query(HasPosition, HasColorDefault).pool_list[0]
    np.testing.assert_array_equal(pool.position[0], np.array([1, 2], "float32"))  # existing field preserved
    np.testing.assert_array_equal(pool.color[0], np.array([10, 20, 30], "int32"))  # new field defaulted


def test_add_component_explicit_value_overrides_default():
    """An explicit value passed to add_component wins over the field's default."""
    world = World([HasPosition, HasColorDefault])
    eid = world.add_entity((HasPosition,), position=np.array([1, 2], "float32"))
    world.update()

    world.get_entity(eid).add_component(HasColorDefault, color=np.array([1, 2, 3], "int32"))
    world.update()

    pool = world.query(HasPosition, HasColorDefault).pool_list[0]
    np.testing.assert_array_equal(pool.color[0], np.array([1, 2, 3], "int32"))


# --- field -> component map: what makes entity.set_data(**fields) possible (microecs #29) -------------------------
# set_data takes field names, no component arg, so the World must be able to answer "which component owns field
# 'pose'?". That lookup is world.field_to_component, built at __init__ and keyed by field NAME. It is only
# well-defined if names are unique across components -- so a duplicate name is rejected at world creation
# (microecs #23). Two things to pin: the map is keyed by str (not by dataclasses.Field, which hashes by identity
# and would make both the lookup and the duplicate check silently useless), and the duplicate actually raises.

def test_field_to_component_is_keyed_by_field_name():
    """world.field_to_component maps 'field name' -> owning component type, for every field of every component."""
    world = World([HasPosition, HasBox])

    assert world.field_to_component["position"] is HasPosition
    assert world.field_to_component["lo"] is HasBox
    assert world.field_to_component["hi"] is HasBox                     # every field, not just the first
    assert set(world.field_to_component) == {"position", "lo", "hi"}     # plain strings, nothing else


def test_field_to_component_resolves_the_name_entity_set_data_passes():
    """The map is the resolution set_data does: a field name reaches the component that owns it -- which is how
    one call can span several components and still validate each field against the right schema (#42: the write
    itself is eager, so the proof is the value in the pool, no longer a SET_DATA command carrying the type)."""
    world = World([HasPosition, HasVelocity])
    eid = world.add_entity((HasPosition, HasVelocity),
                           position=np.array([1, 2], "float32"), velocity=np.array([3, 4], "float32"))
    world.update()

    assert world.field_to_component["velocity"] is HasVelocity
    world.get_entity(eid).set_data(velocity=np.array([9, 8], "float32"))

    np.testing.assert_array_equal(world.get_entity(eid).velocity, [9, 8])
    assert len(world._command_buffer) == 0


def test_world_accepts_the_same_component_field_names_in_separate_worlds():
    """The uniqueness rule is per-World, not global: the same component can be reused in another world."""
    World([HasPosition, HasVelocity])
    World([HasPosition, HasBox])                                        # no leakage between worlds


# --- task 41: a failed update() must not brick the world ------------------------------------------------------------
# update() is deliberately NOT atomic (#22), which is a choice about *partial application* -- not a licence to leave
# the buffer unusable. Today a mid-loop failure leaves it intact, with `args["components"]` already popped off the
# command that was being applied, so EVERY later update() raises `KeyError: 'components'`. The world is bricked, and
# the error names a dict key rather than the cause. An app that logs-and-continues (any server loop) spins on it.
#
# The trigger below is public API, not a monkeypatch: #39's aliasing. add_entity holds YOUR array by reference, so a
# resize after staging slips a shape past the validation that already approved it. #39 (snapshot staged writes) is
# what removes the trigger; task 41 is what stops the failure from being permanent -- both are wanted.
# Note the failure is the same under `python -O`: the tripped guard is an assert, but numpy's broadcast raises
# anyway, hence the (AssertionError, ValueError) pair.

def _stage_a_spawn_that_will_fail_at_commit(world):
    """Stage two spawns where the FIRST one's array is resized behind the buffer's back. Returns nothing: the
    point is the world's state afterwards."""
    arr = np.zeros(2, "float32")
    world.add_entity((HasPosition,), position=arr)                  # accepted: shape (2,) at the call
    world.add_entity((HasVelocity,), velocity=np.zeros(2, "float32"))   # an innocent command behind it
    arr.resize(3, refcheck=False)                                   # public numpy; the staged row is now (3,)


def test_failed_update_propagates_the_original_error():
    """Control for the three xfails below: whatever the fix does, it must not swallow the failure. Green today."""
    world = World(components=[HasPosition, HasVelocity])
    _stage_a_spawn_that_will_fail_at_commit(world)

    with pytest.raises((AssertionError, ValueError)):
        world.update()


@pytest.mark.xfail(strict=True, reason="task 41: the buffer survives a failed update(), commands and all")
def test_failed_update_leaves_an_empty_buffer():
    """A buffer that outlives its own failed commit is a buffer whose commands may apply twice."""
    world = World(components=[HasPosition, HasVelocity])
    _stage_a_spawn_that_will_fail_at_commit(world)

    with pytest.raises((AssertionError, ValueError)):
        world.update()
    assert len(world._command_buffer) == 0


@pytest.mark.xfail(strict=True, reason="task 41: the popped `components` key bricks every later update()")
def test_update_after_a_failed_update_is_a_clean_noop():
    """The headline. One bad frame must cost one frame -- not the world. Today the second update() raises
    `KeyError: 'components'`, and so does the third, and every one after it."""
    world = World(components=[HasPosition, HasVelocity])
    _stage_a_spawn_that_will_fail_at_commit(world)

    with pytest.raises((AssertionError, ValueError)):
        world.update()

    world.update()                                                  # must not raise, and must do nothing
    world.update()
    assert len(world._command_buffer) == 0


@pytest.mark.xfail(strict=True, reason="task 41: _cache is only dropped after the loop, so a partial apply keeps it")
def test_failed_update_drops_the_query_cache():
    """A partial apply moved rows, so every cached QueryResult is suspect -- whether or not the loop finished."""
    world = World(components=[HasPosition, HasVelocity])
    world.add_entity((HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.update()
    world.query(HasPosition)                                        # populate the cache
    assert len(world._cache) == 1

    _stage_a_spawn_that_will_fail_at_commit(world)
    with pytest.raises((AssertionError, ValueError)):
        world.update()
    assert len(world._cache) == 0
