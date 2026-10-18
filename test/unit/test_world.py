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

add_entity is the template: it is fully eager (world.py:78) and it is the SOLE validator of a spawn --
CommandBuffer.append does not repeat the pass. add_component /
remove_component have no such single owner (Entity queues a bare command), so for them append IS the gate.
Those buffer-level unit tests live in test_command_buffer.py; the eager-id-tracking tests below exercise the
same staging model through the Entity/World API.

Structural transitions use the `formal` state language: `setup` makes a state, `act` applies a flat action
list and commits, and a state compares directly to a plain dict of component lists. Data/query/cache
tests stay direct.
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
from microecs import World, Component, f32, i32, obj
from microecs.command_buffer import CommandType
from microecs.query_result import QUERY_RESULT_RESERVED_NAMES
from microecs.entity import ENTITY_RESERVED_NAMES
from microecs.pool import POOL_RESERVED_NAMES

from formal import ActError, act, addc, adde, assert_pool_ids_invariants, rmc, rme, setup


class C1(Component):  # state-transition components: C1/C2 default data, C3 tag, C4 required, C5 two fields
    c1: np.ndarray = f32((2,), default=[0.0, 0.0])


class C2(Component):
    c2: np.ndarray = f32((2,), default=[0.0, 0.0])


class C3(Component):
    pass


class C4(Component):
    c4: np.ndarray = f32((2,))


class C5(Component):
    c5a: np.ndarray = f32((2,), default=[0.0, 0.0])
    c5b: np.ndarray = f32((2,), default=[0.0, 0.0])


class C6(Component):
    c6: np.ndarray = obj((1,))


class C7(Component):  # never registered with a world
    pass


class C8(Component):
    position: np.ndarray = f32((2,))


class C9(Component):
    velocity: np.ndarray = f32((2,))


class C10(Component):
    radius: np.ndarray = f32((1,))


class C11(Component):  # two fields, to exercise multi-field merge/ordering across migrations
    lo: np.ndarray = f32((2,))
    hi: np.ndarray = f32((2,))


class C12(Component):  # object-dtype field: holds one arbitrary Python object per entity
    label: np.ndarray = obj((1,))


class C13(Component):  # zero-field tag: a marker with no data, queried/filtered on as an archetype bit
    pass


class C14(Component):  # 0-d array field: exactly one scalar per entity (shape ())
    scale: np.ndarray = f32(())


class C15(Component):  # optional field: metadata carries a real default, filled when omitted
    color: np.ndarray = i32((3,), default=[10, 20, 30])


_COMPONENTS = [C1, C2, C3, C4, C5, C6]


def test_add_entity_rejects_field_from_an_unrequested_component():
    """An entity declared with only C8 may not pass `velocity` (a field of the unrequested C9).
    Validation is eager: the bad field crashes at the add_entity call, before any update()."""
    world = World(components=[C8, C9])  # both components known to the world

    with pytest.raises(ValueError, match="velocity"):
        world.add_entity(
            components=(C8,),                  # entity declares C8 only
            position=np.array([1.0, 2.0], "float32"),   # required by C8
            velocity=np.array([3.0, 4.0], "float32"),   # extra: belongs to C9, not requested
        )


def test_fresh_world_has_no_pools():
    """A world creates pools lazily; an empty state has none."""
    state = setup({}, _COMPONENTS)
    assert state.world.pools == {}


def test_add_one_entity_creates_exactly_one_pool_with_one_entity():
    """One entity creates a single pool holding that single entity."""
    state = setup({0: [C1]}, _COMPONENTS)

    assert len(state.world.pools) == 1
    pool = next(iter(state.world.pools.values()))
    assert len(pool) == 1


def test_entity_lands_in_the_pool_keyed_by_its_components():
    """The entity goes into the pool whose key is exactly the bitmask of its declared components."""
    state = setup({0: [C1]}, _COMPONENTS)

    key = state.world._make_key((C1,))
    assert key in state.world.pools                                   # a pool with that exact key exists
    assert state.world.pools[key] is state.world.query(C1).pool_list[0]     # and query finds the same pool


def test_added_entity_field_values_are_stored():
    """The component data we passed is readable back from the pool, unchanged."""
    world = World(components=[C8, C9])

    world.add_entity(components=(C8,), position=np.array([1.5, 2.5], "float32"))
    world.update()

    pool = world.query(C8).pool_list[0]
    np.testing.assert_array_equal(pool.position[0], np.array([1.5, 2.5], "float32"))


def test_same_archetype_entities_share_a_single_pool():
    """N entities with the same component set all land in one pool, in insertion order."""
    state = setup({0: [C1], 1: [C1], 2: [C1]}, _COMPONENTS)

    assert len(state.world.pools) == 1                                # still just one archetype
    pool = state.world.query(C1).pool_list[0]
    assert len(pool) == 3
    assert state.world._pool_ids[pool] == [0, 1, 2]                   # insertion order


def test_distinct_archetypes_get_distinct_pools():
    """Entities with different component sets are stored in separate pools."""
    state = setup({0: [C1], 1: [C1, C2]}, _COMPONENTS)

    assert len(state.world.pools) == 2
    assert len(state.world.pools[state.world._make_key((C1,))]) == 1
    assert len(state.world.pools[state.world._make_key((C1, C2))]) == 1


def test_component_order_does_not_create_a_second_pool():
    """(C1, C2) and (C2, C1) are the same archetype."""
    state = setup({0: [C1, C2], 1: [C2, C1]}, _COMPONENTS)

    assert len(state.world.pools) == 1                                # key is a bitmask, order-independent
    assert len(next(iter(state.world.pools.values()))) == 2


def test_query_returns_all_pools_that_are_supersets():
    """query(C8) returns every pool containing C8, not just the pos-only one."""
    world = World(components=[C8, C9, C10])

    world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(C8, C9),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))
    world.update()

    assert len(world.query(C8)) == 2            # both pools contain C8
    assert len(world.query(C8, C9)) == 1  # only the richer pool has both


def test_query_is_empty_when_no_pool_has_the_component():
    """Querying a component that no existing pool carries returns no pools."""
    world = World(components=[C8, C9])

    world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert len(world.query(C9)) == 0


def test_query_exclude_drops_pools_that_carry_the_excluded_component():
    """exclude=[C9] keeps only pools that have C8 AND lack C9: the pos-only entity
    stays, the pos+vel one is filtered out. Without the exclude both would match."""
    world = World(components=[C8, C9])
    a = world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(C8, C9),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))
    world.update()

    assert len(world.query(C8)) == 2                       # both pools have C8
    narrowed = world.query(C8, exclude=[C9])     # ...but only one lacks C9
    assert narrowed.entity_ids.tolist() == [a]


def test_query_exclude_tag_component():
    """The common case: a tag as the exclude filter. query(C8, exclude=[C13]) skips every
    frozen entity and returns only the un-tagged ones."""
    world = World(components=[C8, C13])
    moving = world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(C8, C13), position=np.array([3.0, 4.0], "float32"))
    world.update()

    qr = world.query(C8, exclude=[C13])
    assert qr.entity_ids.tolist() == [moving]                         # the frozen one is excluded


def test_query_exclude_unmatched_component_is_a_noop():
    """Excluding a component that no matching pool carries removes nothing: the result equals the un-excluded
    query. exclude only ever subtracts, it can't invent matches."""
    world = World(components=[C8, C9])
    a = world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    b = world.add_entity(components=(C8,), position=np.array([3.0, 4.0], "float32"))
    world.update()

    qr = world.query(C8, exclude=[C9])          # no pos-only pool has velocity
    assert sorted(qr.entity_ids.tolist()) == sorted([a, b])


def test_query_exclude_multiple_components():
    """exclude is satisfied only by pools that carry NONE of the excluded bits. With four archetypes, excluding
    both C9 and C13 leaves the single pool that has neither."""
    world = World(components=[C8, C9, C13])
    a = world.add_entity(components=(C8,), position=np.array([1.0, 1.0], "float32"))
    world.add_entity(components=(C8, C9),
                     position=np.array([2.0, 2.0], "float32"), velocity=np.array([0.0, 0.0], "float32"))
    world.add_entity(components=(C8, C13), position=np.array([3.0, 3.0], "float32"))
    world.add_entity(components=(C8, C9, C13),
                     position=np.array([4.0, 4.0], "float32"), velocity=np.array([0.0, 0.0], "float32"))
    world.update()

    qr = world.query(C8, exclude=[C9, C13])
    assert qr.entity_ids.tolist() == [a]                             # only the pos-only pool survives both filters


def test_query_include_and_exclude_are_distinct_cache_entries():
    """The cache key is (include, exclude), so the same include with and without an exclude are independent
    entries: each re-query returns its OWN cached object, never the other and never the cache dict itself."""
    world = World(components=[C8, C9])
    world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(components=(C8, C9),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))
    world.update()

    wide = world.query(C8)                               # no exclude -> exclude_key 0
    narrow = world.query(C8, exclude=[C9])     # same include, different exclude

    assert wide is not narrow                                        # distinct keys -> distinct entries
    assert len(wide) == 2 and len(narrow) == 1
    assert world.query(C8) is wide                       # each key keeps returning its own result
    assert world.query(C8, exclude=[C9]) is narrow


def test_query_exclude_none_matches_empty_exclude():
    """exclude=None is the default and means 'exclude nothing' -- identical key to exclude=[], so both hit the
    same cache entry and return the same object."""
    world = World(components=[C8])
    world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert world.query(C8, exclude=None) is world.query(C8, exclude=[])


def test_query_exclude_spans_multiple_pools():
    """exclude can leave more than one surviving archetype. With {Pos}, {Pos,Vel}, {Pos,C13}, excluding
    C13 keeps the first two pools; a vectorised write through qr.position scatters back per pool and
    entity_ids stays aligned with the rows."""
    world = World(components=[C8, C9, C13])
    a = world.add_entity(components=(C8,), position=np.array([1.0, 1.0], "float32"))
    b = world.add_entity(components=(C8, C9),
                         position=np.array([2.0, 2.0], "float32"), velocity=np.array([0.0, 0.0], "float32"))
    world.add_entity(components=(C8, C13), position=np.array([3.0, 3.0], "float32"))
    world.update()

    qr = world.query(C8, exclude=[C13])
    assert len(qr.pool_list) == 2                              # two archetypes survive the C13 exclusion
    assert sorted(qr.entity_ids.tolist()) == sorted([a, b])    # the C13 entity is gone

    qr.position[:] = qr.position + 10.0                        # vectorised write across both surviving pools
    for eid, pos in zip(qr.entity_ids, qr.position):           # scattered back, id-aligned per pool
        np.testing.assert_array_equal(world.get_entity(int(eid)).position, pos)


def test_query_exclude_cache_invalidated_on_mutation():
    """The excluding query is cached, but a mutating update() drops it: after a C13 entity is added and
    committed, re-querying with exclude=[C13] is a fresh object that still excludes it."""
    world = World(components=[C8, C13])
    a = world.add_entity(components=(C8,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    before = world.query(C8, exclude=[C13])
    assert before.entity_ids.tolist() == [a]

    world.add_entity(components=(C8, C13), position=np.array([2.0, 2.0], "float32"))
    world.update()                                             # mutating commit -> cache cleared
    after = world.query(C8, exclude=[C13])

    assert after is not before                                 # fresh object, not the stale cached one
    assert after.entity_ids.tolist() == [a]                    # the new C13 entity is still excluded


def test_query_exclude_unregistered_component_raises():
    """Excluding a component the world never registered is an error, surfaced by _make_key's raise -- the
    same guard that protects the include side. A query is user input, so it must survive `python -O`:
    ValueError, not AssertionError. (C10 is not registered in this world.)"""
    world = World(components=[C8])
    world.add_entity(components=(C8,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    with pytest.raises(ValueError, match="not in"):
        world.query(C8, exclude=[C10])


def test_query_exclude_contradiction_is_empty():
    """include and exclude overlapping is a contradiction: a pool can't both have and lack C8, so the
    result is empty -- not a crash. (archetype & exclude) != 0 for every otherwise-matching pool."""
    world = World(components=[C8])
    world.add_entity(components=(C8,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    qr = world.query(C8, exclude=[C8])
    assert len(qr) == 0
    assert qr.entity_ids.tolist() == []


def test_entities_are_conserved_across_pools():
    """Summing len over all pools equals the number of entities in the state."""
    state = setup({0: [C1], 1: [C1], 2: [C1, C2]}, _COMPONENTS)

    assert sum(len(pool) for pool in state.world.pools.values()) == 3


def test_remove_entity_leaves_empty_pool():
    """Removing the only entity empties its pool and leaves the id bookkeeping consistent (the `else` branch)."""
    state = setup({0: [C1]}, _COMPONENTS)

    state = act(state, [rme(0)])                              # last entity out -> pool is reclaimed
    assert state == {}
    assert state.world.pools == {}                                   # pool is gone
    assert len(state.world._pool_ids) == 0                          # no dangling pool_ids
    assert state.world._eid_to_pool_ix == {}                        # removed id is gone, not pointing at an empty slot


def test_empty_pool_is_reclaimed():
    """When the last entity leaves a pool, that archetype is fully dropped from the world (not leaked)."""
    state = setup({0: [C1]}, _COMPONENTS)
    pos_key = state.world._make_key((C1,))
    pos_pool = state.world.pools[pos_key]
    assert pos_key in state.world.pools                              # pool exists while it holds an entity

    state = act(state, [addc(0, C2)])                         # migration empties the C1-only pool
    assert state == {0: [C1, C2]}
    assert pos_key not in state.world.pools                         # reclaimed from the archetype registry
    assert pos_pool not in state.world.pool_to_components           # and released from the reverse map -> fully reclaimed


def test_remove_last_index_drops_only_that_entity():
    """Removing the last row (no swap happens) must drop that id, not resurrect it at a now-dead slot."""
    state = setup({0: [C1], 1: [C1]}, _COMPONENTS)
    state = act(state, [rme(1)])                              # 1 is the last row (no swap)
    assert state == {0: [C1]}

    pool = state.world.query(C1).pool_list[0]
    assert len(pool) == 1
    assert 1 not in state.world._eid_to_pool_ix                     # removed id gone, not pointing at a dead slot
    assert state.world._eid_to_pool_ix == {0: (pool, 0)}            # only the survivor remains, at its row
    assert state.world._pool_ids[pool] == [0]                       # reverse map agrees


def test_remove_middle_entity_repoints_swapped_id():
    """Removing a middle row swaps the tail into the gap: the tail's id re-points, the removed id vanishes."""
    state = setup({0: [C1], 1: [C1], 2: [C1]}, _COMPONENTS)
    state = act(state, [rme(1)])                              # c slides from slot 2 into slot 1
    assert state == {0: [C1], 2: [C1]}

    pool = state.world.query(C1).pool_list[0]
    assert len(pool) == 2
    assert 1 not in state.world._eid_to_pool_ix                     # removed id gone
    assert state.world._eid_to_pool_ix[0] == (pool, 0)              # 0 untouched
    assert state.world._eid_to_pool_ix[2] == (pool, 1)              # 2 re-pointed to the freed slot
    assert state.world._pool_ids[pool] == [0, 2]                    # reverse map consistent


def test_three_spawns_in_one_tick_get_sequential_ids():
    """Three add_entity calls before one commit allocate distinct, increasing ids."""
    state = setup({}, _COMPONENTS)

    state = act(state, [adde([C1]), adde([C1]), adde([C1])])

    assert state == {0: [C1], 1: [C1], 2: [C1]}
    assert list(state) == [0, 1, 2]


def test_id_resolves_after_sibling_removed():
    """Swap-remove moves the tail row; the moved entity's id must still resolve to it, not to its new neighbour.
    The whole add/remove/add/remove sequence commits in order on a single update(), same as immediate mode would."""
    state = setup({0: [C1], 1: [C1]}, _COMPONENTS)
    state = act(state, [rme(0),                              # swap-remove: 1's row slides into 0's old slot
                         adde([C1]),                           # World allocates 2
                         rme(1)])                              # must drop 1, not 2, despite the shuffle
    assert state == {2: [C1]}
    assert len(state.world.query(C1).pool_list[0]) == 1


def test_add_component_moves_entity_and_preserves_fields():
    """add_component widens the archetype: entity leaves the old pool, old field intact, new field set."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([1.0, 2.0], "float32")}})
    state = act(state, [addc(0, C2, c2=np.array([3.0, 4.0], "float32"))])
    assert state == {0: [C1, C2]}

    assert state.world._make_key((C1,)) not in state.world.pools               # old pool emptied -> reclaimed
    pool = state.world.pools[state.world._make_key((C1, C2))]
    assert len(pool) == 1
    np.testing.assert_array_equal(pool.c1[0], [1.0, 2.0])         # carried-over value intact
    np.testing.assert_array_equal(pool.c2[0], [3.0, 4.0])         # new value set


def test_add_component_keeps_entity_id():
    """The id is the caller's stable handle: migrating via add_component must NOT change it."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([1.0, 2.0], "float32")}})
    state = act(state, [addc(0, C2, c2=np.array([3.0, 4.0], "float32"))])
    assert state == {0: [C1, C2]}

    assert 0 in state.world._eid_to_pool_ix                              # original id still resolves
    pool, ix = state.world._eid_to_pool_ix[0]
    assert pool is state.world.pools[state.world._make_key((C1, C2))]
    np.testing.assert_array_equal(pool.c1[ix], [1.0, 2.0])
    np.testing.assert_array_equal(pool.c2[ix], [3.0, 4.0])


def test_remove_component_narrows_archetype():
    """remove_component drops a field and moves the entity to the smaller pool."""
    state = setup({0: [C1, C2]}, _COMPONENTS,
                  data={0: {"c1": np.array([1.0, 2.0], "float32"),
                            "c2": np.array([3.0, 4.0], "float32")}})
    state = act(state, [rmc(0, C2)])
    assert state == {0: [C1]}

    assert state.world._make_key((C1, C2)) not in state.world.pools
    pool = state.world.pools[state.world._make_key((C1,))]
    assert len(pool) == 1
    np.testing.assert_array_equal(pool.c1[0], [1.0, 2.0])         # kept field survives
    assert not hasattr(pool, "c2")                                 # dropped field is gone


def test_add_component_only_needs_new_fields():
    """Caller supplies just the new component's field; existing fields carry over without being re-passed."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([5.0, 6.0], "float32")}})
    state = act(state, [addc(0, C2, c2=np.array([7.0, 8.0], "float32"))])
    assert state == {0: [C1, C2]}

    pool = state.world.pools[state.world._make_key((C1, C2))]
    np.testing.assert_array_equal(pool.c1[0], [5.0, 6.0])
    np.testing.assert_array_equal(pool.c2[0], [7.0, 8.0])


def test_add_unknown_component_raises():
    """Adding a component the world never registered is rejected eagerly; the recovered state is unchanged."""
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [addc(0, C7)])
    assert isinstance(exc.value.cause, KeyError)
    assert exc.value.state == {0: [C1]}
    assert state == {0: [C1]}
    assert len(exc.value.state.world._command_buffer) == 0


def test_remove_entity_by_id():
    """remove_entity(eid) drops exactly that entity; the rest are conserved and still correct."""
    state = setup({0: [C1], 1: [C1]}, _COMPONENTS)
    state = act(state, [rme(0)])
    assert state == {1: [C1]}
    assert sum(len(pool) for pool in state.world.pools.values()) == 1            # counts conserved


def test_add_component_to_middle_sibling_keeps_all_ids():
    """Migrating the middle of three siblings triggers a swap in the old pool; every id must still resolve to its row."""
    state = setup({0: [C1], 1: [C1], 2: [C1]}, _COMPONENTS,
                  data={eid: {"c1": np.array([eid, eid], "float32")} for eid in range(3)})
    state = act(state, [addc(1, C2, c2=np.array([9.0, 9.0], "float32"))])
    assert state == {0: [C1], 1: [C1, C2], 2: [C1]}

    assert sum(len(pool) for pool in state.world.pools.values()) == 3
    for eid, expected in zip(range(3), ([0, 0], [1, 1], [2, 2])):
        pool, ix = state.world._eid_to_pool_ix[eid]
        np.testing.assert_array_equal(pool.c1[ix], expected)


def test_add_then_remove_component_round_trips():
    """add_component then remove_component returns the entity to its original archetype, same id, data intact."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([5.0, 6.0], "float32")}})
    state = act(state, [addc(0, C2, c2=np.array([7.0, 8.0], "float32")), rmc(0, C2)])
    assert state == {0: [C1]}

    pool, ix = state.world._eid_to_pool_ix[0]
    assert pool is state.world.pools[state.world._make_key((C1,))]
    assert state.world._make_key((C1, C2)) not in state.world.pools
    np.testing.assert_array_equal(pool.c1[ix], [5.0, 6.0])


def test_remove_component_lands_in_existing_pool_and_conserves_entities():
    """remove_component moving an entity into an already-populated smaller pool keeps both entities intact."""
    state = setup({0: [C1], 1: [C1, C2]}, _COMPONENTS,
                  data={0: {"c1": np.array([1.0, 1.0], "float32")},
                        1: {"c1": np.array([2.0, 2.0], "float32"),
                            "c2": np.array([3.0, 3.0], "float32")}})
    state = act(state, [rmc(1, C2)])
    assert state == {0: [C1], 1: [C1]}

    assert sum(len(pool) for pool in state.world.pools.values()) == 2
    for eid, expected in ((0, [1.0, 1.0]), (1, [2.0, 2.0])):
        pool, ix = state.world._eid_to_pool_ix[eid]
        assert pool is state.world.pools[state.world._make_key((C1,))]
        np.testing.assert_array_equal(pool.c1[ix], expected)


def test_add_component_reuses_existing_archetype_pool():
    """Reaching an archetype via add_component lands in the same pool a direct add_entity would."""
    state = setup({0: [C2, C1], 1: [C1]}, _COMPONENTS,
                  data={0: {"c1": np.array([1.0, 1.0], "float32"),
                            "c2": np.array([2.0, 2.0], "float32")},
                        1: {"c1": np.array([3.0, 3.0], "float32")}})
    state = act(state, [addc(1, C2, c2=np.array([4.0, 4.0], "float32"))])
    assert state == {0: [C1, C2], 1: [C1, C2]}

    assert state.world._eid_to_pool_ix[0][0] is state.world._eid_to_pool_ix[1][0]
    assert len(state.world.pools[state.world._make_key((C1, C2))]) == 2


def test_migrate_multi_field_component_preserves_all_fields():
    """A component with several fields migrates with every field intact and correctly named in the new pool."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([1.0, 2.0], "float32")}})
    state = act(state, [addc(0, C5, c5a=np.array([0.0, 0.0], "float32"),
                                   c5b=np.array([4.0, 4.0], "float32"))])
    assert state == {0: [C1, C5]}

    pool, ix = state.world._eid_to_pool_ix[0]
    np.testing.assert_array_equal(pool.c1[ix], [1.0, 2.0])
    np.testing.assert_array_equal(pool.c5a[ix], [0.0, 0.0])
    np.testing.assert_array_equal(pool.c5b[ix], [4.0, 4.0])


# --- eager id tracking -------------------------------------------------------------------------------------------
# A structural op on an entity that is NOT currently live fails at the CALL (clear AssertionError), not later inside
# update() as a cryptic KeyError. "Live" = committed or pending-spawn this tick, minus pending-despawn; World keeps
# this in live_entities. add_entity adds the new id; remove_entity removes it; add/remove_component just validate.


def test_spawn_then_add_component_same_tick():
    """An id minted by add_entity is a valid handle for a later action before the single commit."""
    state = setup({}, _COMPONENTS)
    state = act(state, [adde([C1], c1=np.array([1.0, 2.0], "float32")),
                        addc(0, C2, c2=np.array([3.0, 4.0], "float32"))])
    assert state == {0: [C1, C2]}

    pool, ix = state.world._eid_to_pool_ix[0]
    assert pool is state.world.pools[state.world._make_key((C1, C2))]
    np.testing.assert_array_equal(pool.c1[ix], [1.0, 2.0])
    np.testing.assert_array_equal(pool.c2[ix], [3.0, 4.0])


# --- a REJECTED add_entity must not leave bookkeeping behind ------------------------------------------------------
# add_entity bumps _last_id and inserts live_entities[id]=None BEFORE the append. That ordering is safe because
# add_entity validates FIRST -- a bad spawn raises before any bookkeeping happens. add_entity owns
# that validation, so the ordering is load-bearing rather than incidental: append does
# nothing for ADD_ENTITY but the liveness check, which add_entity has just satisfied by construction. Move the
# validation out of add_entity without also moving the two mutations after the append, and a rejected spawn burns
# an id and leaks a dangling live_entities entry -- `world.get_entity(that_id)` hands back an entity that will
# never exist. These two are that tripwire.

def test_rejected_add_entity_burns_no_id():
    """A refused spawn must not consume an entity id: the next successful add_entity gets the id it would have."""
    state = setup({}, _COMPONENTS)

    with pytest.raises(ActError) as exc:
        act(state, [adde([C4], c4=np.array([1.0, 2.0, 3.0], "float32"))])   # (3,) into a (2,) field

    assert exc.value.state == {}
    state = act(exc.value.state, [adde([C1])])
    assert state == {0: [C1]}              # id 0, not 1


def test_rejected_add_entity_leaves_no_live_entity_and_nothing_staged():
    """A refused spawn must leave live_entities and the buffer exactly as they were -- no dangling handle."""
    state = setup({}, _COMPONENTS)

    with pytest.raises(ActError) as exc:
        act(state, [adde([C4], c4=np.array([1.0, 2.0, 3.0], "float32"))])

    assert exc.value.state == {}            # no id registered -> get_entity(0) would raise
    assert len(exc.value.state.world._command_buffer) == 0
    assert len(exc.value.state.world.pools) == 0                   # and nothing materialized


# --- remove_entity is idempotent WITHIN A TICK ---------------------------------------------------------------------
# A kill is decided by several systems in one tick (damage, TTL, out-of-bounds), so an already-dead id is a NO-OP
# REQUEST, not a programming error -- raising would make every app write the same kill-set helper to avoid it.
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
    state = setup({0: [C1]}, _COMPONENTS)
    state = act(state, [rme(0), rme(0)])                       # a doubled command would KeyError at commit

    assert state == {}
    assert 0 not in state.world._eid_to_pool_ix
    assert len(state.world.pools) == 0


def test_remove_entity_after_the_despawn_committed_raises():
    """The boundary. One tick later the same call is no longer a race -- the id is a stale reference, and the
    caller has to hear about it. This is the half that separates "two systems agreed" from "you kept a dead id"."""
    state = setup({0: [C1]}, _COMPONENTS)
    state = act(state, [rme(0)])
    assert state == {}

    with pytest.raises(ActError) as exc:
        act(state, [rme(0)])                                  # despawn committed; the tick is over
    assert exc.value.state == {}
    assert state == {}
    assert len(exc.value.state.world._command_buffer) == 0    # the refusal staged nothing


def test_removed_this_tick_is_dropped_by_update():
    """The mechanism behind the boundary, pinned directly: the set lives and dies with the buffer. If it ever
    outlived an update(), every stale remove would silently become a no-op -- and nothing else would notice."""
    world = World(components=[C1])
    world.add_entity([C1])
    world.update()

    world.remove_entity(0)
    assert world._command_buffer.removed_this_tick == {0}
    world.update()
    assert world._command_buffer.removed_this_tick == set()


def test_removed_this_tick_agrees_with_the_commands_it_summarizes():
    """It is derived state, maintained by hand in two places (append + clear). Pin it against its own source, so
    a REMOVE_ENTITY that stops updating the set is caught here rather than as a mystery no-op."""
    world = World(components=[C1])
    for _ in range(3):
        world.add_entity([C1])
    world.update()

    world.remove_entity(0)
    world.remove_entity(2)
    staged = {cmd.entity_id for cmd in world._command_buffer if cmd.command_type == CommandType.REMOVE_ENTITY}
    assert world._command_buffer.removed_this_tick == staged == {0, 2}


@pytest.mark.parametrize("bad_id", [999, -1, -5])
def test_remove_entity_of_an_id_the_world_never_handed_out_raises(bad_id):
    """Never-spawned and long-dead are the SAME answer now: not live, no kill staged this tick -> raise. Negative
    ids are covered by the same rule rather than by a separate bounds check, which is why the rule is one line."""
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [rme(0), rme(bad_id)])                     # a real kill then an id never handed out

    assert exc.value.state == {}              # the accepted kill materialized
    assert state == {0: [C1]}                 # the input state is untouched
    assert len(exc.value.state.world._command_buffer) == 0


def test_remove_entity_accepts_the_numpy_ids_a_query_hands_out():
    """qr.entity_ids is int64. The no-op is a set lookup, so it must not care which integer type it is given --
    otherwise `kill_all(world, qr.entity_ids[mask])` raises on the duplicate kill it is supposed to absorb."""
    state = setup({0: [C1]}, _COMPONENTS)
    numpy_id = state.world.query(C1).entity_ids[0]
    state = act(state, [rme(numpy_id), rme(0)])                # np.int64 then python int: same entity
    assert state == {}
    assert len(state.world.pools) == 0


def test_remove_twice_after_a_same_tick_spawn_is_a_noop():
    """A spawn followed by two despawns in one action list: the first is legal, the repeat is the no-op."""
    state = setup({}, _COMPONENTS)
    state = act(state, [adde([C1]), rme(0), rme(0)])
    assert state == {}
    assert len(state.world.pools) == 0 and len(state.world.live_entities) == 0


def test_add_component_after_remove_entity_fails():
    """A system removes an entity; a later system tries to widen it the same tick -> reject eagerly."""
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [rme(0), addc(0, C2)])
    assert exc.value.state == {}              # the accepted despawn is the recovered state
    assert state == {0: [C1]}                 # the input state is untouched
    assert len(exc.value.state.world._command_buffer) == 0


def test_remove_component_after_remove_entity_fails():
    """A system removes an entity; a later system tries to narrow it the same tick -> reject eagerly."""
    state = setup({0: [C1, C2]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [rme(0), rmc(0, C2)])
    assert exc.value.state == {}              # the accepted despawn is the recovered state
    assert state == {0: [C1, C2]}             # the input state is untouched
    assert len(exc.value.state.world._command_buffer) == 0


def test_remove_unknown_entity_id_fails():
    """An id the world never handed out is not live -> remove_entity must reject it at the call, not at commit."""
    state = setup({}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [rme(123)])
    assert exc.value.state == {}


def test_spawn_into_archetype_reclaimed_by_earlier_despawn_same_tick():
    """Despawn the last entity of an archetype, then spawn a new one of the SAME archetype, same tick.
    The newcomer must land in a live, queryable pool, not an orphaned one.

    Reclamation is a single sweep at the END of update(), not mid-commit inside `_pop_from_pool`, so the emptied
    pool is still there when the spawn arrives and gets REUSED -- see the identity assertion below. One less Pool
    allocation than deleting it and building a brand-new one behind it."""
    world = World(components=[C1])
    old = world.add_entity([C1])
    world.update()
    pool_before = world.pools[world._make_key((C1,))]

    world.remove_entity(old)                                   # queued first: empties the C1 pool
    world.add_entity([C1])                                     # id 1
    world.update()

    assert old not in world._eid_to_pool_ix
    assert sum(len(pool) for pool in world.pools.values()) == 1   # exactly the newcomer, and it is visible
    assert world._make_key((C1,)) in world.pools               # its pool is live / registered (not orphaned)
    assert world._eid_to_pool_ix[1][0] is pool_before          # reused, not blinked: no realloc, no new Pool


def test_an_archetype_emptied_and_refilled_in_one_tick_is_never_torn_down():
    """The same property stated as the thing it fixes: a bullet archetype that momentarily hits zero mid-tick used
    to cost a full Pool teardown + rebuild (INITIAL_CAPACITY rows per field, per blink -- the space-shooter run
    measured 747 pool builds in a chaos session). With reclamation deferred to the end of update(), a pool that is
    refilled before the sweep runs is never torn down at all."""
    world = World(components=[C1])
    ids = [world.add_entity([C1]) for _ in range(3)]
    world.update()
    pool_before = world.pools[world._make_key((C1,))]

    for _ in range(5):                                         # drain to zero and refill, 5 ticks running
        for eid in ids:
            world.remove_entity(eid)
        ids = [world.add_entity([C1]) for _ in range(3)]
        world.update()
        assert world.pools[world._make_key((C1,))] is pool_before

    assert len(world.pools) == 1 and len(pool_before) == 3
    assert_pool_ids_invariants(world)


def test_an_archetype_left_empty_at_the_end_of_a_tick_is_reclaimed():
    """The other side of the sweep: if nothing refills it, the pool and BOTH its bookkeeping entries go away.
    A pool left in `pool_to_components` or `_pool_ids` would be a leak the query loop still walks every tick."""
    state = setup({0: [C1], 1: [C1, C2]}, _COMPONENTS)
    assert len(state.world.pools) == 2

    state = act(state, [rme(0)])
    assert state == {1: [C1, C2]}
    assert len(state.world.pools) == len(state.world.pool_to_components) == len(state.world._pool_ids) == 1
    assert state.world._make_key((C1,)) not in state.world.pools         # the emptied one is gone, not just empty
    assert_pool_ids_invariants(state.world)


# --- fully-eager staging -----------------------------------------------------------------------------------------
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
    """A queued add is visible to a later remove the same tick: addc then rmc commits back to C1."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([1.0, 2.0], "float32")}})
    state = act(state, [addc(0, C2, c2=np.array([3.0, 4.0], "float32")), rmc(0, C2)])
    assert state == {0: [C1]}
    np.testing.assert_array_equal(state.world.get_entity(0).c1, [1.0, 2.0])


def test_remove_then_add_same_component_same_tick_ok():
    """The mirror: rmc then addc commits back to {C1, C2} with the re-added value winning."""
    state = setup({0: [C1, C2]}, _COMPONENTS,
                  data={0: {"c1": np.array([1.0, 2.0], "float32"),
                            "c2": np.array([3.0, 4.0], "float32")}})
    state = act(state, [rmc(0, C2), addc(0, C2, c2=np.array([5.0, 6.0], "float32"))])
    assert state == {0: [C1, C2]}
    np.testing.assert_array_equal(state.world.get_entity(0).c2, [5.0, 6.0])


def test_add_remove_cycle_same_tick_ok():
    """A longer churn in one tick validates each op against the running projected set."""
    state = setup({0: [C1]}, _COMPONENTS)
    state = act(state, [addc(0, C2, c2=np.array([3.0, 4.0], "float32")),
                         rmc(0, C2),
                         addc(0, C2, c2=np.array([5.0, 6.0], "float32")),
                         rmc(0, C2)])
    assert state == {0: [C1]}


def test_double_add_raises_eagerly_at_second_call():
    """The second add of an already-staged component raises at the call; the first add is the recovered state."""
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [addc(0, C2, c2=np.array([3.0, 4.0], "float32")),
                    addc(0, C2, c2=np.array([5.0, 6.0], "float32"))])
    assert exc.value.state == {0: [C1, C2]}
    assert state == {0: [C1]}                  # the input state is untouched
    assert len(exc.value.state.world._command_buffer) == 0


def test_remove_after_add_then_remove_is_a_noop():
    """A repeat removal after its own tick-local add is a no-op: the newest buffered decision is the removal."""
    state = setup({0: [C1]}, _COMPONENTS)
    state = act(state, [addc(0, C2, c2=np.array([3.0, 4.0], "float32")), rmc(0, C2), rmc(0, C2)])
    assert state == {0: [C1]}


def test_remove_component_twice_is_a_noop():
    """A repeat removal in one tick stages one command and narrows the archetype once."""
    state = setup({0: [C1, C2]}, _COMPONENTS)
    state = act(state, [rmc(0, C2), rmc(0, C2)])
    assert state == {0: [C1]}


def test_remove_component_after_the_removal_committed_raises():
    """The no-op is tick-local: after commit, another remove is stale state, not a repeat request."""
    state = setup({0: [C1, C2]}, _COMPONENTS)
    state = act(state, [rmc(0, C2)])
    assert state == {0: [C1]}

    with pytest.raises(ActError) as exc:
        act(state, [rmc(0, C2)])
    assert exc.value.state == {0: [C1]}
    assert state == {0: [C1]}
    assert len(exc.value.state.world._command_buffer) == 0


# -- add/remove_component state transitions ------------------------------------------------------------------------
# state = setup({eid: [components]})
# state = act(state, [addc(eid, C), rmc(eid, C), ...])
# assert state == {eid: [components]}

def test_add_component_transitions_state():
    state = setup({0: [C1]}, _COMPONENTS)
    state = act(state, [addc(0, C2)])
    assert state == {0: [C1, C2]}


def test_remove_component_transitions_state():
    state = setup({0: [C1, C2]}, _COMPONENTS)
    state = act(state, [rmc(0, C2)])
    assert state == {0: [C1]}


def test_state_is_mapping_compatible():
    """The formula compares to a plain dict and supports normal mapping reads."""
    state = setup({0: [C1], 1: [C1, C2]}, _COMPONENTS)
    assert state == {0: [C1], 1: [C1, C2]}
    assert {0: [C1], 1: [C1, C2]} == state
    assert state == {1: [C2, C1], 0: [C1]}                    # order-free
    assert state == {0: {C1}, 1: {C1, C2}}                    # any component iterable
    assert state[1] == frozenset({C1, C2})
    assert list(state) == [0, 1]
    assert dict(state) == {0: frozenset({C1}), 1: frozenset({C1, C2})}


def test_state_equality_includes_the_next_id():
    """The live mapping is a projection: removal history changes the next spawn id, so equal mappings are not
    equal states. Contrast with the plain-dict comparison above, which deliberately ignores allocation history."""
    a = act(setup({0: [C1], 1: [C1], 2: [C1]}, _COMPONENTS), [rme(1)])
    b = act(setup({0: [C1], 1: [C1], 2: [C1], 3: [C1]}, _COMPONENTS), [rme(1), rme(3)])

    assert a.entities == b.entities == {0: frozenset({C1}), 2: frozenset({C1})}
    assert a.next_id == 3
    assert b.next_id == 4
    assert a != b

    assert a == {0: [C1], 2: [C1]}                            # projection equality still works
    assert b == {0: [C1], 2: [C1]}
    assert act(a, [adde([C1])]) == {0: [C1], 2: [C1], 3: [C1]}
    assert act(b, [adde([C1])]) == {0: [C1], 2: [C1], 4: [C1]}


@pytest.mark.parametrize("entities", [{1: [C1]}, {0: [C1], 2: [C1]}, {5: [C1]}])
def test_setup_rejects_eids_that_are_not_contiguous_from_zero(entities):
    """setup uses the World's monotonic allocator, so it can only spawn 0..N-1; a hole would silently shift every id."""
    with pytest.raises(AssertionError, match="contiguous from 0"):
        setup(entities, _COMPONENTS)


def test_successful_act_returns_a_new_state_and_leaves_the_input_untouched():
    """Copy-on-act: the returned state is checked; the input still describes the pre-transition world."""
    state = setup({0: [C1]}, _COMPONENTS)
    original = dict(state)

    next_state = act(state, [addc(0, C2)])

    assert next_state is not state
    assert next_state.world is not state.world
    assert next_state == {0: [C1, C2]}
    assert state == {0: [C1]}
    assert dict(state) == original
    assert len(state.world._command_buffer) == 0


def test_act_materializes_the_action_sequence_once():
    """Validation and application see the same sequence: a generator is consumed once, so every action applies."""
    state = setup({0: [C1]}, _COMPONENTS)
    actions = (action for action in [addc(0, C2), rmc(0, C1)])

    assert act(state, actions) == {0: [C2]}
    assert state == {0: [C1]}


def test_act_rejects_a_generic_callable():
    """act accepts only the four formal verbs, not arbitrary World mutators."""
    state = setup({0: [C1]}, _COMPONENTS)

    def rogue(_world):
        raise AssertionError("the generic callable must never run")

    with pytest.raises(AssertionError, match="four formal verbs"):
        act(state, [rogue])
    assert state == {0: [C1]}


def test_act_rejects_a_duck_typed_action():
    """An object with an apply() method is not a verb; the alphabet is four exact classes, not a protocol."""
    state = setup({0: [C1]}, _COMPONENTS)

    class Rogue:
        def apply(self, world):
            raise AssertionError("the duck-typed action must never run")

    with pytest.raises(AssertionError, match="four formal verbs"):
        act(state, [Rogue()])
    assert state == {0: [C1]}


def test_act_branches_do_not_share_state():
    """Two transitions from one state are independent; neither sees the other's rows."""
    root = setup({0: [C1, C2]}, _COMPONENTS)
    added = act(root, [addc(0, C3)])
    removed = act(root, [rmc(0, C2)])

    assert root == {0: [C1, C2]}
    assert added == {0: [C1, C2, C3]}
    assert removed == {0: [C1]}


def test_act_copies_pool_data():
    """The copy is row data too: mutating the returned state cannot reach the input state."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([1.0, 2.0], "float32")}})
    next_state = act(state, [addc(0, C2)])

    next_state.world.get_entity(0).c1[0] = 9.0

    np.testing.assert_array_equal(state.world.get_entity(0).c1, [1.0, 2.0])


def test_failed_act_exposes_the_accepted_prefix():
    """A failure mid-sequence returns the prefix state; the input state stays untouched."""
    state = setup({0: [C1], 1: [C1]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [addc(0, C2), addc(0, C3), addc(0, C2)])

    assert exc.value.state == {0: [C1, C2, C3], 1: [C1]}
    assert state == {0: [C1], 1: [C1]}
    assert len(exc.value.state.world._command_buffer) == 0


def test_failed_act_exposes_unrelated_prefix_work():
    """The recovered state spans entities: a completed despawn is kept when a later mutation is rejected."""
    state = setup({0: [C1], 1: [C1]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [rme(1), rmc(0, C3)])

    assert exc.value.state == {0: [C1]}
    assert state == {0: [C1], 1: [C1]}
    assert len(exc.value.state.world._command_buffer) == 0


@pytest.mark.xfail(strict=True, reason="a removal that empties the projected set is accepted")
def test_remove_only_component_is_rejected():
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError, match="last component") as exc:
        act(state, [rmc(0, C1)])
    assert exc.value.state == {0: [C1]}
    assert state == {0: [C1]}


@pytest.mark.xfail(strict=True, reason="a removal that empties the projected set is accepted")
def test_remove_after_a_cancelled_staged_add_is_rejected():
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError, match="last component") as exc:
        act(state, [addc(0, C2), rmc(0, C2), rmc(0, C1)])
    assert exc.value.state == {0: [C1]}        # the add/remove pair cancels itself
    assert state == {0: [C1]}
    assert len(exc.value.state.world._command_buffer) == 0


@pytest.mark.xfail(strict=True, reason="two removals can empty the projected set")
def test_second_remove_that_empties_the_entity_is_rejected():
    state = setup({0: [C1, C3]}, _COMPONENTS)
    with pytest.raises(ActError, match="last component") as exc:
        act(state, [rmc(0, C1), rmc(0, C3)])
    assert exc.value.state == {0: [C3]}        # the first removal is recovered
    assert state == {0: [C1, C3]}
    assert len(exc.value.state.world._command_buffer) == 0


@pytest.mark.xfail(strict=True, reason="a staged swap can still be emptied")
def test_remove_of_the_only_staged_survivor_is_rejected():
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError, match="last component") as exc:
        act(state, [addc(0, C2), rmc(0, C1), rmc(0, C2)])
    assert exc.value.state == {0: [C2]}        # add C2 + remove C1 is recovered
    assert state == {0: [C1]}
    assert len(exc.value.state.world._command_buffer) == 0


@pytest.mark.xfail(strict=True, reason="the projected set of a same-tick spawn is not checked")
def test_remove_only_component_of_a_same_tick_spawn_is_rejected():
    state = setup({}, _COMPONENTS)
    with pytest.raises(ActError, match="last component") as exc:
        act(state, [adde([C1]), rmc(0, C1)])
    assert exc.value.state == {0: [C1]}        # the accepted spawn is recovered
    assert state == {}


@pytest.mark.xfail(strict=True, reason="a removal before its survivor is staged is not refused")
def test_remove_before_its_survivor_is_staged_is_rejected():
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError, match="last component") as exc:
        act(state, [rmc(0, C1)])
    assert exc.value.state == {0: [C1]}
    state = act(exc.value.state, [addc(0, C2), rmc(0, C1)])
    assert state == {0: [C2]}


def test_remove_leaves_a_staged_add_as_survivor():
    state = setup({0: [C1]}, _COMPONENTS)
    state = act(state, [addc(0, C2), rmc(0, C1)])
    assert state == {0: [C2]}


def test_remove_leaves_a_committed_survivor():
    state = setup({0: [C1, C2]}, _COMPONENTS)
    state = act(state, [rmc(0, C1)])
    assert state == {0: [C2]}


def test_remove_leaves_a_tag_as_survivor():
    state = setup({0: [C1, C3]}, _COMPONENTS)
    state = act(state, [rmc(0, C1)])
    assert state == {0: [C3]}


def test_remove_leaves_a_same_tick_spawn_survivor():
    state = setup({}, _COMPONENTS)
    state = act(state, [adde([C1, C3]), rmc(0, C1)])
    assert state == {0: [C3]}


def test_remove_leaves_a_staged_data_component():
    state = setup({0: [C1]}, _COMPONENTS)
    state = act(state, [addc(0, C2, c2=np.array([3.0, 4.0], "float32")), rmc(0, C1)])
    assert state == {0: [C2]}
    np.testing.assert_array_equal(state.world.get_entity(0).c2, [3.0, 4.0])


def test_add_component_wrong_shape_raises_eagerly():
    """A wrong-shaped value is rejected at the call; the recovered state is unchanged."""
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [addc(0, C2, c2=np.array([1.0, 2.0, 3.0], "float32"))])
    assert exc.value.state == {0: [C1]}
    assert state == {0: [C1]}
    assert len(exc.value.state.world._command_buffer) == 0


def test_add_component_wrong_dtype_raises_eagerly():
    """A float64 value where the field declares float32 is caught at the call; the recovered state is unchanged."""
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [addc(0, C2, c2=np.array([3.0, 4.0], "float64"))])
    assert exc.value.state == {0: [C1]}
    assert state == {0: [C1]}
    assert len(exc.value.state.world._command_buffer) == 0


def test_add_component_missing_required_field_raises_eagerly():
    """A required field omitted at add is rejected at the call; the recovered state is unchanged."""
    state = setup({0: [C1]}, _COMPONENTS)
    with pytest.raises(ActError) as exc:
        act(state, [addc(0, C4)])
    assert exc.value.state == {0: [C1]}
    assert state == {0: [C1]}
    assert len(exc.value.state.world._command_buffer) == 0


# --- object-dtype components -------------------------------------------------------------------------------------
# A component field may declare dtype "object": its storage holds arbitrary Python objects (dicts, callbacks, handles)
# by reference rather than numeric data. Everything else (pools, migrations, swap-remove) must treat it like any field.


def test_world_accepts_object_dtype_component():
    """Construction validation allows dtype 'object'; the world records it for the field."""
    world = World(components=[C12])
    assert world.component_to_dtypes[C12] == ["object"]


def test_world_rejects_str_dtype_component():
    """`str` is not an allowed dtype. numpy strings are fixed-width (<U n) and a pre-allocated pool
    can't size the width ahead of the data -- writes silently truncate. A field declaring dtype 'str' must
    fail loud at World construction (where the bad field lives), not corrupt data later. Python strings
    live in dtype='object' (see test_object_field_holds_python_string_and_compares_by_equality)."""
    class C16(Component):
        name: np.ndarray = field(metadata={"shape": (1,), "dtype": "str", "default": None})

    with pytest.raises(TypeError, match="str not a string or not in"):
        World(components=[C16])


def test_world_rejects_non_dataclass_component():
    """A component that isn't a dataclass has no fields to lay out -- rejected with the TypeError that names it."""
    class NotADataclass:                                    # not a Component subclass -> no @dataclass applied
        pass

    with pytest.raises(TypeError, match="NotADataclass"):
        World(components=[NotADataclass])


@pytest.mark.parametrize("bad", ["serializable", ("serializable",), {"serializable"}])
def test_world_rejects_non_list_extra_metadata(bad):
    """extra_metadata is a ctor arg -- user input -- so a wrong type is a real `raise`, not an assert.
    Measured with the assert stripped by -O: a bare str is splatted into single characters and construction dies
    later with `expected_meta = {'i','e','s','z',...}` -- the right rejection for an unreadable reason."""
    with pytest.raises(TypeError):
        World(components=[C8], extra_metadata=bad)


def _two_components_sharing_a_field_name() -> tuple[type[Component], type[Component]]:
    """Two components declaring the SAME field name -- illegal: rejected at World construction."""
    class C17(Component):
        payload: np.ndarray = f32((1,))

    class C18(Component):
        payload: np.ndarray = f32((1,))  # same name!

    return C17, C18


def test_world_rejects_duplicate_field_name_across_components():
    """Two components may not declare the SAME field name, rejected at construction -- earliest and loudest.

    A pool merges fields by name and query() sums field names across components, so a clash would silently
    alias two components' data. It is also what makes `entity.set_data(**fields)` well-defined: the
    field name alone must identify its component, via world.field_to_component.

    With the construction guard, update()'s field-dict merge ("Duplicate keys") is unreachable through the
    public API, so the merge check in _do_add_component is pure defence-in-depth and has no test of its own."""
    C17, C18 = _two_components_sharing_a_field_name()

    with pytest.raises(ValueError, match="payload"):
        World(components=[C17, C18])


# --- reserved field names: a field named like an internal attr must be rejected at construction --------------------
# A field name that collides with QueryResult's own attrs can never be caught downstream -- QueryResult.__setattr__
# checks the allow-list BEFORE the data-field branch, so `qr.entity_ids = v` replaces the id array instead of
# scattering, and len()/repr()/id lookups then read the written value. World._check_components (world.py:264) is the
# one gate: ENTITY_RESERVED_NAMES | POOL_RESERVED_NAMES | QUERY_RESULT_RESERVED_NAMES, each set being that class's
# instance attrs plus its class dict. Pool's names are in the union, and every check is a real `raise`, so
# `python -O` does not erase them (pinned below).

def _component_with_field(name: str) -> type[Component]:
    """A one-field Component whose field is named `name` (a dataclass field name can't be dynamic in a class body)."""
    return type("C24", (Component,), {
        "__annotations__": {name: np.ndarray},
        name: f32((1,)),
    })


# A field name colliding with a class's own attribute is rejected with ValueError (world.py:277) -- the guard is a
# real raise, so it survives `python -O`.
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
    Pool.__getattr__ only runs after normal lookup fails. Past World they would blow up much later, when the
    first entity of that archetype commits and Pool rejects at construction -- the late, quiet failure this
    construction-time check exists to prevent. One union, one gate."""
    with pytest.raises(_REJECTED, match=name):
        World(components=[_component_with_field(name)])


def test_world_reserved_name_guard_survives_python_dash_o():
    """The guard must survive `python -O`, which strips asserts. A component definition is user input, so it is
    `raise` territory, not `assert` (asserts are for our own bugs). As an assert, under -O World() would accept a
    field named `entity_ids` and the first write through a QueryResult would silently replace the id array. Run in
    a subprocess because -O is decided at interpreter start."""
    script = ("import numpy as np\n"
              "from dataclasses import field\n"
              "from microecs import World, Component\n"
              "C = type('C24', (Component,), {'__annotations__': {'entity_ids': np.ndarray},\n"
              "     'entity_ids': field(metadata={'shape': (1,), 'dtype': 'float32', 'default': None})})\n"
              "World(components=[C])\n")
    env = {**os.environ, "PYTHONPATH": str(Path(microecs.__file__).parent.parent)}

    res = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True, env=env, check=False)

    assert res.returncode != 0, f"-O accepted a field named 'entity_ids'\nstdout: {res.stdout}\nstderr: {res.stderr}"


# --- World's own kwarg names are a fourth collidable surface -------------------------------------------------------
# The union above covers the three CLASSES' attrs+methods. It misses the keyword-PARAMETER names of the World
# methods that take field data as **kwargs, so `**kwargs` shadows the parameter. These four pass World() and only
# die at the first add_entity, with an internal "got multiple values for argument" TypeError that names a parameter
# instead of the offending field -- the late, cryptic failure this gate exists to prevent. `data` and `entity_id`
# are already rejected, but only incidentally (they happen to be Pool / Entity attrs).

_WORLD_KWARG_NAMES = ["check_extra", "component", "components", "strict"]
@pytest.mark.parametrize("name", _WORLD_KWARG_NAMES)
def test_world_rejects_field_name_colliding_with_its_own_kwargs(name):
    """A field named like one of World's own **kwargs-taking parameters must be refused at construction."""
    with pytest.raises(_REJECTED, match=name):
        World(components=[_component_with_field(name)])

def test_object_field_holds_python_string_and_compares_by_equality():
    """The sanctioned replacement for the removed str dtype: store a python string in a dtype='object'
    field. The full string is kept (no <U width cap, no truncation) and string `==` on the pool column
    works -- element-wise and vectorised."""
    class C19(Component):  # e.g. a 'component_kind' tag stored as a string-in-object
        kind: np.ndarray = obj((1,))

    world = World(components=[C19])
    a = world.add_entity(components=(C19,), kind=np.array(["enemy"], dtype=object))             # idx 0
    b = world.add_entity(components=(C19,), kind=np.array(["player_one_long_name"], dtype=object))  # idx 1
    world.update()

    pool, ia = world._eid_to_pool_ix[a]
    _, ib = world._eid_to_pool_ix[b]
    assert pool.kind.dtype == object
    assert pool.kind[ia, 0] == "enemy"                                  # exact string, not truncated to "e"
    assert pool.kind[ib, 0] == "player_one_long_name"                   # full length kept, no <U cap
    np.testing.assert_array_equal(pool.kind[:, 0] == "enemy", [True, False])  # vectorised string equality


def test_object_component_stores_and_reads_back_the_same_object():
    """The exact Python object passed in is readable back from the pool -- by identity, not just by value."""
    world = World(components=[C12])

    payload = {"name": "drone-7", "tags": ["a", "b"]}
    eid = world.add_entity(components=(C12,), label=np.array([payload], dtype=object))
    world.update()

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool.label.dtype == object
    assert pool.label[ix, 0] is payload                            # same reference, not a copy


def test_object_component_survives_migration():
    """add_component carries an object field over to the wider pool with its reference intact."""
    obj = object()
    world = World([C1, C6])
    eid = world.add_entity([C6], c6=np.array([obj], dtype=object))
    world.update()

    world.get_entity(eid).add_component(C1, c1=np.array([1.0, 2.0], "float32"))
    world.update()

    pool, ix = world._eid_to_pool_ix[eid]
    assert pool is world.pools[world._make_key((C1, C6))]
    assert pool.c6[ix, 0] is obj                                   # object preserved across the archetype move
    np.testing.assert_array_equal(pool.c1[ix], [1.0, 2.0])


def test_distinct_objects_per_entity_survive_swap_remove():
    """Each entity keeps its own object; removing one swaps the tail in, and the survivor's object is unchanged."""
    world = World(components=[C12])

    first, second = {"id": 1}, {"id": 2}
    a = world.add_entity(components=(C12,), label=np.array([first], dtype=object))   # idx 0
    b = world.add_entity(components=(C12,), label=np.array([second], dtype=object))  # idx 1 (tail)
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
    world = World(components=[C8])
    eid = world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    entity = world.get_entity(eid)
    np.testing.assert_array_equal(entity.position, [1.0, 2.0])
    assert set(entity.get_components()) == {C8}


def test_get_entity_returns_all_fields_of_a_multi_component_entity():
    """Every field of a multi-component archetype comes back, keyed by field name."""
    world = World(components=[C8, C9])
    eid = world.add_entity(components=(C8, C9),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    entity = world.get_entity(eid)
    np.testing.assert_array_equal(entity.position, [1.0, 2.0])
    np.testing.assert_array_equal(entity.velocity, [3.0, 4.0])
    assert set(entity.get_components()) == {C8, C9}


def test_get_entity_is_read_only_and_id_still_resolves():
    """get_entity must NOT consume the entity: the id keeps resolving, the call is repeatable, removal still works."""
    world = World(components=[C8])
    eid = world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
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
    world = World(components=[C8])
    a = world.add_entity(components=(C8,), position=np.array([0.0, 0.0], "float32"))  # idx 0
    b = world.add_entity(components=(C8,), position=np.array([1.0, 1.0], "float32"))  # idx 1
    c = world.add_entity(components=(C8,), position=np.array([2.0, 2.0], "float32"))  # idx 2 (tail)
    world.update()

    world.remove_entity(a)                                         # c swaps into slot 0; b stays at slot 1
    world.update()

    entity_b = world.get_entity(b)
    entity_c = world.get_entity(c)
    np.testing.assert_array_equal(entity_b.position, [1.0, 1.0])  # b unmoved
    np.testing.assert_array_equal(entity_c.position, [2.0, 2.0])  # c followed its id into the freed slot


def test_get_entity_unknown_id_raises():
    """An id the world never handed out has no data -> raise, not return an empty/garbage result."""
    world = World(components=[C8])
    with pytest.raises(ValueError):                               # a clear ValueError, like the other ops
        world.get_entity(123)


# --- lazy Entity allocation + cache -------------------------------------------------------------------------------
# live_entities does two jobs: a liveness registry (id present == live) and an Entity cache. The Entity object is
# NOT built at add_entity (a pure-vectorised sim that never calls get_entity pays zero Entity objects); it is built
# on the FIRST get_entity and cached (same object thereafter), and evicted at remove_entity. These pin that.


def test_add_entity_allocates_no_entity_object():
    """add_entity registers the id as live but stores no Entity -- the slot is None until someone asks for it."""
    world = World(components=[C8])
    eid = world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert eid in world.live_entities                              # the id is live (registry job)
    assert world.live_entities[eid] is None                       # but no Entity was built (lazy: pay only on demand)


def test_get_entity_builds_once_then_caches_same_object():
    """First get_entity builds the Entity and stores it; later calls hand back the SAME object (stable identity)."""
    world = World(components=[C8])
    eid = world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert world.live_entities[eid] is None                       # nothing built yet
    first = world.get_entity(eid)
    assert world.live_entities[eid] is first                      # cached on first request
    assert world.get_entity(eid) is first                         # repeat call -> same object, not a rebuild


def test_remove_entity_evicts_the_cached_entity():
    """remove_entity drops the id from live_entities, taking the cached Entity with it -- no stale view lingers."""
    world = World(components=[C8])
    eid = world.add_entity(components=(C8,), position=np.array([1.0, 2.0], "float32"))
    world.update()
    world.get_entity(eid)                                         # populate the cache

    world.remove_entity(eid)                                      # eager eviction from the registry+cache
    assert eid not in world.live_entities
    with pytest.raises(ValueError):                             # a removed id no longer resolves
        world.get_entity(eid)


def test_get_entity_resolves_each_id_to_its_own_object():
    """Two ids never share one cached Entity: each builds its own, keyed by id."""
    world = World(components=[C8])
    a = world.add_entity(components=(C8,), position=np.array([0.0, 0.0], "float32"))
    b = world.add_entity(components=(C8,), position=np.array([1.0, 1.0], "float32"))
    world.update()

    ea, eb = world.get_entity(a), world.get_entity(b)
    assert ea is not eb and ea.entity_id == a and eb.entity_id == b


# --- _pool_ids randomized churn: the reverse id-map must mirror the pools through every popswap ------------------

_CHURN_COMPONENTS = {C8: ("position", (2,)), C9: ("velocity", (2,)), C10: ("radius", (1,))}


def _rand_fields(comp, rng: random.Random) -> dict:
    """Random field-data kwargs for one churn component (one (shape,) float32 field, name unique per component)."""
    name, shape = _CHURN_COMPONENTS[comp]
    return {name: np.array([rng.random() for _ in range(shape[0])], "float32")}


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

        assert_pool_ids_invariants(world)
        for eid, fields in shadow.items():
            entity = world.get_entity(eid)
            for name, value in fields.items():
                np.testing.assert_array_equal(getattr(entity, name), value)  # each id keeps its OWN data thru swaps

    assert len(world.live_entities) > 0                            # sanity: the churn left a populated world


# --- the two despawn paths must not drift -------------------------------------------------------------------------
# `_remove_from_pool` (no copy) sits beside `_pop_from_pool`. The duplication is small on purpose -- empty-pool
# reclamation is a single sweep at the end of update(), out of both -- but they still carry the same pop-swap and
# id-repointing, and differ only in whether the row is copied out first. update() sends REMOVE_ENTITY to the no-copy
# one and component migration to the copying one, so a fix applied to one and not the other silently desyncs half
# the churn paths. Nothing else compares them directly, so this does.


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
        world = World(components=[C8, C9])
        ids = [world.add_entity((C8, C9),
                                position=np.array([float(i), float(i)], "float32"),
                                velocity=np.array([float(-i), float(i)], "float32")) for i in range(5)]
        world.add_entity((C8,), position=np.array([99.0, 99.0], "float32"))   # a 2nd pool, untouched
        world.update()
        for v in victims:
            getattr(world, path_name)(ids[v])
            del world.live_entities[ids[v]]        # what World.remove_entity does around the pool call
        return _world_shape(world)

    assert _run("_remove_from_pool") == _run("_pop_from_pool")


def test_the_last_despawn_in_a_world_leaves_no_bookkeeping_behind():
    """Drain the world entirely through the no-copy path: every map must end empty. Neither despawn path reclaims
    pools -- the end-of-update sweep does -- so this pins that the sweep covers the no-copy path too, and
    that nothing is left dangling in `pool_to_components` / `_pool_ids` / `_eid_to_pool_ix`."""
    state = setup({0: [C1]}, _COMPONENTS)
    assert len(state.world.pools) == 1

    state = act(state, [rme(0)])
    assert state == {}
    assert state.world.pools == {} and state.world.pool_to_components == {} and state.world._pool_ids == {}
    assert state.world._eid_to_pool_ix == {} and state.world.live_entities == {}


# --- QueryResult.entity_ids: a flat (N,) integer array, pool-by-pool aligned with the qr.field parts -----------

def test_query_result_entity_ids_is_flat_and_aligned_across_pools():
    """qr.entity_ids is a flat (N,) integer array covering every matched entity across archetypes, in the same
    pool-by-pool order as qr.position -- so zip(qr.entity_ids, qr.position) pairs each id with its own row."""
    world = World(components=[C8, C9])
    a = world.add_entity(components=(C8,), position=np.array([0.0, 0.0], "float32"))
    b = world.add_entity(components=(C8, C9),
                         position=np.array([1.0, 1.0], "float32"), velocity=np.array([9.0, 9.0], "float32"))
    c = world.add_entity(components=(C8, C9),
                         position=np.array([2.0, 2.0], "float32"), velocity=np.array([8.0, 8.0], "float32"))
    world.update()

    qr = world.query(C8)                           # matches both archetypes -> two pools

    assert isinstance(qr.entity_ids, np.ndarray)
    assert np.issubdtype(qr.entity_ids.dtype, np.integer)
    assert qr.entity_ids.shape == (len(qr),)                       # flat, one entry per entity
    assert set(qr.entity_ids.tolist()) == {a, b, c}                # exactly the matched ids
    for eid, pos in zip(qr.entity_ids, qr.position):               # id <-> row alignment, across pools
        np.testing.assert_array_equal(world.get_entity(int(eid)).position, pos)


def test_query_result_entity_ids_supports_flat_array_ops():
    """entity_ids is a real ndarray, not a Field: entity-axis indexing and fancy ops that Field rejects --
    qr.entity_ids[i], slicing, np.isin -- all work, because ids are materialized by World, not a per-pool view."""
    world = World(components=[C8])
    ids = [world.add_entity(components=(C8,), position=np.array([i, i], "float32")) for i in range(4)]
    world.update()

    qr = world.query(C8)

    assert int(qr.entity_ids[0]) in ids                            # entity-axis index -> allowed (unlike Field)
    assert qr.entity_ids[1:3].shape == (2,)                        # slicing the entity axis -> allowed
    assert np.isin(qr.entity_ids, ids[:2]).sum() == 2              # fancy / set ops -> allowed


def test_query_result_entity_ids_empty_query_is_empty_flat_array():
    """A query that matches no pool yields an empty flat (0,) id array, mirroring an empty field -- not a crash."""
    world = World(components=[C8, C9])
    world.add_entity(components=(C8,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    qr = world.query(C9)                           # nothing has velocity

    assert len(qr) == 0
    assert qr.entity_ids.shape == (0,)


def test_query_result_entity_ids_track_rows_after_swap_remove():
    """After a swap-remove relocates rows, qr.entity_ids still aligns with qr.position: each surviving id pairs
    with its own (moved) data, never a neighbour's."""
    world = World(components=[C8])
    a = world.add_entity(components=(C8,), position=np.array([1.0, 1.0], "float32"))
    b = world.add_entity(components=(C8,), position=np.array([2.0, 2.0], "float32"))
    c = world.add_entity(components=(C8,), position=np.array([3.0, 3.0], "float32"))
    world.update()
    world.remove_entity(b)                                         # swap: c slides into b's slot
    world.update()

    qr = world.query(C8)

    assert set(qr.entity_ids.tolist()) == {a, c}
    for eid, pos in zip(qr.entity_ids, qr.position):
        np.testing.assert_array_equal(world.get_entity(int(eid)).position, pos)


def test_query_cache_returns_same_object_between_updates():
    """Two query calls for the same components, with no mutating update between, return the SAME QueryResult
    object -- the second is served from the cache, not rebuilt."""
    world = World(components=[C8])
    world.add_entity(components=(C8,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    first = world.query(C8)
    second = world.query(C8)

    assert first is second


def test_noop_update_keeps_cache():
    """An update() that commits nothing (empty command buffer) changes no pool, so the cache survives: a re-query
    returns the same object cached before that update()."""
    world = World(components=[C8])
    world.add_entity(components=(C8,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    cached = world.query(C8)
    world.update()                                  # empty buffer -> no structural change
    again = world.query(C8)

    assert again is cached


def test_mutating_update_invalidates_cache():
    """An update() that commits a structural change drops the cache: the next query is a fresh object whose len
    and entity_ids reflect the new entity, not the stale cached result."""
    world = World(components=[C8])
    a = world.add_entity(components=(C8,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    before = world.query(C8)
    assert len(before) == 1

    b = world.add_entity(components=(C8,), position=np.array([1.0, 1.0], "float32"))
    world.update()                                  # mutating commit -> cache cleared
    after = world.query(C8)

    assert after is not before
    assert len(after) == 2
    assert set(after.entity_ids.tolist()) == {a, b}


def test_cache_keyed_per_query():
    """The cache is keyed by the query, so different component sets get independent entries and never collide:
    query (C8,) and query (C9,) are distinct objects, and each key keeps returning its own."""
    world = World(components=[C8, C9])
    world.add_entity(components=(C8, C9),
                     position=np.array([0.0, 0.0], "float32"), velocity=np.array([1.0, 1.0], "float32"))
    world.update()

    pos = world.query(C8)
    vel = world.query(C9)

    assert pos is not vel                           # distinct queries -> distinct cache entries
    assert world.query(C8) is pos   # each key returns its own cached result
    assert world.query(C9) is vel


def test_new_archetype_appears_after_invalidation():
    """Spawning the first entity of a brand-new archetype and committing it must show up in a re-query: the new
    pool is not masked by a stale cached result. (C8,) matches both the position-only pool and the new
    position+velocity pool."""
    world = World(components=[C8, C9])
    world.add_entity(components=(C8,), position=np.array([0.0, 0.0], "float32"))
    world.update()

    before = world.query(C8)
    assert len(before) == 1                         # only the position-only entity so far

    world.add_entity(components=(C8, C9),                       # brand-new archetype
                     position=np.array([1.0, 1.0], "float32"), velocity=np.array([2.0, 2.0], "float32"))
    world.update()
    after = world.query(C8)

    assert len(after) == 2                           # the new pool is visible after invalidation


def test_no_stale_views_across_realloc():
    """Growing a pool past its capacity makes Pool._realloc swap in a NEW backing array, so a cached query taken
    before the growth must not be reused -- its views point at the old, freed array. After the committing
    update() the re-query is a fresh object whose views write through to the live (reallocated) pool."""
    world = World(components=[C8])
    for i in range(100):                            # fill exactly to INITIAL_CAPACITY (100)
        world.add_entity(components=(C8,), position=np.array([i, i], "float32"))
    world.update()

    stale = world.query(C8)         # views into the capacity-100 backing array
    assert len(stale) == 100

    world.add_entity(components=(C8,), position=np.array([999, 999], "float32"))  # forces _realloc(200)
    world.update()                                  # mutating commit -> cache cleared
    fresh = world.query(C8)

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
    bad = type("C25", (Component,), {"__annotations__": {reserved: np.ndarray},
                                      reserved: f32((2,))})
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
    bad = type("C25", (Component,), {"__annotations__": {reserved: np.ndarray},
                                      reserved: f32((2,))})
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
    class C20(Component):    # field carries only the always-required keys
        a: np.ndarray = f32((2,))

    class C21(Component):   # same field, plus the extra "serializable" key
        b: np.ndarray = f32((2,), serializable=True)

    World([C20])                                          # ok: {shape,dtype} == {shape,dtype}
    World([C21], extra_metadata=["serializable"])  # ok: {shape,dtype,ser} == {shape,dtype,ser}

    with pytest.raises(ValueError):                         # missing the required "serializable"
        World([C20], extra_metadata=["serializable"])
    with pytest.raises(ValueError):                         # carries "serializable" the world never declared
        World([C21])


def test_tag_component_is_valid_and_queryable():
    """A field-less component is a valid 'tag': it registers with empty field/shape/dtype maps, needs no data
    on add, lands in its own (pure-tag) pool, and is usable both as a query filter and a tag-only query."""
    world = World([C8, C13])

    assert world.component_to_field_names[C13] == []     # registers with empty per-field maps
    assert world.component_to_shapes[C13] == []
    assert world.component_to_dtypes[C13] == []

    tagged = world.add_entity((C8, C13), position=np.array([1.0, 2.0], "float32"))
    plain = world.add_entity((C8,), position=np.array([3.0, 4.0], "float32"))
    pure = world.add_entity((C13,))                      # pure tag: no data at all
    world.update()

    qr = world.query(C8, C13)             # tag as a filter: only the tagged entity, position exposed
    assert qr.entity_ids.tolist() == [tagged]
    np.testing.assert_array_equal(qr.position.numpy(), [[1.0, 2.0]])

    qr_tag = world.query(C13)                     # tag-only query spans {Pos,C13} + pure {C13} pools
    assert sorted(qr_tag.entity_ids.tolist()) == sorted([tagged, pure])
    assert qr_tag.fields == []                             # a tag exposes no fields
    assert len(qr_tag) == 2
    assert plain not in qr_tag.entity_ids.tolist()          # the untagged entity is excluded


def test_tag_component_add_remove_migrates():
    """Adding/removing a tag migrates the entity between archetypes and round-trips its data + id."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([5.0, 6.0], "float32")}})
    assert state.world.query(C3).entity_ids.tolist() == []

    state = act(state, [addc(0, C3)])
    assert state == {0: [C1, C3]}
    assert state.world.query(C3).entity_ids.tolist() == [0]
    np.testing.assert_array_equal(state.world.get_entity(0).c1, [5.0, 6.0])

    state = act(state, [rmc(0, C3)])
    assert state == {0: [C1]}
    assert state.world.query(C3).entity_ids.tolist() == []
    np.testing.assert_array_equal(state.world.get_entity(0).c1, [5.0, 6.0])


def test_zero_dim_array_field_roundtrips():
    """A field with shape () (a 0-d / scalar array) is valid: it stores, queries as (N,) and round-trips per entity."""
    world = World([C14])
    e0 = world.add_entity((C14,), scale=np.array(2.5, "float32"))
    world.add_entity((C14,), scale=np.array(4.0, "float32"))
    world.update()

    qr = world.query(C14)
    np.testing.assert_array_equal(qr.scale.numpy(), [2.5, 4.0])  # (N,) contiguous view over the 0-d field

    entity = world.get_entity(e0)
    assert entity.scale.shape == ()                            # still a 0-d scalar per entity
    np.testing.assert_array_equal(entity.scale, 2.5)


def test_add_entity_wrong_dtype_crashes_eagerly():
    """A field declared float32 must reject an int32 array *at the add_entity call*.

    Field-name validation is eager (bad name crashes at the call) and dtype is too:
    _check_components_against_pool validates dtype against component metadata, so a wrong dtype
    crashes at the call instead of slipping through to pool.add_entity at world.update()."""
    world = World(components=[C10])  # C10.radius is shape (1,) dtype float32

    # name + shape are correct here, so the only thing that can crash at the call is the dtype mismatch
    with pytest.raises(TypeError):
        world.add_entity((C10,), radius=np.zeros((1,), "int32"))  # wrong dtype must crash here


def test_add_entity_wrong_shape_crashes_eagerly():
    """A field declared shape (1,) must reject a (2,) array *at the add_entity call*.

    Companion to test_add_entity_wrong_dtype_crashes_eagerly: shape is the other half of the same guard.
    _check_components_against_pool validates shape against component metadata, so a wrong
    shape crashes at the call instead of slipping through to pool.add_entity at world.update()."""
    world = World(components=[C10])  # C10.radius is shape (1,) dtype float32

    # name + dtype are correct here, so the only thing that can crash at the call is the shape mismatch
    with pytest.raises(ValueError):
        world.add_entity((C10,), radius=np.zeros((2,), "float32"))  # wrong shape must crash here


def test_add_entity_missing_required_field_crashes_eagerly():
    """A field with default=None is REQUIRED at spawn: omitting it raises at the add_entity call, not at commit
    (where it would surface as a KeyError deep in pool materialization, mid-loop and un-attributable)."""
    world = World(components=[C8])  # C8.position has default=None

    with pytest.raises(KeyError):
        world.add_entity((C8,))     # position not supplied and has no default

    assert list(world.live_entities) == []
    assert len(world._command_buffer) == 0


def test_add_entity_converts_array_like_fields():
    """Array-like values (lists/tuples/scalars/strings) are converted to the field's dtype/shape at the call;
    ndarrays are still taken as-is and validated strictly."""
    world = World(components=[C8, C15])

    eid = world.add_entity((C8,), position=[1.0, 2.0])   # list -> float32 (2,)
    eid2 = world.add_entity((C15,), color=7)          # int scalar broadcast -> int32 (3,)
    world.update()

    pos = world.get_entity(eid).position
    assert pos.dtype == np.float32 and pos.shape == (2,)
    assert np.array_equal(pos, np.array([1.0, 2.0], "float32"))

    color = world.get_entity(eid2).color
    assert color.dtype == np.int32 and color.shape == (3,)
    assert np.array_equal(color, np.array([7, 7, 7], "int32"))


def test_staged_spawn_snapshots_the_values_at_the_call():
    """Mutating the caller's array after add_entity cannot change what the spawn commits."""
    world = World(components=[C8])
    arr = np.array([1.0, 2.0], "float32")
    world.add_entity((C8,), position=arr)
    arr[:] = 9.0
    world.update()

    np.testing.assert_array_equal(world.get_entity(0).position, [1.0, 2.0])


def test_staged_spawn_snapshots_the_callers_array():
    """Resizing the caller's array after add_entity cannot change the staged shape or poison commit."""
    world = World(components=[C8])
    arr = np.zeros(2, "float32")
    world.add_entity((C8,), position=arr)
    arr.resize(3, refcheck=False)
    world.update()

    np.testing.assert_array_equal(world.get_entity(0).position, [0.0, 0.0])


def test_staged_add_component_snapshots_the_values_at_the_call():
    """add_component shares the buffer gate: later mutation or resizing cannot change the staged row."""
    world = World(components=[C8, C9])
    world.add_entity((C8,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    velocity = np.array([3.0, 4.0], "float32")
    world.get_entity(0).add_component(C9, velocity=velocity)
    velocity[:] = 9.0
    velocity.resize(3, refcheck=False)
    world.update()

    entity = world.get_entity(0)
    assert set(entity.get_components()) == {C8, C9}
    np.testing.assert_array_equal(entity.velocity, [3.0, 4.0])


def test_spawn_loop_snapshots_each_scratch_buffer():
    """The idiomatic scratch-buffer loop commits the value at each add_entity call, not the last write."""
    world = World(components=[C8])
    scratch = np.zeros(2, "float32")
    for i in range(3):
        scratch[:] = i
        world.add_entity((C8,), position=scratch)
    world.update()

    np.testing.assert_array_equal(world.query(C8).position.numpy(), [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])


def test_staged_object_field_snapshots_the_array_not_the_object():
    """Object fields copy the array shallowly: a replaced slot is not seen, the referenced object is shared."""
    payload = {"name": "first"}
    replacement = {"name": "second"}
    arr = np.array([payload], object)

    world = World(components=[C12])
    world.add_entity((C12,), label=arr)
    arr[0] = replacement
    world.update()

    assert world.get_entity(0).label[0] is payload


def test_add_entity_rejects_non_convertible_and_bad_shape():
    """The tolerance is dtype/shape conversion, not garbage: non-convertible data and lists that cannot be
    broadcast to the field shape still raise at the call (before any entity is staged)."""
    world = World(components=[C8])

    for bad in ({"x": 1.0}, [1.0, 2.0, 3.0]):   # dict not convertible; (3,) cannot broadcast to (2,)
        with pytest.raises((TypeError, ValueError)):
            world.add_entity((C8,), position=bad)

    assert list(world.live_entities) == []
    assert len(world._command_buffer) == 0


def test_add_entity_rejects_an_empty_component_list():
    """A spawn declares at least one component: the empty archetype would be invisible to every typed query, and
    a refused spawn leaves no id or buffer entry behind."""
    world = World(components=[C8])

    with pytest.raises(ValueError, match="no components"):
        world.add_entity([])

    assert list(world.live_entities) == []
    assert len(world._command_buffer) == 0


# --- the spawn path validates ONCE ----------------------------------------------------------------------------------
# The verb has ONE owner: World.add_entity runs _validate_components + _defaults_for, and append -- whose only
# ADD_ENTITY producer is world.py:82 -- stages it verbatim. A second pass would be a superset-check of the check that
# just happened, whose _defaults_for could only return {}: ~2.0 us/spawn, 34% of add_entity, 20% of a full spawn,
# 16% of a spawn+despawn pair, landing on w5 churn (the one workload microecs loses at every N).
#
# These two catch BOTH directions: a pass added in append and a second pass anywhere else on the spawn path. The
# extra pass is invisible otherwise, which is exactly why a plain behaviour test cannot protect it -- the counter is
# the test.


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
    """One spawn, one validation pass."""
    world = World([C8])
    calls = _count_calls(world, "_validate_components")

    world.add_entity((C8,), position=np.array([1.0, 2.0], "float32"))
    world.update()

    assert len(calls) == 1, f"spawn validated {len(calls)}x -- the duplicate pass is back"


def test_spawn_computes_defaults_exactly_once():
    """Same for the defaults pass, on the archetype that actually has one to fill -- and the default still lands.
    The second pass ran on the ALREADY-merged kwargs, so it could only ever return {}: pure cost, no effect."""
    world = World([C15])
    calls = _count_calls(world, "_defaults_for")

    world.add_entity((C15,))                        # color omitted -> filled once, by add_entity
    world.update()

    assert len(calls) == 1, f"defaults computed {len(calls)}x -- the duplicate pass is back"
    pool = world.query(C15).pool_list[0]
    np.testing.assert_array_equal(pool.color[0], np.array([10, 20, 30], "int32"))


def test_rejected_spawn_validates_exactly_once_too():
    """A REJECTED spawn must not validate twice either: it raises on the first pass and stops there -- no retry,
    no second opinion. (Pins that the fix did not just move the duplicate behind the happy path.)"""
    world = World([C8])
    calls = _count_calls(world, "_validate_components")

    with pytest.raises((ValueError, TypeError, KeyError)):
        world.add_entity((C8,), position=np.array([1.0, 2.0, 3.0], "float32"))   # (3,) into a (2,) field

    assert len(calls) == 1


# --- duplicate components at spawn ---------------------------------------------------------------------------------
# _validate_components de-dupes into a set for its checks, while _get_entity_pool iterates the LIST -- so an
# accepted dup would build pool.fields == ['position', 'position']: every add/remove/realloc doing that column
# twice, forever, and to_dict() reporting the malformed component list back out. The archetype key is a bitmask, so
# the FIRST caller's argument list decides the pool's shape: later well-formed spawns of the same archetype would
# land in the bad pool. One `len(cs) != len(components)` raise (world.py:232), beside the empty/unknown checks.


def test_add_entity_rejects_duplicate_components():
    """`add_entity([Pos, Pos, Vel])` is a malformed archetype -- rejected at the call, like the empty one."""
    world = World(components=[C8, C9])

    with pytest.raises(ValueError, match="Duplicate components"):
        world.add_entity((C8, C8, C9),
                         position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))


def test_rejected_duplicate_component_spawn_leaves_nothing_behind():
    """The refusal must be complete: no id burnt, no dangling live_entities handle, nothing staged or built."""
    world = World(components=[C8, C9])

    with pytest.raises(ValueError):
        world.add_entity((C8, C8), position=np.array([1.0, 2.0], "float32"))

    assert list(world.live_entities) == []
    assert len(world._command_buffer) == 0
    world.update()
    assert len(world.pools) == 0

    # and the clean spawn that follows gets id 0 and a single-column pool
    eid = world.add_entity((C8,), position=np.array([1.0, 2.0], "float32"))
    world.update()
    assert eid == 0
    assert world._eid_to_pool_ix[eid][0].fields == ["position"]


def test_query_with_a_repeated_component_is_harmless():
    """Control for the two above: a dup in a QUERY is de-duped by the bitmask key, and must stay legal -- the new
    spawn-side raise must not leak into the read side."""
    world = World(components=[C8, C9])
    world.add_entity((C8, C9),
                     position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()

    assert len(world.query(C8, C8).entity_ids) == 1
    np.testing.assert_array_equal(world.query(C8, C8).position.numpy(), [[1.0, 2.0]])


# --- default metadata: omitted fields fall back to the component's declared default ---

def test_add_entity_fills_default_when_field_omitted():
    """Omitting a field whose metadata declares a (non-None) default fills it with that default."""
    world = World([C15])

    world.add_entity((C15,))  # color omitted -> default [10, 20, 30]
    world.update()

    pool = world.query(C15).pool_list[0]
    np.testing.assert_array_equal(pool.color[0], np.array([10, 20, 30], "int32"))


def test_add_entity_explicit_value_overrides_default():
    """An explicit value for a defaulted field wins over the default."""
    world = World([C15])

    world.add_entity((C15,), color=np.array([1, 2, 3], "int32"))
    world.update()

    pool = world.query(C15).pool_list[0]
    np.testing.assert_array_equal(pool.color[0], np.array([1, 2, 3], "int32"))


def test_default_and_explicit_coexist_per_row():
    """Same pool: an omitted field takes the default while an explicit one keeps its own value, per row."""
    world = World([C15])

    world.add_entity((C15,))                                       # row 0: default
    world.add_entity((C15,), color=np.array([1, 2, 3], "int32"))   # row 1: explicit
    world.update()

    pool = world.query(C15).pool_list[0]
    np.testing.assert_array_equal(pool.color[0], np.array([10, 20, 30], "int32"))
    np.testing.assert_array_equal(pool.color[1], np.array([1, 2, 3], "int32"))


def test_component_default_wrong_dtype_rejected():
    """A default whose dtype mismatches the declared dtype is rejected.

    Filling a default runs it through the same dtype check as an explicit value, so an int32 default for
    a float32 field raises TypeError -- at World() construction if validated there, else when the default
    is filled at the add_entity call. (Assumes the same raise-TypeError convention as the explicit-value
    dtype check; tighten if the construction-time guard chooses a different exception.)"""
    class C22(Component):
        x: np.ndarray = f32((1,), default=np.zeros((1,), "int32"))

    with pytest.raises(TypeError):
        world = World([C22])
        world.add_entity((C22,))  # omit x -> fill int32 default -> dtype mismatch


def test_component_default_wrong_shape_rejected():
    """A default whose shape mismatches the declared shape is rejected.

    Companion to test_component_default_wrong_dtype_rejected: a (2,) default for a (1,) field raises
    ValueError -- at World() construction if validated there, else when the default is filled at the
    add_entity call."""
    class C23(Component):
        x: np.ndarray = f32((1,), default=np.zeros((2,), "float32"))

    with pytest.raises(ValueError):
        world = World([C23])
        world.add_entity((C23,))  # omit x -> fill (2,) default -> shape mismatch


def test_add_component_fills_default_when_field_omitted():
    """add_component fills a defaulted field that's omitted, exactly like add_entity."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([1.0, 2.0], "float32")}})
    state = act(state, [addc(0, C2)])
    assert state == {0: [C1, C2]}

    pool = state.world.query(C1, C2).pool_list[0]
    np.testing.assert_array_equal(pool.c1[0], [1.0, 2.0])
    np.testing.assert_array_equal(pool.c2[0], [0.0, 0.0])          # metadata default


def test_add_component_explicit_value_overrides_default():
    """An explicit value passed to add_component wins over the field's default."""
    state = setup({0: [C1]}, _COMPONENTS, data={0: {"c1": np.array([1.0, 2.0], "float32")}})
    state = act(state, [addc(0, C2, c2=np.array([1.0, 2.0], "float32"))])
    assert state == {0: [C1, C2]}

    pool = state.world.query(C1, C2).pool_list[0]
    np.testing.assert_array_equal(pool.c2[0], [1.0, 2.0])


# --- field -> component map: what makes entity.set_data(**fields) possible ----------------------------------------
# set_data takes field names, no component arg, so the World must be able to answer "which component owns field
# 'pose'?". That lookup is world.field_to_component, built at __init__ and keyed by field NAME. It is only
# well-defined if names are unique across components -- so a duplicate name is rejected at world creation
# Two things to pin: the map is keyed by str (not by dataclasses.Field, which hashes by identity
# and would make both the lookup and the duplicate check silently useless), and the duplicate actually raises.

def test_field_to_component_is_keyed_by_field_name():
    """world.field_to_component maps 'field name' -> owning component type, for every field of every component."""
    world = World([C8, C11])

    assert world.field_to_component["position"] is C8
    assert world.field_to_component["lo"] is C11
    assert world.field_to_component["hi"] is C11                     # every field, not just the first
    assert set(world.field_to_component) == {"position", "lo", "hi"}     # plain strings, nothing else


def test_field_to_component_resolves_the_name_entity_set_data_passes():
    """The map is the resolution set_data does: a field name reaches the component that owns it -- which is how
    one call can span several components and still validate each field against the right schema (the write
    itself is eager, so the proof is the value in the pool)."""
    world = World([C8, C9])
    eid = world.add_entity((C8, C9),
                           position=np.array([1, 2], "float32"), velocity=np.array([3, 4], "float32"))
    world.update()

    assert world.field_to_component["velocity"] is C9
    world.get_entity(eid).set_data(velocity=np.array([9, 8], "float32"))

    np.testing.assert_array_equal(world.get_entity(eid).velocity, [9, 8])
    assert len(world._command_buffer) == 0


def test_world_accepts_the_same_component_field_names_in_separate_worlds():
    """The uniqueness rule is per-World, not global: the same component can be reused in another world."""
    World([C8, C9])
    World([C8, C11])                                        # no leakage between worlds


# --- a failed update() must not brick the world ---------------------------------------------------------------------
# update() is deliberately NOT atomic, which is a choice about *partial application* -- not a licence to leave
# the buffer unusable. Today a mid-loop failure leaves it intact, with `args["components"]` already popped off the
# command that was being applied, so EVERY later update() raises `KeyError: 'components'`. The world is bricked, and
# the error names a dict key rather than the cause. An app that logs-and-continues (any server loop) spins on it.
#
# Staged data is now snapshotted, so mutating the caller's array no longer reaches the buffer. The tests below
# corrupt the buffer's own staged snapshot instead: the recovery behavior must hold for any bad command, not only
# for the alias route that used to expose it. The failure is the same under `python -O`: the tripped guard is an
# assert, but numpy's broadcast raises anyway, hence the (AssertionError, ValueError) pair.

def _stage_a_spawn_that_will_fail_at_commit(world):
    """Stage two spawns, then widen the FIRST spawn's staged snapshot so commit fails mid-loop."""
    world.add_entity((C8,), position=np.zeros(2, "float32"))
    world.add_entity((C9,), velocity=np.zeros(2, "float32"))   # an innocent command behind it
    world._command_buffer.data[0].args["position"].resize(3, refcheck=False)


def test_failed_update_propagates_the_original_error():
    """Control for the three xfails below: whatever the fix does, it must not swallow the failure. Green today."""
    world = World(components=[C8, C9])
    _stage_a_spawn_that_will_fail_at_commit(world)

    with pytest.raises((AssertionError, ValueError)):
        world.update()


@pytest.mark.xfail(strict=True, reason="the buffer survives a failed update(), commands and all")
def test_failed_update_leaves_an_empty_buffer():
    """A buffer that outlives its own failed commit is a buffer whose commands may apply twice."""
    world = World(components=[C8, C9])
    _stage_a_spawn_that_will_fail_at_commit(world)

    with pytest.raises((AssertionError, ValueError)):
        world.update()
    assert len(world._command_buffer) == 0


@pytest.mark.xfail(strict=True, reason="the popped `components` key bricks every later update()")
def test_update_after_a_failed_update_is_a_clean_noop():
    """The headline. One bad frame must cost one frame -- not the world. Today the second update() raises
    `KeyError: 'components'`, and so does the third, and every one after it."""
    world = World(components=[C8, C9])
    _stage_a_spawn_that_will_fail_at_commit(world)

    with pytest.raises((AssertionError, ValueError)):
        world.update()

    world.update()                                                  # must not raise, and must do nothing
    world.update()
    assert len(world._command_buffer) == 0


@pytest.mark.xfail(strict=True, reason="_cache is only dropped after the loop, so a partial apply keeps it")
def test_failed_update_drops_the_query_cache():
    """A partial apply moved rows, so every cached QueryResult is suspect -- whether or not the loop finished."""
    world = World(components=[C8, C9])
    world.add_entity((C8,), position=np.array([1.0, 1.0], "float32"))
    world.update()
    world.query(C8)                                        # populate the cache
    assert len(world._cache) == 1

    _stage_a_spawn_that_will_fail_at_commit(world)
    with pytest.raises((AssertionError, ValueError)):
        world.update()
    assert len(world._cache) == 0
