"""Integration: structural changes queued by multiple systems in ONE tick, committed by world.update().

These exercise the deferred command buffer the way a real main loop does: several systems each queue
spawns / despawns / component migrations during a tick, then a single world.update() applies the whole
buffer in enqueue order. Each test packs several related corner cases (own fresh world per block, so
re-running is idempotent). Self-contained: own components/systems, no cross-file imports.

These cover the cases that WORK. The fatal "operate on an entity removed earlier this tick" cases live
in test/unit/test_world.py as eager-validation specs (they should fail at the call, not at commit).
"""
from dataclasses import field
import numpy as np

from microecs import World, Component


class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasRadius(Component):
    radius: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32"})

class SpawnSystem:
    """Spawns one entity, recording its eager id into `sink` so later systems can act on it this tick."""
    def __init__(self, components, fields, sink):
        self.components, self.fields, self.sink = components, fields, sink

    def __call__(self, world):
        self.sink.append(world.add_entity(components=self.components, **self.fields))


class DespawnSystem:
    def __init__(self, ids):
        self.ids = ids

    def __call__(self, world):
        for eid in list(self.ids):
            world.remove_entity(eid)


class AddComponentSystem:
    def __init__(self, ids, component, fields):
        self.ids, self.component, self.fields = ids, component, fields

    def __call__(self, world):
        for eid in list(self.ids):
            world.add_component(eid, self.component, **self.fields)


class RemoveComponentSystem:
    def __init__(self, ids, component):
        self.ids, self.component = ids, component

    def __call__(self, world):
        for eid in list(self.ids):
            world.remove_component(eid, self.component)


def _run_systems(world, systems):
    """Run each system's on_tick (queuing structural changes), then commit once -- a single tick."""
    for system in systems:
        system(world)
    world.update()


def _count(world):
    return sum(len(pool) for pool in world.pools.values())


def test_same_tick_widening_across_systems():
    """Spawn + two add_components in one tick: chained migrations land the entity in the full archetype.
    Also: two systems widening a pre-existing entity with different components in one tick."""
    # spawn then widen twice, same tick -> {pos, vel, rad}
    world = World(components=[HasPosition, HasVelocity, HasRadius])
    spawned = []
    _run_systems(world, [
        SpawnSystem((HasPosition,), {"position": np.array([1.0, 2.0], "float32")}, spawned),
        AddComponentSystem(spawned, HasVelocity, {"velocity": np.array([3.0, 4.0], "float32")}),
        AddComponentSystem(spawned, HasRadius, {"radius": np.array([5.0], "float32")}),
    ])
    pool, ix = world._eid_to_pool_ix[spawned[0]]
    assert pool is world.pools[world._make_key((HasPosition, HasVelocity, HasRadius))]
    np.testing.assert_array_equal(pool.position[ix], [1.0, 2.0])
    np.testing.assert_array_equal(pool.velocity[ix], [3.0, 4.0])
    np.testing.assert_array_equal(pool.radius[ix], [5.0])
    assert _count(world) == 1

    # pre-existing entity widened by two different systems, same tick -> same full archetype
    world = World(components=[HasPosition, HasVelocity, HasRadius])
    eid = world.add_entity(components=(HasPosition,), position=np.array([0.0, 0.0], "float32"))
    world.update()
    _run_systems(world, [
        AddComponentSystem([eid], HasVelocity, {"velocity": np.array([1.0, 1.0], "float32")}),
        AddComponentSystem([eid], HasRadius, {"radius": np.array([2.0], "float32")}),
    ])
    pool, ix = world._eid_to_pool_ix[eid]
    assert pool is world.pools[world._make_key((HasPosition, HasVelocity, HasRadius))]
    assert _count(world) == 1


def test_same_tick_component_round_trips_across_systems():
    """add-then-remove of a component nets back to the original archetype; remove-then-readd takes the NEW data."""
    # add X then remove X, same tick -> back to {pos}, transient {pos, vel} reclaimed
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition,), position=np.array([5.0, 6.0], "float32"))
    world.update()
    _run_systems(world, [
        AddComponentSystem([eid], HasVelocity, {"velocity": np.array([7.0, 8.0], "float32")}),
        RemoveComponentSystem([eid], HasVelocity),
    ])
    pool, ix = world._eid_to_pool_ix[eid]
    assert pool is world.pools[world._make_key((HasPosition,))]
    assert world._make_key((HasPosition, HasVelocity)) not in world.pools
    np.testing.assert_array_equal(pool.position[ix], [5.0, 6.0])
    assert not hasattr(pool, "velocity")

    # remove X then re-add X with new data, same tick -> ends in {pos, vel} carrying the re-added value
    world = World(components=[HasPosition, HasVelocity])
    eid = world.add_entity(components=(HasPosition, HasVelocity),
                           position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.update()
    _run_systems(world, [
        RemoveComponentSystem([eid], HasVelocity),
        AddComponentSystem([eid], HasVelocity, {"velocity": np.array([9.0, 9.0], "float32")}),
    ])
    pool, ix = world._eid_to_pool_ix[eid]
    assert pool is world.pools[world._make_key((HasPosition, HasVelocity))]
    np.testing.assert_array_equal(pool.position[ix], [1.0, 2.0])           # untouched
    np.testing.assert_array_equal(pool.velocity[ix], [9.0, 9.0])           # the re-added value, not the old [3, 4]


def test_same_tick_lifecycle_across_systems():
    """Spawn+despawn nets to nothing; despawn+spawn conserves count; migrate-all+despawn-one reclaims & resolves;
    spawn-all+despawn-all empties the world."""
    # spawn then despawn the newcomer -> net no-op, sibling untouched
    world = World(components=[HasPosition])
    keep = world.add_entity(components=(HasPosition,), position=np.array([7.0, 7.0], "float32"))
    world.update()
    spawned = []
    _run_systems(world, [
        SpawnSystem((HasPosition,), {"position": np.array([0.0, 0.0], "float32")}, spawned),
        DespawnSystem(spawned),
    ])
    assert spawned[0] not in world._eid_to_pool_ix
    assert keep in world._eid_to_pool_ix
    assert _count(world) == 1

    # despawn the middle of three + spawn a newcomer -> count conserved (3 - 1 + 1), survivors resolve
    world = World(components=[HasPosition])
    ids = [world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32")) for i in range(3)]
    world.update()
    spawned = []
    _run_systems(world, [
        DespawnSystem([ids[1]]),
        SpawnSystem((HasPosition,), {"position": np.array([9.0, 9.0], "float32")}, spawned),
    ])
    assert ids[1] not in world._eid_to_pool_ix
    assert _count(world) == 3
    for eid, expected in ((ids[0], [0, 0]), (ids[2], [2, 2]), (spawned[0], [9, 9])):
        pool, ix = world._eid_to_pool_ix[eid]
        np.testing.assert_array_equal(pool.position[ix], expected)

    # migrate ALL to {pos, vel} + despawn one, same tick -> old pool reclaimed, survivors in richer pool
    world = World(components=[HasPosition, HasVelocity])
    ids = [world.add_entity(components=(HasPosition,), position=np.array([i, i], "float32")) for i in range(3)]
    world.update()
    _run_systems(world, [
        AddComponentSystem(ids, HasVelocity, {"velocity": np.array([1.0, 1.0], "float32")}),
        DespawnSystem([ids[0]]),
    ])
    assert ids[0] not in world._eid_to_pool_ix
    assert world._make_key((HasPosition,)) not in world.pools
    assert _count(world) == 2
    for eid, expected in ((ids[1], [1, 1]), (ids[2], [2, 2])):
        pool, ix = world._eid_to_pool_ix[eid]
        assert pool is world.pools[world._make_key((HasPosition, HasVelocity))]
        np.testing.assert_array_equal(pool.position[ix], expected)

    # spawn several + despawn all of them, same tick -> world ends empty, pools reclaimed
    world = World(components=[HasPosition])
    spawned = []
    _run_systems(world, [
        SpawnSystem((HasPosition,), {"position": np.array([1.0, 1.0], "float32")}, spawned),
        SpawnSystem((HasPosition,), {"position": np.array([2.0, 2.0], "float32")}, spawned),
        DespawnSystem(spawned),
    ])
    assert _count(world) == 0
    assert world.pools == {}
    assert world._eid_to_pool_ix == {}
