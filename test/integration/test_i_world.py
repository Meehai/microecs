"""
Integration red-spec for ecs.World — structural change during iteration must be safe.

TDD target for task 3 (deferred command buffer). Instead of migrating immediately
mid-iteration (which skips entities — they land in a new pool the query snapshot
doesn't include), a system QUEUES the migration via `world.defer_add_component`,
keeps iterating its snapshot, and the main loop applies the queue at the tick
boundary via `world.update()`.

Because the migration is deferred, every entity is still in the iterated pool this
tick and gets its update -> after 2 ticks every counter == 2 (nobody skipped).

Marked xfail until task 3 lands: `world.update()` / `world.defer_add_component()`
don't exist yet, so this fails today. When the deferred buffer is implemented it
should XPASS -> drop the marker. (The immediate-API version of this loop is the bug
that motivates the task; see .tracker/todos/open/3-deferred-command-buffer.)

API names here are the assumed shape — rename in lockstep with the implementation.
"""
from dataclasses import field
import numpy as np
import pytest

from ecs import World, TickSystem, Component


class HasCounter(Component):
    counter: np.ndarray = field(metadata={"shape": (1,), "dtype": "int32"})


class HasTag(Component):  # added mid-tick to push an entity into a new archetype pool
    tag: np.ndarray = field(metadata={"shape": (1,), "dtype": "int32"})


class CountAndMigrateSystem(TickSystem):
    """Increments every counter each tick; on its first run it QUEUES a migration for
    some entities. Deferred -> they stay in the iterated pool this tick (so they ARE
    counted), and the migration is applied later by world.update()."""

    def __init__(self, to_migrate: set[int]):
        self.to_migrate = set(to_migrate)

    def on_tick(self, world: World):
        pools = world.query_and((HasCounter,))
        for eid in self.to_migrate:                                   # queue, don't apply
            world.defer_add_component(eid, HasTag, tag=np.array([0], "int32"))
        self.to_migrate.clear()
        for pool in pools:
            pool.counter[:] += 1                                      # nobody has left the pool yet


def _counter_of(world: World, eid: int) -> int:
    pool, ix = world._eid_to_pool_ix[eid]
    return int(pool.counter[ix][0])


@pytest.mark.xfail(reason="task 3: deferred buffer (world.update / world.defer_add_component) not implemented yet")
def test_deferred_structural_change_keeps_every_entity_updated():
    world = World(components=[HasCounter, HasTag])
    eids = [world.add_entity(components=(HasCounter,), counter=np.array([0], "int32")) for _ in range(5)]
    systems = [CountAndMigrateSystem({eids[1], eids[3]})]             # these migrate mid-tick

    for _ in range(2):                                                # fake main loop, 2 ticks
        for system in systems:
            system.on_tick(world)
        world.update()                                               # apply deferred buffer at tick boundary

    counters = [_counter_of(world, eid) for eid in eids]
    assert counters == [2, 2, 2, 2, 2]                               # deferred migration -> nobody skipped
