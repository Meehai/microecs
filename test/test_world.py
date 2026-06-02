"""Unit tests for ecs.World"""
from dataclasses import dataclass, field
import numpy as np
import pytest

from ecs import World


@dataclass(kw_only=True)
class HasPosition:
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


@dataclass(kw_only=True)
class HasVelocity:
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


@dataclass(kw_only=True)
class HasRadius:
    radius: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32"})


def test_add_entity_rejects_field_from_an_unrequested_trait():
    """An entity declared with only HasPosition may not pass `velocity` (a field of the unrequested HasVelocity)."""
    world = World(traits=[HasPosition, HasVelocity])  # both traits known to the world

    with pytest.raises(AssertionError, match="velocity"):
        world.add_entity(
            traits=(HasPosition,),                          # entity declares HasPosition only
            position=np.array([1.0, 2.0], "float32"),       # required by HasPosition
            velocity=np.array([3.0, 4.0], "float32"),       # extra: belongs to HasVelocity, not requested
        )


def test_fresh_world_has_no_pools():
    """A world creates pools lazily; before any add_entity there are none."""
    world = World(traits=[HasPosition, HasVelocity])
    assert world.pools == {}


def test_add_one_entity_creates_exactly_one_pool_with_one_entity():
    """First add_entity creates a single pool, holding that single entity."""
    world = World(traits=[HasPosition, HasVelocity])

    world.add_entity(traits=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    assert len(world.pools) == 1
    pool = next(iter(world.pools.values()))
    assert len(pool) == 1


def test_entity_lands_in_the_pool_keyed_by_its_traits():
    """The entity goes into the pool whose key is exactly the bitmask of its declared traits."""
    world = World(traits=[HasPosition, HasVelocity])

    world.add_entity(traits=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    key = world._make_key((HasPosition,))
    assert key in world.pools                                   # a pool with that exact key exists
    assert world.pools[key] is world.query_and((HasPosition,))[0]  # and query_and finds the same pool


def test_added_entity_field_values_are_stored():
    """The component data we passed is readable back from the pool, unchanged."""
    world = World(traits=[HasPosition, HasVelocity])

    world.add_entity(traits=(HasPosition,), position=np.array([1.5, 2.5], "float32"))

    pool = world.query_and((HasPosition,))[0]
    np.testing.assert_array_equal(pool.position[0], np.array([1.5, 2.5], "float32"))


def test_same_archetype_entities_share_a_single_pool():
    """N entities with the same trait set all land in one pool, in insertion order."""
    world = World(traits=[HasPosition, HasVelocity])

    for i in range(3):
        world.add_entity(traits=(HasPosition,), position=np.array([i, i], "float32"))

    assert len(world.pools) == 1                                # still just one archetype
    pool = world.query_and((HasPosition,))[0]
    assert len(pool) == 3
    np.testing.assert_array_equal(pool.position, np.array([[0, 0], [1, 1], [2, 2]], "float32"))


def test_distinct_archetypes_get_distinct_pools():
    """Entities with different trait sets are stored in separate pools."""
    world = World(traits=[HasPosition, HasVelocity])

    world.add_entity(traits=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(traits=(HasPosition, HasVelocity),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))

    assert len(world.pools) == 2
    assert len(world.query_and((HasPosition,))[0]) >= 1
    pos_vel_pool = world.pools[world._make_key((HasPosition, HasVelocity))]
    assert len(pos_vel_pool) == 1


def test_trait_order_does_not_create_a_second_pool():
    """(HasPosition, HasVelocity) and (HasVelocity, HasPosition) are the same archetype."""
    world = World(traits=[HasPosition, HasVelocity])

    world.add_entity(traits=(HasPosition, HasVelocity),
                     position=np.array([1.0, 2.0], "float32"), velocity=np.array([3.0, 4.0], "float32"))
    world.add_entity(traits=(HasVelocity, HasPosition),
                     position=np.array([5.0, 6.0], "float32"), velocity=np.array([7.0, 8.0], "float32"))

    assert len(world.pools) == 1                                # key is a bitmask, order-independent
    assert len(next(iter(world.pools.values()))) == 2


def test_query_and_returns_all_pools_that_are_supersets():
    """query_and((HasPosition,)) returns every pool containing HasPosition, not just the pos-only one."""
    world = World(traits=[HasPosition, HasVelocity, HasRadius])

    world.add_entity(traits=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(traits=(HasPosition, HasVelocity),
                     position=np.array([3.0, 4.0], "float32"), velocity=np.array([5.0, 6.0], "float32"))

    assert len(world.query_and((HasPosition,))) == 2            # both pools contain HasPosition
    assert len(world.query_and((HasPosition, HasVelocity))) == 1  # only the richer pool has both


def test_query_and_is_empty_when_no_pool_has_the_trait():
    """Querying a trait that no existing pool carries returns no pools."""
    world = World(traits=[HasPosition, HasVelocity])

    world.add_entity(traits=(HasPosition,), position=np.array([1.0, 2.0], "float32"))

    assert world.query_and((HasVelocity,)) == []


def test_entities_are_conserved_across_pools():
    """Summing len over all pools equals the number of entities added."""
    world = World(traits=[HasPosition, HasVelocity])

    world.add_entity(traits=(HasPosition,), position=np.array([1.0, 2.0], "float32"))
    world.add_entity(traits=(HasPosition,), position=np.array([3.0, 4.0], "float32"))
    world.add_entity(traits=(HasPosition, HasVelocity),
                     position=np.array([5.0, 6.0], "float32"), velocity=np.array([7.0, 8.0], "float32"))

    assert sum(len(pool) for pool in world.pools.values()) == 3


def test_pop_then_migrate_entity_to_a_richer_archetype():
    """Two entities share a pool; pop one, give it a new trait, re-add it -> it moves to a second pool."""
    world = World(traits=[HasPosition, HasVelocity])

    world.add_entity(traits=(HasPosition,), position=np.array([1.0, 1.0], "float32"))
    world.add_entity(traits=(HasPosition,), position=np.array([2.0, 2.0], "float32"))

    pos_key = world._make_key((HasPosition,))
    pos_vel_key = world._make_key((HasPosition, HasVelocity))
    assert len(world.pools) == 1                            # both share the position-only pool
    assert len(world.pools[pos_key]) == 2

    popped = world.pools[pos_key].pop_entity(1)             # take the 2nd entity out of pool 1
    np.testing.assert_array_equal(popped["position"], [2.0, 2.0])

    world.add_entity(traits=(HasPosition, HasVelocity),     # add a trait -> richer archetype -> pool 2
                     position=popped["position"], velocity=np.array([9.0, 9.0], "float32"))

    assert len(world.pools) == 2

    pos_pool = world.pools[pos_key]                         # 1st entity stays alone in pool 1
    assert len(pos_pool) == 1
    np.testing.assert_array_equal(pos_pool.position[0], [1.0, 1.0])

    pos_vel_pool = world.pools[pos_vel_key]                 # migrated entity is alone in pool 2
    assert len(pos_vel_pool) == 1
    np.testing.assert_array_equal(pos_vel_pool.position[0], [2.0, 2.0])
    np.testing.assert_array_equal(pos_vel_pool.velocity[0], [9.0, 9.0])
