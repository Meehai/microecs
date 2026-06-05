"""Integration: a full-N raw mask mixed with _Field operands, across more than one archetype.

`example 02 WallBounceSystem` builds a raw (N, 2) boolean mask and feeds it to
`np.where(mask, -qr.velocity, qr.velocity)`: the mask is a plain numpy array spanning every matched entity,
the velocities are _Field (per-pool views). Per-pool dispatch splits the full-N mask into per-pool chunks
(like a _Field), so the WallBounce idiom works through a real World query that matches two pools.
"""
from dataclasses import field
import numpy as np

from microecs import World, Component

class HasPosition2D(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasMotion2D(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasRadius(Component):
    radius: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32"})


class HasColor(Component):
    color: np.ndarray = field(metadata={"shape": (4,), "dtype": "int32"})


class WallBounceSystem:
    """Verbatim idiom from example 02: a raw (N, 2) mask built per column, then a mixed np.where write-back."""
    def __init__(self, scene_size):
        self.scene_size = scene_size

    def __call__(self, world):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius)
        mask_velocity = np.zeros((len(qr.position), 2), bool)
        mask_velocity[:, 0] = np.logical_or(qr.position[:, 0] - qr.radius[:, 0] < 0,
                                            qr.position[:, 0] + qr.radius[:, 0] > self.scene_size[0])
        mask_velocity[:, 1] = np.logical_or(qr.position[:, 1] - qr.radius[:, 0] < 0,
                                            qr.position[:, 1] + qr.radius[:, 0] > self.scene_size[1])
        qr.velocity[:] = np.where(mask_velocity, -qr.velocity, qr.velocity)


def test_full_n_mask_writes_back_across_two_archetypes():
    """Two archetypes both match (HasPosition2D, HasMotion2D, HasRadius): one bare {pos, vel, rad}, one richer
    {pos, vel, rad, color}. A wall-bounce over the combined query flips exactly the entities touching a wall,
    in WHICHEVER pool they live -- the global (N, 2) mask must split correctly across both pools."""
    scene = (600, 600)
    world = World(components=[HasPosition2D, HasMotion2D, HasRadius, HasColor])

    # archetype A: {pos, vel, rad}.  e1 hits the left wall (x-bounce); e2 is interior (no bounce).
    world.add_entity(components=(HasPosition2D, HasMotion2D, HasRadius),
                     position=np.array([0.0, 300.0], "float32"), velocity=np.array([3.0, 3.0], "float32"),
                     radius=np.array([5.0], "float32"))
    world.add_entity(components=(HasPosition2D, HasMotion2D, HasRadius),
                     position=np.array([300.0, 300.0], "float32"), velocity=np.array([3.0, 3.0], "float32"),
                     radius=np.array([5.0], "float32"))
    # archetype B: {pos, vel, rad, color}.  e3 hits the right wall (x-bounce); e4 hits the bottom (y-bounce).
    for pos in ([599.0, 300.0], [300.0, 599.0]):
        world.add_entity(components=(HasPosition2D, HasMotion2D, HasRadius, HasColor),
                         position=np.array(pos, "float32"), velocity=np.array([3.0, 3.0], "float32"),
                         radius=np.array([5.0], "float32"), color=np.zeros(4, "int32"))
    world.update()

    pool_a = world.pools[world._make_key((HasPosition2D, HasMotion2D, HasRadius))]
    pool_b = world.pools[world._make_key((HasPosition2D, HasMotion2D, HasRadius, HasColor))]
    assert len(pool_a) == 2 and len(pool_b) == 2          # the query really spans two pools

    WallBounceSystem(scene)(world)

    np.testing.assert_array_equal(pool_a.velocity, [[-3.0, 3.0], [3.0, 3.0]])   # e1 x-bounced, e2 untouched
    np.testing.assert_array_equal(pool_b.velocity, [[-3.0, 3.0], [3.0, -3.0]])  # e3 x-bounced, e4 y-bounced
