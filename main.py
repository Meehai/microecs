#!/usr/bin/env python3
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import random
import numpy as np
import raylib as rl
from loggez import loggez_logger as logger

from ecs import World, TickSystem
from ecs.utils import Clock

Point2D = tuple[float, float]
DT = 1 / 100
MAX_SUBTICKS_PER_RENDER_TICK = 3

# traits

@dataclass(kw_only=True)
class HasRadius: # for drawing circles basically
    radius: np.ndarray = field(metadata={"shape": (1, ), "dtype": "float32"})

@dataclass(kw_only=True)
class HasPosition2D:
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})

@dataclass(kw_only=True)
class HasMotion2D:
    velocity: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})

@dataclass(kw_only=True)
class HasColor:
    color: np.ndarray = field(metadata={"shape": (4, ), "dtype": "int32"})

# systems

class RenderSystem(TickSystem):
    def on_tick(self, scene: World):
        entity_pools = scene.query_and((HasRadius, HasPosition2D, HasColor))
        for pool in entity_pools:
            for position, radius, color in zip(pool.position, pool.radius, pool.color):
                rl.DrawCircle(int(position[0].item()), int(position[1].item()), int(radius.item()), color.tolist())

class MotionSystem(TickSystem):
    def on_tick(self, scene: World):
        entity_pools = scene.query_and((HasMotion2D, HasPosition2D))
        for pool in entity_pools:
            pool.position[:] = pool.position + pool.velocity * DT # (N, 2)

class WallBounceSystem(TickSystem):
    def __init__(self, scene_size: tuple[int, int]):
        self.scene_size = scene_size

    def on_tick(self, scene: World):
        entity_pools = scene.query_and((HasPosition2D, HasMotion2D, HasRadius))

        for pool in entity_pools:
            mask_velocity = np.zeros((len(pool.position), 2), bool)
            mask_velocity[:, 0] = np.logical_or(pool.position[:, 0] - pool.radius[:, 0] < 0,
                                                pool.position[:, 0] + pool.radius[:, 0] > self.scene_size[0])
            mask_velocity[:, 1] = np.logical_or(pool.position[:, 1] - pool.radius[:, 0] < 0,
                                                pool.position[:, 1] + pool.radius[:, 0] > self.scene_size[1])
            pool.velocity[:] = np.where(mask_velocity, -pool.velocity, pool.velocity)

class CollisionBounceSystem(TickSystem):
    def on_tick(self, scene: World):
        entity_pools = scene.query_and((HasPosition2D, HasMotion2D, HasRadius, HasColor))

        for pool in entity_pools:
            _red = np.array(rl.RED, dtype="int32")[None].repeat(len(pool), axis=0)
            _black = np.array(rl.BLACK, dtype="int32")[None].repeat(len(pool), axis=0)
            collisions = self._get_collisions(pool.position, pool.radius)
            pool.color[:] = np.where(collisions, _red, _black)

    def _get_collisions(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        dists = np.sqrt(((positions[:, None] - positions[None])**2).sum(-1))  # (N, 1, 2) - (1, N, 2) -> ... -> (N, N)
        radii_sum =  (radii[None] + radii[:, None])[..., 0] # (N, N)
        collisions_nn = (dists < radii_sum) - np.eye(len(positions)) # (N, N)
        res = (collisions_nn.sum(axis=1) > 0)[..., None] # (N, 1)
        return res

def main(args: Namespace):
    rl.InitWindow(800, 800, b"Entity Component Style + SoA (batched)")
    scene_size = (600, 600)

    render_systems: list[TickSystem] = [RenderSystem()]
    update_systems: list[TickSystem] = [MotionSystem(), WallBounceSystem(scene_size), CollisionBounceSystem()]

    scene = World(traits=[HasRadius, HasPosition2D, HasMotion2D, HasColor])

    for _ in range(args.n_objects):
        radius = random.randint(3, 7)
        position = random.randint(radius, scene_size[0] - radius), random.randint(radius, scene_size[1] - radius)
        velocity = (200 * random.random() * 2 - 1, 200 * random.random() * 2 - 1)
        scene.add_entity(traits=(HasRadius, HasPosition2D, HasMotion2D, HasColor),
                         position=np.array(position, "float32"), velocity=np.array(velocity, "float32"),
                         color=np.array(rl.BLACK, dtype="int32"), radius=np.array([radius], "float32"),)

    clock = Clock(dt=DT, max_ticks=MAX_SUBTICKS_PER_RENDER_TICK)
    while not rl.WindowShouldClose():
        rl.WaitTime(max(DT - (rl.GetTime() - clock.prev_time), 0))
        clock.tick()

        for _ in clock.drain():
            logger.log_every_s(f"Applying {clock.accumulator // clock.dt} update ticks per render tick", "DEBUG", True)
            _ = [system.on_tick(scene=scene) for system in update_systems]

        rl.BeginDrawing()
        rl.ClearBackground(rl.RAYWHITE)
        rl.DrawFPS(rl.GetScreenWidth() - 100, 0)

        rl.DrawRectangleLinesEx((0, 0, *scene_size), 2, rl.BLACK)
        _ = [system.on_tick(scene=scene) for system in render_systems]

        rl.EndDrawing()

        logger.log_every_s(f"FPS: {rl.GetFPS()}", "INFO")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_objects", type=int, default=10)
    main(parser.parse_args())
