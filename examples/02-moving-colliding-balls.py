#!/usr/bin/env python3
from dataclasses import field
from typing import Callable
from argparse import ArgumentParser, Namespace
import random
import numpy as np
import raylib as rl
from loggez import loggez_logger as logger

from microecs import World, Component

Point2D = tuple[float, float]
DT = 1 / 100
MAX_SUBTICKS_PER_RENDER_TICK = 3

# utils

class Clock:
    """clock used for physics with fixed DT in main loops"""

    def __init__(self, dt: float, max_ticks: int):
        self.dt = dt
        self.max_ticks = max_ticks
        self.prev_time = rl.GetTime()
        self.accumulator = 0

    def tick(self):
        """tick once by adding the delta between prev frame and now"""
        now = rl.GetTime()
        frame_time = now - self.prev_time
        self.prev_time = now
        self.accumulator += frame_time

    def drain(self):
        """drain the accumulator. in main loop: for _ in clock.drain(): ..."""
        n_ticks = 0
        while self.accumulator >= self.dt and n_ticks < self.max_ticks:
            yield
            self.accumulator -= self.dt
            n_ticks += 1
        self.accumulator = min(self.accumulator, self.dt) # Drop residual debt instead of it piling up across frames

    def wait(self):
        """waits the leftover time in case the previous tick ran too fast to maintain consistent FPS"""
        rl.WaitTime(max(self.dt - (rl.GetTime() - self.prev_time), 0))

    def wait_and_tick(self):
        """calls wait() then tick(). Put this at the beginning of the main loop :)"""
        self.wait()
        self.tick()

# components

class HasRadius(Component):
    radius: np.ndarray = field(metadata={"shape": (1, ), "dtype": "float32"})

class HasPosition2D(Component):
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})

class HasMotion2D(Component):
    velocity: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})

class HasCollision(Component):
    is_colliding: np.ndarray = field(metadata={"shape": (1, ), "dtype": "bool"})

# systems

class RenderSystem:
    def __call__(self, world: World):
        qr = world.query(HasRadius, HasPosition2D, HasCollision)
        for position, radius, is_colliding in zip(qr.position, qr.radius, qr.is_colliding):
            color = rl.RED if is_colliding else rl.BLACK
            rl.DrawCircle(int(position[0].item()), int(position[1].item()), int(radius.item()), color)

class MotionSystem:
    def __call__(self, world: World):
        qr = world.query(HasMotion2D, HasPosition2D)
        qr.position[:] = qr.position + qr.velocity * DT # (N, 2)

class WallBounceSystem:
    def __init__(self, scene_size: tuple[int, int]):
        self.scene_size = scene_size

    def __call__(self, world: World):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius)
        mask_velocity = np.zeros((len(qr.position), 2), bool)
        mask_velocity[:, 0] = np.logical_or(qr.position[:, 0] - qr.radius[:, 0] < 0,
                                            qr.position[:, 0] + qr.radius[:, 0] > self.scene_size[0])
        mask_velocity[:, 1] = np.logical_or(qr.position[:, 1] - qr.radius[:, 0] < 0,
                                            qr.position[:, 1] + qr.radius[:, 0] > self.scene_size[1])
        qr.velocity[:] = np.where(mask_velocity, -qr.velocity, qr.velocity)

class CollisionDetectionSystem:
    def __call__(self, world: World):
        qr = world.query(HasPosition2D, HasMotion2D, HasRadius, HasCollision)
        collisions = self._get_collisions(qr.position.numpy(), qr.radius.numpy())
        qr.is_colliding[:] = np.where(collisions, True, False)

    def _get_collisions(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        dists = np.sqrt(((positions[:, None] - positions[None])**2).sum(-1))  # (N, 1, 2) - (1, N, 2) -> ... -> (N, N)
        radii_sum =  (radii[None] + radii[:, None])[..., 0] # (N, N)
        collisions_nn = (dists < radii_sum) - np.eye(len(positions)) # (N, N)
        res = (collisions_nn.sum(axis=1) > 0)[..., None] # (N, 1)
        return res

def _spawn_circle(world: World, position: Point2D, radius: float, velocity: Point2D):
    world.add_entity(components=(HasRadius, HasPosition2D, HasMotion2D, HasCollision),
                     position=np.array(position, "float32"), velocity=np.array(velocity, "float32"),
                     radius=np.array([radius], "float32"), is_colliding=np.zeros((1, ), "bool"))

def main(args: Namespace):
    rl.InitWindow(800, 800, b"Entity Component Style + SoA (batched)")
    scene_size = (600, 600)
    mouse_radius = 10

    render_systems: list[Callable] = [RenderSystem()]
    update_systems: list[Callable] = [MotionSystem(), WallBounceSystem(scene_size), CollisionDetectionSystem()]

    world = World(components=[HasRadius, HasPosition2D, HasMotion2D, HasCollision])
    for _ in range(args.n_objects):
        radius = random.randint(5, 15)
        position = random.randint(radius, scene_size[0] - radius), random.randint(radius, scene_size[1] - radius)
        velocity = (200 * random.random() * 2 - 1, 200 * random.random() * 2 - 1)
        _spawn_circle(world, position, radius, velocity)

    clock = Clock(dt=DT, max_ticks=MAX_SUBTICKS_PER_RENDER_TICK)
    while not rl.WindowShouldClose():
        world.update()
        clock.wait_and_tick()

        mouse_pos = rl.GetMousePosition()

        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
                if (mouse_pos.x - mouse_radius > 0 and mouse_pos.x + mouse_radius < scene_size[0] and
                    mouse_pos.y - mouse_radius > 0 and mouse_pos.y + mouse_radius < scene_size[1]):
                    velocity = (200 * random.random() * 2 - 1, 200 * random.random() * 2 - 1)
                    _spawn_circle(world, (mouse_pos.x, mouse_pos.y), mouse_radius, velocity)

        for _ in clock.drain():
            logger.log_every_s(f"Applying {clock.accumulator // clock.dt} update ticks per render tick", "DEBUG", True)
            _ = [system(world=world) for system in update_systems]

        rl.BeginDrawing()
        rl.ClearBackground(rl.RAYWHITE)
        rl.DrawFPS(rl.GetScreenWidth() - 100, 0)

        rl.DrawRectangleLinesEx((0, 0, *scene_size), 2, rl.BLACK)
        _ = [system(world=world) for system in render_systems]

        rl.EndDrawing()

        logger.log_every_s(f"FPS: {rl.GetFPS()}", "DEBUG")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_objects", type=int, default=10)
    main(parser.parse_args())
