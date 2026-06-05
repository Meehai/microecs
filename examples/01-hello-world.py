#!/usr/bin/env python3
"""01-hello-world.py The basic hello world for ECS. Creates some static balls. You can add some with the mouse."""
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

# components

class HasRadius(Component):
    radius: np.ndarray = field(metadata={"shape": (1, ), "dtype": "float32"})

class HasPosition2D(Component):
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})

class HasColor(Component):
    color: np.ndarray = field(metadata={"shape": (4, ), "dtype": "int32"})

# systems

class RenderSystem:
    def __call__(self, world: World):
        qr = world.query_and((HasRadius, HasPosition2D, HasColor))
        for position, radius, color in zip(qr.position, qr.radius, qr.color):
            rl.DrawCircle(int(position[0].item()), int(position[1].item()), int(radius.item()), color.tolist())

class CollisionSystem:
    def __call__(self, world: World):
        qr = world.query_and((HasPosition2D, HasRadius, HasColor))
        _red = np.array(rl.RED, dtype="int32")[None].repeat(len(qr), axis=0)
        _black = np.array(rl.BLACK, dtype="int32")[None].repeat(len(qr), axis=0)
        collisions = self._get_collisions(qr.position.numpy(), qr.radius.numpy())
        qr.color[:] = np.where(collisions, _red, _black)

    def _get_collisions(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        dists = np.sqrt(((positions[:, None] - positions[None])**2).sum(-1))  # (N, 1, 2) - (1, N, 2) -> ... -> (N, N)
        radii_sum =  (radii[None] + radii[:, None])[..., 0] # (N, N)
        collisions_nn = (dists < radii_sum) - np.eye(len(positions)) # (N, N)
        res = (collisions_nn.sum(axis=1) > 0)[..., None] # (N, 1)
        return res

def main(args: Namespace):
    rl.InitWindow(800, 800, b"Entity Component Style + SoA (batched)")
    scene_size = (600, 600)

    render_system = RenderSystem()
    update_systems: list[Callable] = [CollisionSystem()]

    world = World(components=[HasRadius, HasPosition2D, HasColor])
    for _ in range(args.n_objects):
        radius = random.randint(5, 20)
        position = random.randint(radius, scene_size[0] - radius), random.randint(radius, scene_size[1] - radius)
        world.add_entity(components=(HasRadius, HasPosition2D, HasColor), position=np.array(position, "float32"),
                         color=np.array(rl.BLACK, dtype="int32"), radius=np.array([radius], "float32"),)

    while not rl.WindowShouldClose():
        world.update()
        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
            radius = random.randint(5, 20)
            position = rl.GetMousePosition().x, rl.GetMousePosition().y
            world.add_entity(components=(HasRadius, HasPosition2D, HasColor), position=np.array(position, "float32"),
                             color=np.array(rl.BLACK, dtype="int32"), radius=np.array([radius], "float32"),)

        _ = [system(world=world) for system in update_systems]

        rl.BeginDrawing()
        rl.ClearBackground(rl.RAYWHITE)
        rl.DrawFPS(rl.GetScreenWidth() - 100, 0)

        rl.DrawRectangleLinesEx((0, 0, *scene_size), 2, rl.BLACK)
        render_system(world=world)

        rl.EndDrawing()

        logger.log_every_s(f"FPS: {rl.GetFPS()}", "DEBUG")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_objects", type=int, default=10)
    main(parser.parse_args())
