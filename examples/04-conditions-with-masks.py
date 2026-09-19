#!/usr/bin/env python3
"""04-conditions-with-masks. Three types of objects that are colored differently. Keybinds: 1, 2, 3 (+click)"""
from argparse import ArgumentParser, Namespace
from enum import IntEnum
import random
import numpy as np
import raylib as rl
from loggez import loggez_logger as logger

from microecs import World, Component, f32, obj, zeros

Point2D = tuple[float, float]
DT = 1 / 100
MAX_SUBTICKS_PER_RENDER_TICK = 3

# components

# NOTE: A different (more ECS-ish) design pattern would involve creating one component per shape kind and then you could
# just do world.query(HasRectangle) for example. We do it this way as there are situations where we want a subset of
# entities that have the same component (e.g. HasShape here) based on some particular value (ShapeKind here).
# This operation is similar to SQL's `SELECT x from y where z > 5`, while in ECS you usually just do `select x from y`.

class ShapeKind(IntEnum):
    CIRCLE    = 0
    RECTANGLE = 1

class HasShape(Component):
    kind:   np.ndarray = obj((1, ))
    radius: np.ndarray = f32((1, ), default=zeros)
    width:  np.ndarray = f32((1, ), default=zeros)
    height: np.ndarray = f32((1, ), default=zeros)

class HasPosition(Component):
    position: np.ndarray = f32((2, ))

# systems

class RenderSystem:
    def __call__(self, world: World):
        kind_to_color = {
            ShapeKind.CIRCLE: rl.RED,
            ShapeKind.RECTANGLE: rl.GREEN,
        }


        qr = world.query(HasShape, HasPosition)

        for kind in [ShapeKind.CIRCLE, ShapeKind.RECTANGLE]:
            color = kind_to_color[kind]
            mask = (qr.kind == kind)[..., 0] # (N, ) mask
            positions = qr.position[mask]    # (n<N) array
            if kind == ShapeKind.CIRCLE:
                radii = qr.radius[mask]      # (n<N) array
                for position, radius in zip(positions, radii):
                    rl.DrawCircle(int(position[0].item()), int(position[1].item()), int(radius.item()), color)
            else: # rectangle
                widths = qr.width[mask]      # (n<N) array
                heights = qr.height[mask]    # (n<N) array
                for position, width, height in zip(positions, widths, heights):
                    rl.DrawRectangle(int(position[0].item()), int(position[1].item()), int(width), int(height), color)

def main(args: Namespace):
    rl.InitWindow(800, 800, b"Entity Component Style + SoA (batched)")
    scene_size = (600, 600)

    render_system = RenderSystem()
    active_kind = ShapeKind.CIRCLE

    world = World(components=[HasShape, HasPosition])
    for _ in range(args.n_objects):
        if random.random() < 0.5:
            radius = random.randint(5, 20)
            pos_y = random.randint(radius, scene_size[0] - radius)
            pos_x = random.randint(radius, scene_size[1] - radius)
            world.add_entity(components=(HasShape, HasPosition),
                             position=[pos_y, pos_x], radius=radius, kind=ShapeKind.CIRCLE)
        else: # rectangle
            width, height = random.randint(5, 20), random.randint(5, 20)
            pos_y = random.randint(height // 2, scene_size[0] - height // 2)
            pos_x = random.randint(width // 2, scene_size[1] - width // 2)
            world.add_entity(components=(HasShape, HasPosition),
                             position=[pos_y, pos_x], width=width, height=height, kind=ShapeKind.RECTANGLE)

    while not rl.WindowShouldClose():
        world.update()
        mouse_position = rl.GetMousePosition().x, rl.GetMousePosition().y

        if rl.IsKeyPressed(49): # KEY_1 (rl doesn't provide)
            active_kind = ShapeKind.CIRCLE
            logger.info("Changed active kind to: circle")
        if rl.IsKeyPressed(50): # KEY_2
            active_kind = ShapeKind.RECTANGLE
            logger.info("Changed active kind to: rectangle")

        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
            if active_kind == ShapeKind.CIRCLE:
                radius = random.randint(5, 20)
                world.add_entity(components=(HasShape, HasPosition),
                                 kind=active_kind, position=mouse_position, radius=radius)
            else: # rectangle
                width, height = random.randint(5, 20), random.randint(5, 20)
                world.add_entity(components=(HasShape, HasPosition),
                                 kind=active_kind, position=mouse_position, width=width, height=height)

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
