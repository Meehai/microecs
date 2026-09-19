# 4. Conditions with Masks

[`examples/04-conditions-with-masks.py`](examples/04-conditions-with-masks.py) selects a **subset** of the entities that share one component, by a field value: SQL's `WHERE` on top of the ECS `SELECT`. Circles and rectangles are both `HasShape`; a boolean mask picks one kind at a time. Run it:

```bash
python examples/04-conditions-with-masks.py --n_objects 10
```

Press **1** for circles and **2** for rectangles; a left-click spawns the active kind.

## One component, several kinds

The ECS-idiomatic design is one component per kind (`HasCircle`, `HasRectangle`) and one query per kind. That is not always what you want: when the kinds share their fields and differ only in a value, you would rather query them together once and split by value — exactly what SQL's `WHERE` does. This example takes the second road.

```python
class ShapeKind(IntEnum):
    CIRCLE = 0
    RECTANGLE = 1

class HasShape(Component):
    kind:   np.ndarray = obj((1, ))                 # the ShapeKind value
    radius: np.ndarray = f32((1, ), default=zeros)  # circle-only fields, zero when unused
    width:  np.ndarray = f32((1, ), default=zeros)
    height: np.ndarray = f32((1, ), default=zeros)
```

`zeros` is the builders' sentinel for "the zero value of this shape and dtype", so the fields a kind does not use are still there, at zero. `obj` is the object dtype — see [Declaring fields](primitives.md) for the builders.

## The mask

A `QueryResult` field compares like a numpy array: `qr.kind == kind` produces an `(N, 1)` bool array, and `[..., 0]` squeezes it to the `(N, )` mask. Indexing a field with the mask **gathers** the matching rows — a plain `(n, ...)` array:

```python
qr = world.query(HasShape, HasPosition)

for kind, color in ((ShapeKind.CIRCLE, rl.RED), (ShapeKind.RECTANGLE, rl.GREEN)):
    mask = (qr.kind == kind)[..., 0]     # (N, ) bool
    positions = qr.position[mask]        # (n, 2) gather
    radii = qr.radius[mask]              # (n, 1) gather
```

The same mask **scatters** on assignment (`qr.radius[mask] = 0` touches only the selected rows). Gather and scatter are the two halves of `WHERE`; [Primitives](primitives.md) has the mask rules, including why the trailing field axis is squeezed first.

## Systems and the main loop

`RenderSystem` queries every shape once, then masks per kind and draws it — raylib draws one shape at a time, so the mask replaces the branch, not the loop:

```python
class RenderSystem:
    def __call__(self, world: World):
        qr = world.query(HasShape, HasPosition)
        kind_to_color = {ShapeKind.CIRCLE: rl.RED, ShapeKind.RECTANGLE: rl.GREEN}
        for kind, color in kind_to_color.items():
            mask = (qr.kind == kind)[..., 0]
            positions = qr.position[mask]
            if kind == ShapeKind.CIRCLE:
                for position, radius in zip(positions, qr.radius[mask]):
                    rl.DrawCircle(int(position[0].item()), int(position[1].item()), int(radius.item()), color)
            else:
                for position, width, height in zip(positions, qr.width[mask], qr.height[mask]):
                    rl.DrawRectangle(int(position[0].item()), int(position[1].item()),
                                     int(width), int(height), color)
```

The main loop is [Hello World](example-1-hello-world.md)'s: `world.update()` first, then the keys and the click, then the render. Spawns stay lazy — the shape clicked this frame is committed by the next `update()`:

```python
if rl.IsKeyPressed(49): active_kind = ShapeKind.CIRCLE     # KEY_1
if rl.IsKeyPressed(50): active_kind = ShapeKind.RECTANGLE  # KEY_2
if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
    if active_kind == ShapeKind.CIRCLE:
        world.add_entity(components=(HasShape, HasPosition), kind=active_kind,
                         position=mouse_position, radius=radius)
    else:
        world.add_entity(components=(HasShape, HasPosition), kind=active_kind,
                         position=mouse_position, width=width, height=height)
```

## Full code
```python
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
```

## See also

- [Systems & Per-Entity Iteration](systems.md) — when a system is batched, when it must loop, and how both fit one frame.
- [Primitives](primitives.md) — `obj` fields, the `zeros` sentinel, mask gather/scatter, and `QueryResult` lifetimes.
- [Moving & Colliding Balls](example-2-motion-and-collision.md) — the per-entity render loop this example reuses.
