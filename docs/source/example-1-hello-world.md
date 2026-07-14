# Hello World (raylib)

[`examples/01-hello-world.py`](examples/01-hello-world.py) is the smallest complete microecs program: static balls in a box that turn **red** when they overlap, plus click-to-add-a-ball. Run it:

```bash
python examples/01-hello-world.py --n_objects 10
```

It has every moving part of a microecs app — components, systems, a `World`, and a main loop — and nothing else.

## Components (data only)

Three data-only components. Each field is a numpy array declared with `shape` + `dtype` metadata; a `default` lets you omit it at spawn time.

```python
from dataclasses import field
import numpy as np
import raylib as rl
from microecs import World, Component

class HasRadius(Component):
    radius: np.ndarray = field(metadata={"shape": (1, ), "dtype": "float32", "default": None})
class HasPosition2D(Component):
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32", "default": None})
class HasColor(Component):
    color: np.ndarray = field(metadata={"shape": (4, ), "dtype": "int32", "default": np.array(rl.BLACK, "int32")})
```

## Systems (behaviour)

A **render** system (per-entity, because raylib draws one circle at a time) and a **collision** system (fully vectorized). Both are just callables over the `World` — see [Systems & Per-Entity Iteration](systems.md) for the patterns.

```python
class RenderSystem:
    def __call__(self, world: World):
        qr = world.query(HasRadius, HasPosition2D, HasColor)
        for position, radius, color in zip(qr.position, qr.radius, qr.color):        # per-entity draw
            rl.DrawCircle(int(position[0].item()), int(position[1].item()), int(radius.item()), color.tolist())

class CollisionSystem:
    def __call__(self, world: World):
        qr = world.query(HasPosition2D, HasRadius, HasColor)
        collisions = self._get_collisions(qr.position.numpy(), qr.radius.numpy())    # (N, 1) bool
        _red   = np.array(rl.RED,   "int32")[None].repeat(len(qr), axis=0)
        _black = np.array(rl.BLACK, "int32")[None].repeat(len(qr), axis=0)
        qr.color[:] = np.where(collisions, _red, _black)                             # recolor all entities at once
```

`RenderSystem` must loop — there is no "draw all circles" call — so it `zip`s the fields. `CollisionSystem` never loops over entities: the overlap check is one broadcasted numpy expression (`(N,1,2) - (1,N,2) → (N,N)` pairwise distances) and the recolor is a single `np.where` written back through `qr.color[:]`.

## World + main loop

Build the `World` with its component set, spawn some entities, then loop:

```python
world = World(components=[HasRadius, HasPosition2D, HasColor])
for _ in range(args.n_objects):
    world.add_entity(components=(HasRadius, HasPosition2D, HasColor),
                     position=np.array(position, "float32"), radius=np.array([radius], "float32"))

render_system, update_systems = RenderSystem(), [CollisionSystem()]
while not rl.WindowShouldClose():
    world.update()                                          # 1. flush last tick's spawns/despawns
    if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
        world.add_entity(...)                               # 2. lazy -> the new ball appears next tick
    _ = [system(world=world) for system in update_systems]  # 3. run systems (collision recolor)

    rl.BeginDrawing()
    rl.ClearBackground(rl.RAYWHITE)
    render_system(world=world)                              # 4. draw
    rl.EndDrawing()
```

The order that matters: **`world.update()` first**, so entities added last tick (or on click) are committed before the systems and the renderer read them. `add_entity` is lazy — a ball clicked this frame appears next frame. That deferral is exactly what keeps a query stable while a system runs over it (see [Primitives — Mutation timing](primitives.md)).

## Next

- [Moving & Colliding Balls](example-2-moving-colliding-balls.md) — adds velocity, wall bounce, and a fixed-`dt` physics clock.
- [Serialization (save & load)](example-3-serialization.md) — save/load the whole world to JSON via `to_dict`.
