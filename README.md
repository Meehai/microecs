# MicroECS

Minimal (<200 LoC) Entity Component System in python and numpy. Examples also use raylib for rendering.

Usage:

- `pip install -r requirements.txt`
- Sandbox: `python main.py`
- Tests: `pytest test/`

There are only four primitives (bottom up): `Component`,`Pool`, `World` and `System`.

- `Component` is a simple python dataclass holding only data. All entries must be numpy arrays with metadata fields: shape and dtype. We support 4 dtypes only: `int32`, `float32`, `str` and `bool`.
- `Pool` is a simple 'archetype' dynamic array, holding entities of the same type (same set of components). Usses `Components` metadata to construct contiguous arrays for all entities of the same type.
- `World` is a manager of `Pools` and has an overview of all the entities in the scene. It also manages the migration of entities from one pool to the other.
- `System` is an abstract class that queries the `World` for a subset of `Pools` matching some components. It updates the entities in these pools given some logic (e.g. collisions, motion physics or simply calls the drawing functions).

Few relevant concepts:

- `Pool` operates on array indices, while `World` operates on entity IDs (also integers). This allows seamless movement between pools while the high-level systems still working as intended.
- All mutable operations on `World` are lazy. These are: `add_entity`, `remove_entity`, `add_component`, `remove_component`. They are added to a command buffer which is only executed when calling `world.update()`.

Super simplified main loop structure:

```python
import numpy as np
import raylib as rl
from microecs import World, Component, TickSystem

# components
class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})
class HasColor(Component):
    color: np.ndarray = field(metadata={"shape": (4, ), "dtype": "int32"})

# systems
class RenderSystem(TickSystem):
    def on_tick(self, world: World): # must override
        for pool in world.query_and((HasPosition, HasColor)): # get all pools of entities having both
            for position, radius, color in zip(pool.position, pool.color): # for each entity in this pool
                DrawEntity(position, color)
class CollisionSystem(TickSystem)
    def on_tick(self, world: World): # must override
        ...

def main():
    render_system = RenderSystem()
    update_systems: list[TickSystem] = [CollisionSystem()]

    world = World(components=[HasPosition, HasColor])
    for _ in range(n_objects):
        # NOTE: world.{add/remove}_{entity/component} are lazy. They take effect after the first world.update() call.
        world.add_entity(components=(HasPosition, HasColor), # tuple of components (types)
                         position=np.array((x, y), "float32"), # data as kwargs
                         color=np.array("black", dtype="int32"))

    while not rl.WindowShouldClose():
        world.update() # must be called at each tick so the lazy methods are processed and entities are updated
        # update stuff...
        _ = [system.on_tick(world=world) for system in update_systems]
        # draw stuff, e.g. using raylib
        rl.BeginDrawing()
        rl.ClearBackground(rl.RAYWHITE)
        rl.DrawFPS(rl.GetScreenWidth() - 100, 0)
        render_system.on_tick(world=world)
        rl.EndDrawing()
```
