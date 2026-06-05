# MicroECS

Minimal (~300 LoC) Entity Component System in python and numpy. Examples also use raylib for rendering.

Usage:

- `pip install -r requirements.txt`
- Sandbox: `python main.py`
- Tests: `pytest test/`

There are only 4 primitives (bottom up): `Component`, `Pool`, `QueryResult`, `World`:

- `Component` is a simple python dataclass holding only data. All entries must be numpy arrays with metadata fields: shape and dtype. We support 5 dtypes only: `int32`, `float32`, `bool`, `str` and `object`. A component with no fields is a valid **tag** for querying (e.g. `class Frozen(Component): pass`).
- `Pool` is a simple 'archetype' dynamic array, holding entities of the same type (same set of components). Usses `Components` metadata to construct contiguous arrays for all entities of the same type.
- `QueryResult` is a list of pools that match some query on all the entities of the `World`. It acts as a contiguous numpy-like container that implements numpy's `__array_function__` and `__array_ufunc__`. For all intents and purposes it should feel like a `(N, ...)` view over all the entities. If you need a numpy array (not all ops are supported, for e.g. indexing on the first axis), use `QueryResult.numpy()`. It also exposes `entity_ids`: a flat `(N,)` array of the matched entities' ids, in the same pool-by-pool order as the fields, so you can `zip(qr.entity_ids, qr.position)` or feed an id back to `world.get_entity` / `world.remove_entity`.
- `World` is a manager of `Pools` and has an overview of all the entities in the scene. It also manages the migration of entities from one pool to the other. A `World` can also require extra metadata keys on every field via `World(extra_field_metadata=["serializable"])`, to enforce component-level behavior such as field serialization.

Few relevant concepts:

- `Pool` operates on array indices, while `World` operates on entity IDs (also integers). This allows seamless movement between pools while the high-level systems still working as intended.
- All mutable operations on `World` are lazy. These are: `add_entity`, `remove_entity`, `add_component`, `remove_component`. They are added to a command buffer which is only executed when calling `world.update()`.
- `Systems` are a convention, they are not part of this library. They can be defined at application level and act as hooks or callbacks. The `World` object doesn't need to know more than entities and components.

Super simplified main loop structure:

```python
from typing import Callable
import numpy as np
import raylib as rl
from microecs import World, Component

# components
class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})
class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})
class HasColor(Component):
    color: np.ndarray = field(metadata={"shape": (4, ), "dtype": "int32"})

# systems: Note they are a convention!
class RenderSystem:
    def __call__(self, world: World): # must override
        query_result = world.query_and((HasPosition, HasColor)) # contiguous-like view of all entities matching
        for position, color in zip(query_result.position, query_result.color): # draw each entity
            DrawEntity(position, color)

class MotionSystem:
    def __call__(self, world: World): # must override
        qr = world.query_and((HasPosition, HasVelocity))
        qr.position[:] = qr.position + qr.velocity * DT # writes back to all the underlying pools using numpy's rules
        # Alternative for per-pool update. Less ergonomic, but maybe faster in extreme cases as it avoids the _Field obj
        for pool in qr.pool_list:
            pool.position[:] = pool.position + pool.velocity * DT

def main():
    render_system: list[Callable] = RenderSystem()
    update_systems: list[Callable] = [MotionSystem()]

    world = World(components=[HasPosition, HasColor, HasVelocity], extra_field_metadata=None)
    for _ in range(n_objects):
        # NOTE: world.{add/remove}_{entity/component} are lazy. They take effect after the first world.update() call.
        world.add_entity(components=(HasPosition, HasVelocity, HasColor), # tuple of components (types)
                         position=           np.array((x, y), "float32"), # data as kwargs
                         color=         np.array("black", dtype="int32"),
                         velocity=         np.array((vx, vy), "float32"))

    while not rl.WindowShouldClose():
        world.update() # must be called at each tick so the lazy methods are processed and entities are updated
        # update stuff...
        _ = [system(world=world) for system in update_systems]
        # draw stuff, e.g. using raylib
        rl.BeginDrawing()
        rl.ClearBackground(rl.RAYWHITE)
        rl.DrawFPS(rl.GetScreenWidth() - 100, 0)
        render_system(world=world)
        rl.EndDrawing()
```
