# Mini ECS

Minimal Entity Component System in python, numpy and raylib.

Usage:

- `pip install requirements.txt`
- Sandbox: `python main.py`
- Tests: `pytest test/`

There are only three primitives: `Pool`, `World` and `System`.

- `Pool` is a simple 'archetype' dynamic array, holding entities of the same type (same set of components).
- `World` is a manager of `Pools` and has an overview of all the entities in the scene. It also manages the migration of entities from one pool to the other.
- `System` is an abstract class that queries the `World` for a subset of `Pools` matching some components. It updates the entities in these pools given some logic (e.g. collisions, motion physics or simply calls the drawing functions).

Note: `Pool` operates on array indices, while `World` operates on entity IDs. This allows seamless movement between pools while the high-level systems still working as intended.

Super simplified main loop structure:

```python
from ecs import World, Component, TickSystem

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
