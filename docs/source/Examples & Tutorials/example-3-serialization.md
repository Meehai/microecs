# 3. Serialization (save & load)

[`examples/03-serialization.py`](examples/03-serialization.py) saves the whole world to JSON and loads it back. Run it, then press **F5** to write `state.json` and **F6** to reload it:

```bash
python examples/03-serialization.py --n_objects 10
```

microecs has **no** save/load feature. It gives you one primitive — `entity.to_dict()` — and this page builds a full save/load on top of it, one step at a time. Nothing below is skipped; by the last step you have the whole thing.

The world we save is [Moving & Colliding Balls](example-2-motion-and-collision.md) with two extra twists, and both are the interesting part: some balls **move and some are static** (different archetypes), and one field is **derived** (recomputed every frame, so saving it would be a lie).

## Step 1 — say which fields are worth saving

A `World` can require an extra metadata key on **every** field. Ask for `serializable`, and each field must then declare it — the builders pass any such kwarg straight into the field's metadata:

```python
import numpy as np
from microecs import World, Component, f32, i32, zeros

class HasRadius(Component):
    radius: np.ndarray = f32((1, ), serializable=True)
class HasPosition2D(Component):
    position: np.ndarray = f32((2, ), serializable=True)
class HasMotion2D(Component):
    velocity: np.ndarray = f32((2, ), serializable=True)
    # magnitude = |velocity|, recomputed every frame. Not a source of truth -> not serializable.
    magnitude: np.ndarray = f32((1, ), serializable=False, default=zeros)
class HasCustom(Component):          # a tag: no fields, it just marks an entity
    pass

world = World(components=[HasRadius, HasPosition2D, HasMotion2D, HasCustom], extra_metadata=["serializable"])
```

Only `shape`/`dtype`/`default` are the world's own keys — `serializable` (or any other kwarg) is just extra
metadata, and the sets must match exactly: a field carrying `serializable` used in a `World` that did not
ask for it raises at construction, and so does a `World(extra_metadata=["serializable"])` with a field that
omits it. See [Declaring fields](primitives.md) for both spellings.
```

Forget the key on one field and the `World` refuses to build — that is the point of `extra_metadata`: the check happens once, at startup, not when you try to save.

Note **where** the flag lives: on the *field*, not the component. `HasMotion2D` carries both a source of truth (`velocity`) and a derived value (`magnitude`, rebuilt each frame by `qr.magnitude = np.linalg.norm(qr.velocity, axis=1, keepdims=True)`). A per-component flag could not split them.

## Step 2 — one entity to a dict

`entity.to_dict()` converts a single row to plain python (`.tolist()` per field). Pass the flag name and it emits only the fields where that flag is `True`:

```python
e = world.get_entity(eid)

e.to_dict()
# {'components': ['HasRadius', 'HasPosition2D', 'HasMotion2D', 'HasCustom'],
#  'data': {'radius': [5.0], 'position': [50.0, 50.0], 'velocity': [20.0, -30.0],
#           'magnitude': [36.055511474609375]}}

e.to_dict(serialization_field="serializable")
# {'components': ['HasRadius', 'HasPosition2D', 'HasMotion2D', 'HasCustom'],
#  'data': {'radius': [5.0], 'position': [50.0, 50.0], 'velocity': [20.0, -30.0]}}     # no magnitude
```

Two things to read off that output. It reports its **`components`** as well as its `data` — that is what will let us put the entity back in the right pool. And a *static* ball, being a different archetype, dumps a shorter dict on its own:

```python
world.get_entity(static_eid).to_dict(serialization_field="serializable")
# {'components': ['HasRadius', 'HasPosition2D'], 'data': {'radius': [10.0], 'position': [100.0, 100.0]}}
```

Serialization is the one place where per-entity iteration is the *right* shape: JSON is row-major, the pools are column-major, and there is no vectorized way to cross that. See [Systems — per-entity iteration](systems.md).

## Step 3 — the world to a dict

Loop over `world.live_entities` and collect the rows. Add a small header so the loader can rebuild the same `World`:

```python
def world_to_dict(world: World) -> dict:
    return {"components": world.component_names,          # which components this world accepts
            "extra_metadata": world.extra_metadata,        # ["serializable"] -- so the reload asks for it too
            "entities": [world.get_entity(eid).to_dict(serialization_field="serializable")
                         for eid in world.live_entities]}
```

The header is two lines and it is what makes the file self-describing: `World(...)` needs its component set *before* the first `add_entity`, and only the save file knows it.

```json
{
  "components": ["HasRadius", "HasPosition2D", "HasMotion2D", "HasCustom"],
  "extra_metadata": ["serializable"],
  "entities": [
    {"components": ["HasRadius", "HasPosition2D"],
     "data": {"radius": [10.0], "position": [100.0, 100.0]}},
    {"components": ["HasRadius", "HasPosition2D", "HasMotion2D", "HasCustom"],
     "data": {"radius": [5.0], "position": [50.0, 50.0], "velocity": [20.0, -30.0]}}
  ]
}
```

That is the whole save side. `json.dump(world_to_dict(world), fp)` and you are done.

## Step 4 — a dict back to a world

Loading is the mirror image, and it needs one helper first: a spawn function that takes app-level arguments and picks the components to match. Every microecs app has one, and it is exactly what makes variable archetypes work.

```python
def add_entity(world: World, radius: list[float], position: tuple, velocity: tuple | None = None,
               custom: bool = False):
    components = [HasRadius, HasPosition2D]
    data = {"radius": np.array(radius, "float32"), "position": np.array(position, "float32")}
    if velocity is not None:                       # a mover: one more component, two more fields
        components.append(HasMotion2D)
        data["velocity"] = np.array(velocity, "float32")
        data["magnitude"] = np.zeros((1, ), "float32")   # derived: a dummy 0, the next frame overwrites it
    if custom is True:                             # a tag: a component with no data
        components.append(HasCustom)
    world.add_entity(components=components, **data)
```

Now the loader is four lines. `data["components"]` are **names** (strings), so map them back to the classes, then replay each row through the helper:

```python
def world_from_dict(data: dict) -> World:
    components = [globals()[c] for c in data["components"]]        # "HasRadius" -> HasRadius
    world = World(components=components, extra_metadata=data["extra_metadata"])
    for entity in data["entities"]:
        add_entity(world, **entity["data"], custom="HasCustom" in entity["components"])
    return world
```

Each entity's saved `components` list is what routes it back: `velocity` present in `data` ⇒ it was a mover ⇒ `HasMotion2D` ⇒ the movers' pool; `HasCustom` in the list ⇒ the tag goes back on. Static balls, movers and tagged balls all land in the right [pool](primitives.md) without a special case anywhere.

## Step 5 — wire it into the main loop

`world.add_entity` is buffered, so the loaded world's first `world.update()` is what actually materializes those rows — which the loop does at the top anyway:

```python
pth = Path(__file__).parent / "state.json"

while not rl.WindowShouldClose():
    world.update()                                        # commits the spawns from load (and from clicks)

    if rl.IsKeyPressed(rl.KEY_F5):
        with open(pth, "w") as fp:
            json.dump(world_to_dict(world), fp, indent=4)
    if rl.IsKeyPressed(rl.KEY_F6):
        with open(pth, "r") as fp:
            world = world_from_dict(json.load(fp))        # a brand new World replaces the old one

    qr = world.query(HasMotion2D)                         # rebuild the derived field we chose not to save
    qr.magnitude = np.linalg.norm(qr.velocity, axis=1, keepdims=True)
    _ = [system(world=world) for system in update_systems]
    # ... BeginDrawing / RenderSystem / EndDrawing ...
```

Load builds a **new** `World` rather than patching the old one — that is the cheapest correct thing, since a save file may describe a different component set entirely. Any `Entity` or `QueryResult` the app was holding belongs to the old world and must be re-fetched ([lifetimes](primitives.md#lifetimes-how-long-is-a-queryresult-good-for)).

## What survives, and what does not

| | round-trips? |
|---|---|
| field values, per entity | **yes** — `float32` via `.tolist()`, exactly |
| archetypes (who has which components) | **yes** — each entity carries its own `components` list |
| `serializable=False` fields | **no, by design** — rebuilt on the first frame (`magnitude` loads as `0`) |
| entity ids | **no** — a fresh `World` numbers from 0 in file order |
| `dtype="object"` fields | only if JSON can hold them: `to_dict` calls `.item()` and hands you the raw reference |

The ids are the one to watch. Save a world holding ids `[0, 2]` and it loads back as `[0, 1]`: the data is identical, the identities are not. If the app stores entity ids anywhere else (a selection, a target, a mission), save your own stable key as a field and re-resolve it after loading.

## Full code
```python
#!/usr/bin/env python3
"""
03-serialization. Showcases how one can implement serialization of all entities on top of microecs
Usage: ./03-serialization.py [--state_path STATE.JSON] [--n_objects N]
Keybinds:
- F5 to store sthe state
- F6 to load the state
"""
from typing import Callable, Any
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json
import random
import numpy as np
import raylib as rl
from loggez import loggez_logger as logger

from microecs import World, Component, f32, i32, zeros

Point2D = tuple[float, float]
DT = 1 / 100
MAX_SUBTICKS_PER_RENDER_TICK = 3

# components

class HasRadius(Component):
    radius: np.ndarray = f32((1, ), serializable=True)

class HasColor(Component):
    color: np.ndarray = i32((4, ), serializable=True)

class HasPosition2D(Component):
    position: np.ndarray = f32((2, ), serializable=True)

class HasMotion2D(Component):
    velocity: np.ndarray = f32((2, ), serializable=True)
    # magnitude is a derived property from velocity (not a source a truth). Not serializable as it is updated each frame
    magnitude: np.ndarray = f32((1, ), serializable=False, default=zeros)

class HasCustom(Component):
    pass

# serialization

def world_to_dict(world: World) -> dict[str, Any]:
    """Serialize the world. Goes through all the entities and their components and converts the serializables to dict"""
    res = {"entities": [], "components": world.component_names, "extra_metadata": world.extra_metadata}
    for entity_id in world.live_entities.keys():
        res["entities"].append(world.get_entity(entity_id).to_dict(serialization_field="serializable"))
    return res

def add_entity(world: World, color: "rl.Color", radius: list[float], position: Point2D,
               velocity: Point2D | None = None, custom: bool = False):
    """spawns a new entity in the world given some parameters"""
    components = [HasRadius, HasColor, HasPosition2D]
    data = {"position": position, "color": color, "radius": radius}
    if velocity is not None:
        components.append(HasMotion2D)
        data |= {"velocity": velocity, "magnitude": 0} # dummy 0 at start, as it's continuously updated in the main loop
    if custom is True:
        components.append(HasCustom)
    world.add_entity(components=components, **data)

def world_from_dict(data: dict[str, Any]) -> World:
    """Creates a world from a serialized representation e.g. from world_to_dict()"""
    components = [globals()[c] for c in data["components"]]
    world = World(components=components, extra_metadata=data["extra_metadata"])
    for entity in data["entities"]:
        add_entity(world, **entity["data"], custom="HasCustom" in entity["components"])
    return world

# systems

class RenderSystem:
    def __call__(self, world: World):
        qr = world.query(HasRadius, HasPosition2D, HasColor)
        for position, radius, color in zip(qr.position, qr.radius, qr.color):
            rl.DrawCircle(int(position[0].item()), int(position[1].item()), int(radius.item()), color.tolist())

class MotionSystem:
    def __call__(self, world: World):
        qr = world.query(HasMotion2D, HasPosition2D)
        qr.position = qr.position + qr.velocity * DT # (N, 2)

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
        qr.velocity = np.where(mask_velocity, -qr.velocity, qr.velocity)

def create_init_world(n_objects: int, scene_size: tuple[int, int]) -> World:
    world = World(components=[HasRadius, HasColor, HasMotion2D, HasPosition2D, HasCustom],
                  extra_metadata=["serializable"])
    for _ in range(n_objects):
        radius = random.randint(5, 20)
        position = random.randint(radius, scene_size[0] - radius), random.randint(radius, scene_size[1] - radius)
        velocity = (100 * random.random() * 2 - 1, 100 * random.random() * 2 - 1) if random.random() < 0.3 else None
        custom = random.random() < 0.5 # just a custom attribute that's only sometimes there.
        add_entity(world, color=rl.BLACK, radius=radius, position=position, velocity=velocity, custom=custom)
    return world

def main(args: Namespace):
    rl.InitWindow(800, 800, b"Entity Component Style + SoA (batched)")
    scene_size = (600, 600)

    render_system = RenderSystem()
    update_systems: list[Callable] = [MotionSystem(), WallBounceSystem(scene_size)]

    if args.world_state is not None:
        with open(pth := Path(__file__).parent / "state.json", "r") as fp:
            world = world_from_dict(json.load(fp))
        logger.info(f"Loaded world state to '{pth}'")
    else:
        world = create_init_world(args.n_objects, scene_size)

    while not rl.WindowShouldClose():
        world.update()
        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
            radius = random.randint(5, 20)
            position = rl.GetMousePosition().x, rl.GetMousePosition().y
            velocity = (20 * random.random() * 2 - 1, 20 * random.random() * 2 - 1) if random.random() < 1 else None
            add_entity(world, color=rl.BLACK, radius=[radius], position=position, velocity=velocity)

        if rl.IsKeyPressed(rl.KEY_F5):
            with open(pth := Path(__file__).parent / "state.json", "w") as fp:
                json.dump(world_to_dict(world), fp, indent=4)
            logger.info(f"Wrote world state to '{pth}'")

        if rl.IsKeyPressed(rl.KEY_F6):
            with open(pth := Path(__file__).parent / "state.json", "r") as fp:
                world = world_from_dict(json.load(fp))
            logger.info(f"Loaded world state to '{pth}'")

        qr = world.query(HasMotion2D)
        qr.magnitude = np.linalg.norm(qr.velocity, axis=1, keepdims=True)
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
    parser.add_argument("--world_state", type=Path)
    main(parser.parse_args())
```

## See also

- [Primitives](<docs/source/primitives.md>) — `World(extra_metadata=[...])`, archetypes and pools, `QueryResult` lifetimes.
- [Systems & Per-Entity Iteration](<docs/source/systems.md>) — why `to_dict` is a per-entity loop and everything else is not.
- [Moving & Colliding Balls](example-2-motion-and-collision.md) — the world this one saves.
- [Conditions with Masks](example-4-conditions-with-masks.md) — the next example: select entities by a field value with a boolean mask (SQL `WHERE` on top of the ECS `SELECT`).
