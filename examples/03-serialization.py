#!/usr/bin/env python3
"""
03-serialization.py - Showcase how one can implement serialization of all entities on top of microecs
Usage: ./03-serialization.py [--state_path STATE.JSON] [--n_objects N]
Keybinds:
- F5 to store sthe state
- F6 to load the state
"""
from dataclasses import field
from typing import Callable, Any
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json
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
    radius: np.ndarray = field(metadata={"shape": (1, ), "dtype": "float32", "serializable": True, "default": None})

class HasColor(Component):
    color: np.ndarray = field(metadata={"shape": (4, ), "dtype": "int32", "serializable": True, "default": None})

class HasPosition2D(Component):
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32", "serializable": True, "default": None})

class HasMotion2D(Component):
    velocity: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32", "serializable": True, "default": None})
    # magnitude is a derived property from velocity (not a source a truth). Not serializable as it is updated each frame
    magnitude: np.ndarray = field(metadata={"shape": (1, ), "dtype": "float32", "serializable": False, "default": None})

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
    data = {"position": np.array(position, "float32"), "color": np.array(color, dtype="int32"),
            "radius": np.array(radius, "float32")}
    if velocity is not None:
        components.append(HasMotion2D)
        data["velocity"] = np.array(velocity, "float32")
        data["magnitude"] = np.zeros((1, ), "float32") # dummy 0 at start, as it's continuously updated in the main loop
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

def create_init_world(n_objects: int, scene_size: tuple[int, int]) -> World:
    world = World(components=[HasRadius, HasColor, HasMotion2D, HasPosition2D, HasCustom],
                  extra_metadata=["serializable"])
    for _ in range(n_objects):
        radius = random.randint(5, 20)
        position = random.randint(radius, scene_size[0] - radius), random.randint(radius, scene_size[1] - radius)
        velocity = (100 * random.random() * 2 - 1, 100 * random.random() * 2 - 1) if random.random() < 0.3 else None
        custom = random.random() < 0.5 # just a custom attribute that's only sometimes there.
        add_entity(world, color=rl.BLACK, radius=[radius], position=position, velocity=velocity, custom=custom)
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
