# Mini ECS

Minimal Entity Component System in python + raylib.

Usage:

- `pip install requirements.txt`
- Sandbox: `./main.py`
- tests: `pytest test/`

There are two classes: `World` and `Pool`. `Pool` is a simple 'archetype' dynamic array, holding entities of the same type (same set of components). `World` is a manager of pools and has an overview of all the entities in the scene. It also manages the movement of entities from one pool to the other. `Pool` operates on indices, while `World` operates on entity IDs. This allows seamless movement between pools while the high-level systems still working as intended.

