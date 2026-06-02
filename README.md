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
