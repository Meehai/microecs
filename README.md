# MicroECS

Minimal (~500 LoC) Entity Component System in python and numpy. Examples also use raylib for rendering.

Usage:

- Via pip: `pip install microecs`
- From source code:
```bash
git clone https://gitlab.com/meehai/microecs      # clone the source code
cd microecs                                       # go in the cloned directory
python -m venv .venv && source .venv/bin/activate # make a virtual env, optional but useful
python -m pip install -e .                        # install micro ecs in this virtual env
python -m pytest test/                            # run the unit & integration tests to verify installation
python examples/01-hello-world.py                 # run the basic hello world example (others in that dir)
```

Docs: [meehai.gitlab.io/microecs](https://meehai.gitlab.io/microecs/)

## Simple example

```python
from dataclasses import field
import numpy as np
from microecs import World, Component

class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32", "default": np.float32([0, 0])})
class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32", "default": np.float32([0, 0])})

world = World(components=[HasPosition, HasVelocity])
# both velocity and position (data) are optional since they have a default
eid1 = world.add_entity(components=[HasPosition, HasVelocity])
# data is passed as kwargs to add_entity
eid2 = world.add_entity(components=[HasPosition, HasVelocity],
                        velocity=np.float32([1, 1]))
world.update() # add_entity uses a command buffer internally until this is called
print(f"Added 2 entities. Id1={eid1}, Id2={eid2}")

# Querying: batch operate on all entities at once.
qr = world.query(HasVelocity) # qr is a QueryResult object, a numpy-based Structure of Arrays (SoA).
qr.velocity += np.float32([0.1, 0.5])
```

## Documentation

- [Primitives](docs/source/primitives.md) — the five building blocks (`Component`, `Entity`, `Pool`, `QueryResult`, `World`), mutation timing, and how numpy-like the query views really are.
- [Systems & Per-Entity Iteration](docs/source/systems.md) — writing systems, the three ways to touch data (vectorized, `zip`-rows, the `Entity` API), and when each is right.
- [Hello World (raylib)](docs/source/example-1-hello-world.md) — a complete runnable program, walked through part by part.
- [Benchmarks](docs/source/benchmarks.md) — microecs vs OOP, and microecs vs other Python ECS libraries.
