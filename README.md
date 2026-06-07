# MicroECS

Minimal (~300 LoC) Entity Component System in python and numpy. Examples also use raylib for rendering.

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

## Relevant primitives: `Component`, `Pool`, `QueryResult`, `World`

There are only 4 primitives (bottom up):

- `Component` is a simple python dataclass holding only data. All entries must be numpy arrays with metadata fields: shape and dtype. We support 5 dtypes only: `int32`, `float32`, `bool`, `str` and `object`. A component with no fields is a valid **tag** for querying (e.g. `class Frozen(Component): pass`).
- `Pool` is a simple 'archetype' dynamic array, holding entities of the same type (same set of components). Usses `Components` metadata to construct contiguous arrays for all entities of the same type.
- `QueryResult` is a list of pools that match some query on all the entities of the `World`. It acts as a contiguous numpy-like container that implements numpy's interface. For all intents and purposes it should feel like a `(N, ...)` view over all selected entities. To get a proper numpy array, use `qr.numpy()`. To iterate over each entity in a query result (e.g. rendering), use `for eid, position in zip(qr.entity_ids, qr.position): ...`.
- `World` is a manager of `Pools` and has an overview of all the entities in the scene. It also manages the migration of entities from one pool to the other. A `World` can also require extra metadata keys on every field via `World(extra_metadata=["serializable"])`, to enforce component-level behavior such as field serialization.

### Few relevant concepts:

- `Pool` operates on array indices, while `World` operates on entity IDs (also integers). This allows seamless movement between pools while the high-level systems still working as intended.
- All mutable operations on `World` are lazy. These are: `add_entity`, `remove_entity`, `add_component`, `remove_component`. They are added to a command buffer which is only executed when calling `world.update()`.
- `Systems` are a convention, they are not part of this library. They can be defined at application level and act as hooks or callbacks. The `World` object doesn't need to know more than entities and components.

## Super simplified main loop structure

```python
from typing import Callable
import numpy as np
import raylib as rl
from microecs import World, Component

# components
class HasPosition(Component):
    # 'shape' + 'dtype' are always required. For additional metadata (e.g. examples/03-serialization) use extra_metadata
    position: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})
class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2, ), "dtype": "float32"})
class HasColor(Component):
    color: np.ndarray = field(metadata={"shape": (4, ), "dtype": "int32"})

# systems: Note they are a convention!
class RenderSystem:
    def __call__(self, world: World):
        query_result = world.query(HasPosition, HasColor, exclude=[]) # contiguous-like view of all entities matching
        for position, color in zip(query_result.position, query_result.color): # draw each entity
            DrawEntity(position, color)

class MotionSystem:
    def __call__(self, world: World):
        qr = world.query(HasPosition, HasVelocity) # 'exclude' is optional
        qr.position[:] = qr.position + qr.velocity * DT # writes back to all the underlying pools using numpy's rules
        # Alternative for per-pool update. Less ergonomic, but maybe faster in extreme cases as it avoids the _Field obj
        for pool in qr.pool_list:
            pool.position[:] = pool.position + pool.velocity * DT

def main():
    render_system: list[Callable] = RenderSystem()
    update_systems: list[Callable] = [MotionSystem()]

    world = World(components=[HasPosition, HasColor, HasVelocity], extra_metadata=None) # extra_metadata is optional
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

## Per-entity systems (e.g. rendering, foreign APIs)

Not every system vectorizes. A renderer calls a draw function per primitive; same for any per-entity foreign API. That's a per-entity system inside an otherwise-vectorized app. The cheapest way to iterate through all the entities is by using `zip()` on the `QueryResult` object on the fields you need, e.g:

```python
qr = world.query(HasPosition, HasColor, HasRadius)
for pos, color, radius in zip(qr.position, qr.color, qr.radius):     # one entity per step, fields aligned
    rl.DrawCircle(int(pos[0]), int(pos[1]), float(radius[0]), color) # per-entity by necessity (no "draw all")
```

There are other ways to extract a single entity e.g. `qr.some_field[i]` or `e = world.get_entity(eid)`, but these are slower in the general case and should be avoided in expensive loops, like rendering, even if they are a bit more ergonomic. See benchmark below.

<details>
<summary> Microbenchmark: ECS vs OOP on a simple physics step </summary>

Benchmark: we run the same physics step `pos += vel*dt` over N=100k entities split across 2 pools in various ways (ECS or OOP). All methods are verified to produce the identical result. Reproduce with `python test/manual/perf/bench_ecs_vs_oop.py` (it prints `{mode: avg_seconds_per_step}`).

| pattern | ns/entity | vs OOP-scalar |
|---|---:|---:|
| `micro-ecs-pool-vectorized` — `for pool: pool.f[:] = pool.f + …` | 0.9 | **52× faster** |
| `micro-ecs-vectorized` — `qr.f[:] = qr.f + …` (the `_Field`) | 1.8 | **27× faster** |
| **`oop-scalar`** — `for o: o.x += o.vx*dt` (python floats) | 48 | 1× (baseline) |
| `oop-numpy` — objects holding `(2,)` numpy arrays | 605 | 13× slower |
| `micro-ecs-zip-rows` — `for p, v in zip(qr.pos, qr.vel)` | 744 | 15× slower |
| `micro-ecs-pool-loop` — `for pool: for i: pool.f[i]` | 870 | 18× slower |
| `micro-ecs-get-entity` — `world.get_entity(eid)` per entity | 1674 | 35× slower |
| `micro-ecs-index` — `qr.f[i]` per entity (an `np.searchsorted` each) | 4573 | 95× slower |

Three things to take from it:

1. **Vectorized wins big.** Batched ops (`_Field` or per-pool) run at 1–2 ns/entity — **27–52×
   faster** than the *fastest* OOP loop. Same for data-parallel branches: an `np.where` clamp or
   bounce is ~34× faster than a per-entity `if`.
2. **Per-entity loops are a cliff, not a tie.** Every per-entity microecs path is **15–95× slower**
   than idiomatic float-based OOP — because microecs is numpy-backed, so a per-entity step pays
   numpy's tiny-array overhead (`oop-numpy` shows the same ~13× tax). One unavoidable per-entity
   pass (~750 ns/entity) costs ~500× a vectorized op (~1.5 ns) and will dominate the frame.
3. **If you must loop, loop right.** `zip`-rows (15×) < pool-loop (18×) < `get_entity` (35×) <
   `qr.f[i]` (95× — never in a hot loop; `qr.f[i] += …` also raises, you must bind the row first).

**Rule of thumb:** keep systems vectorized and push branches into `np.where` / `np.clip`. If a
workload is *irreducibly* per-entity (data-dependent control flow), plain python objects beat
microecs ~15× — use them there. microecs is the right tool for **vectorizable** simulation.

</details>

## How much are `Pool` and `QueryResult` numpy-like and corner cases

Given `qr=world.query(A, B)`, then `qr.position` returns a `_Field`: a view over the matching pools that behaves like one contiguous-like `(N, *e)`
numpy array (e.g. `(N, 2)` for a `(2,)` field). It applies each op **per pool** and stitches result back.

That covers elementwise math and ufuncs (e.g. `np.where`, `np.linalg.norm(..., axis=1)` etc.), broadcasting (every operand shape numpy accepts, and it raises on the ones numpy rejects). See `test/unit/test_field_numpy_parity.py` for a whole set of operations comparing both.

Edge cases worth knowing:

- **Not a full ndarray — these raise, never lie.** Entity-axis indexing beyond a single
  `qr.f[i]` (`qr.f[:]`, `qr.f[2:4]`, `qr.f[mask]`, fancy), partial entity writes, and ndarray
  methods/attrs (`.sum()`, `.mean()`, `.dtype`, `.ndim`, `.T`). Need any of these? Materialize
  first with `qr.f.numpy()`.
- **Axis-0 ops are per-pool, not global (footgun).** `np.sort` / `np.cumsum` / `np.sum` over
  `axis=0` run within each pool and reset at pool boundaries — they do **not** see all entities
  at once, so they differ from numpy. They're allowed, but if you want a global result, do
  `qr.f.numpy()` first. A reduction that collapses the entity axis is rejected when its length no
  longer matches the pool's row count.
- **Operands must come from the same query.** Alignment is per-pool, not by flat index, so don't
  mix a `_Field` from one `world.query(...)` into an op on another.
