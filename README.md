# MicroECS

Minimal (~400 LoC) Entity Component System in python and numpy. Examples also use raylib for rendering.

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

These are the main primitives:

- `Component` is a simple python dataclass holding only data. All entries must be numpy arrays with metadata fields: shape and dtype. We support 4 dtypes only: `int32`, `float32`, `bool` and `object`. Python strings (and any other non-numeric data) go in `dtype=object` — numpy's fixed-width strings truncate in a pre-allocated pool, so they are not supported. A component with no fields is a valid **tag** for querying (e.g. `class Frozen(Component): pass`).
- `Entity` is an `OOP-like` view inside the arrays of components. The data is column-major, so this approach is the slowest (row-major), but is sometimes needed when iterating through all the objects of some type (e.g. rendering or serialization).
- `Pool` is a simple 'archetype' dynamic array, holding entities of the same type (same set of components). Uses `Components` metadata to construct contiguous arrays for all entities of the same type. All fields of all entities of the same archetype are stored in column-major numpy arrays.
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
        # slower variant, but feels more OOP
        for entity in (world.get_entity(eid) for eid in query_result.entity_ids):
            DrawEntity(entity.position, entity.color)

class MotionSystem:
    def __call__(self, world: World):
        qr = world.query(HasPosition, HasVelocity) # 'exclude' is optional
        qr.position[:] = qr.position + qr.velocity * DT # writes back to all the underlying pools using numpy's rules
        # Alternative for per-pool update. Less ergonomic, but maybe faster in extreme cases as it avoids the Field obj
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

For random single-entity access, `e = world.get_entity(eid)` (or `world.get_entity(qr.entity_ids[i])` to go by query position). That's slower in the general case and should be avoided in expensive loops, like rendering, even if it is a bit more ergonomic. See benchmark below. (There is no `qr.field[i]` shortcut — the entity axis is off-limits on a query result; it raises.)

<details>
<summary> Microbenchmark: ECS vs OOP on a simple physics step </summary>

Benchmark: we run the same physics step `pos += vel*dt` over N=100k entities split across 2 pools in various ways (ECS or OOP). All methods are verified to produce the identical result. Reproduce with `python examples/04-benchmark-ecs-vs-oop.py` (it prints `{mode: avg_seconds_per_step}`).

| pattern | ns/entity | vs OOP-scalar |
|---|---:|---:|
| `micro-ecs-pool-vectorized` — `for pool: pool.f[:] = pool.f + …` | 0.9 | **52× faster** |
| `micro-ecs-vectorized` — `qr.f[:] = qr.f + …` (the `Field`) | 1.8 | **27× faster** |
| **`oop-scalar`** — `for o: o.x += o.vx*dt` (python floats) | 48 | 1× (baseline) |
| `oop-numpy` — objects holding `(2,)` numpy arrays | 605 | 13× slower |
| `micro-ecs-zip-rows` — `for p, v in zip(qr.pos, qr.vel)` | 744 | 15× slower |
| `micro-ecs-pool-loop` — `for pool: for i: pool.f[i]` | 870 | 18× slower |
| `micro-ecs-get-entity` — `world.get_entity(eid)` per entity | 1450 | 30× slower |

Three things to take from it:

1. **Vectorized wins big.** Batched ops (`Field` or per-pool) run at 1–2 ns/entity — **27–52×
   faster** than the *fastest* OOP loop. Same for data-parallel branches: an `np.where` clamp or
   bounce is ~34× faster than a per-entity `if`.
2. **Per-entity loops are a cliff, not a tie.** Every per-entity microecs path is **15–30× slower**
   than idiomatic float-based OOP — because microecs is numpy-backed, so a per-entity step pays
   numpy's tiny-array overhead (`oop-numpy` shows the same ~13× tax). One unavoidable per-entity
   pass (~750 ns/entity) costs ~500× a vectorized op (~1.5 ns) and will dominate the frame.
3. **If you must loop, loop right.** `zip`-rows (15×) < pool-loop (18×) < `get_entity` (30×).
   For random single-entity access use `world.get_entity(qr.entity_ids[i])` — there's no `qr.f[i]`
   shortcut (the entity axis is off-limits on a query; it raises).

**Rule of thumb:** keep systems vectorized and push branches into `np.where` / `np.clip`. If a
workload is *irreducibly* per-entity (data-dependent control flow), plain python objects beat
microecs ~15× — use them there. microecs is the right tool for **vectorizable** simulation.

</details>

## Benchmark: microecs vs other Python ECS libraries

The same batched physics step (`vel += acc*dt` then `pos += vel*dt`) over N=100k entities, run
across the most popular Python ECS libraries. One script per library, every result verified
against a float64 numpy reference. Full setup, fairness notes, and analysis in
`test/manual/benchmark-vs-similar-libs/`.

| library | model | step / frame | ns/entity | vs slowest | build (100k) |
|---|---|---:|---:|---:|---:|
| **microecs** | numpy struct-of-arrays | **0.16 ms** | **1.6** | **189×** | 0.85 s |
| xecs | Rust struct-of-arrays | 0.55 ms | 5.5 | 56× | 14 ms |
| esper | pure-python objects | 9.33 ms | 93.3 | 3.3× | 186 ms |
| ecs-pattern | pure-python objects | 10.74 ms | 107.4 | 2.8× | 112 ms |
| snecs | pure-python sparse-set | 30.38 ms | 303.8 | 1.0× | 262 ms |

The batch update — microecs's whole purpose — runs **19–189× faster** than the per-entity python
ECSs, and ~3.4× faster than the only other vectorized library (xecs). microecs's heaviest phase is
the opposite end — **building** the scene (entities created one at a time) is ~4× the pure-python
libs and well above xecs's bulk Rust spawn, so it pays off for long-lived scenes and is the wrong
tool for spawn-heavy churn. If your update loop *isn't* vectorizable, those per-entity libraries are
simpler and faster — see the microbenchmark above.

## How much are `Pool` and `QueryResult` numpy-like and corner cases

Given `qr=world.query(A, B)`, then `qr.position` returns a `Field`: a view over the matching pools that behaves like one contiguous-like `(N, *e)`
numpy array (e.g. `(N, 2)` for a `(2,)` field). It applies each op **per pool** and stitches result back.

That covers elementwise math and ufuncs (e.g. `np.where`, `np.linalg.norm(..., axis=1)` etc.), broadcasting (every operand shape numpy accepts, and it raises on the ones numpy rejects). See `test/unit/test_field_numpy_parity.py` for a whole set of operations comparing both.

Edge cases worth knowing:

- **Not a full ndarray — these raise, never lie.** Entity-axis indexing of any kind
  (`qr.f[i]`, `qr.f[:]`, `qr.f[2:4]`, `qr.f[mask]`, fancy), partial entity writes, and ndarray
  methods/attrs (`.sum()`, `.mean()`, `.dtype`, `.ndim`, `.T`). Need a single entity? Use
  `world.get_entity(qr.entity_ids[i])`. Need a real array? Materialize first with `qr.f.numpy()`.
- **Axis-0 ops are per-pool, not global (footgun).** `np.sort` / `np.cumsum` / `np.sum` over
  `axis=0` run within each pool and reset at pool boundaries — they do **not** see all entities
  at once, so they differ from numpy. They're allowed, but if you want a global result, do
  `qr.f.numpy()` first. A reduction that collapses the entity axis is rejected when its length no
  longer matches the pool's row count.
- **Operands must come from the same query.** Alignment is per-pool, not by flat index, so don't
  mix a `Field` from one `world.query(...)` into an op on another.
- **Reserved field names.** A field is read back as `qr.<field>` and `entity.<field>`, so it may not
  collide with an attribute or method of `QueryResult` or `Entity` — `World(...)` rejects such a
  component at construction instead of silently shadowing it. The reserved set (source of truth:
  `QUERY_RESULT_INTERNAL_ATTRS` in `query_result.py`, `ENTITY_INTERNAL_ATTRS` in `entity.py`):
  - from `QueryResult`: `pool_list`, `entity_ids`, `fields`, `_data`, `_len`, `_field_shapes`, `_field_dtypes`
  - from `Entity`: `entity_id`, `get_components`, `get_fields`, `to_dict`, `_eid_to_pool_ix`, `_pool_to_components`

## Mutation timing: field writes are eager, structural changes are deferred

One frame holds two different timings. Know which is which:

- **Structural changes are lazy (command-buffered).** `add_entity`, `remove_entity`, `add_component`,
  `remove_component` only queue a command; they take effect at the next `world.update()`. This is what
  keeps queries stable within a tick — pools don't move under a running system.
- **Field writes are eager.** A write through an `Entity` (`e.position = ...`, `e.position += ...`,
  `e.position[:] = ...`) and the vectorized `qr.field[:] = ...` path both write straight into
  the pool buffer and are visible immediately — no `update()` needed.

So inside one tick: a freshly spawned entity is **not** visible until `update()`, but a field write on an
already-committed entity **is** visible at once. Rule of thumb: **structure is deferred, data is live.**
If a field write must be ordered against a spawn/despawn, do the structural change, call `update()`, then
write.
