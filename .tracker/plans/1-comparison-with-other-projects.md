# microecs vs. the vectorized-entity landscape — analysis & positioning

**Created**: 2026-06-04
**Updated**: 2026-07-26 (**re-ran Part 8 on `975097c` (v0.8.2)** — the field is now **seven** libraries
(EnTT/flecs bindings joined), restructured into three experiments, and the run answers "did `#42`/`#43`
cost anything?" with **no**. Two new mechanism findings: **churn is a validation cost, not a pop-swap
cost** (32% of a churn pair is validation, half of it done twice → new [task 44]) and a **priced `Entity` path**
(P5). Also: absolute ms drift ±10–25% run-to-run on this machine, so the
comparison method changed — normalize against the field, never compare ms across days. Landscape
re-verified; LoC claims corrected 300 → 846)
**Prev update**: 2026-07-15 (re-ran Part 8 after the `microecs #26` low-N optimization — `_QRArray`
single-pool fast path + `QRField` cache + lazy cumsum; columnar crossover vs xecs moved ~10k → ~1.5–3k)
**Prev update**: 2026-07-15 (added **Part 8 — empirical multi-workload benchmark**: 7 workloads × 5 libs ×
N-sweep, verified crossover + copy-boundary mechanism; corrected the efficiency scorecard with measured
numbers; competitor status re-verified — no material drift since June)
**Prev update**: 2026-06-05 (refocused on the vectorized + Python-interop niche; added GPU batch-ECS, JAX,
and dataframe-ABM clusters; turned "ideas" into a ranked borrow list)
[task 44]: ../todos/open/44-spawn-path-validates-twice/TASK.md

**Type**: Reference / competitive analysis
**Scope**: Assess microecs (efficient? good? nice?), then compare **only against projects in the same domain**:
must (a) interop with Python and (b) be **vectorized** (numpy / torch / jax / polars — bulk array math, not
per-entity Python loops). The per-entity Python ECS crowd is explicitly **out of scope** (pruned to one line).
Decide what ideas are worth borrowing.

**Confidence**: stars/dates are **2026-07-26** point-in-time snapshots (drift). Author-claimed benchmarks are
flagged `[claim]`. Primary sources (GitHub API, PyPI JSON, docs, papers) in Sources.

**See also**: [plan 2 — app-level audit & mutation timing](2-app-level-audit-and-mutation-timing.md) answers
Part 1's "efficient / good / nice?" from the *inside* instead of the benchmark side: a throwaway game built on
the library, 18 probes (7 findings open, 8 closed the same day), and the one structural call (eager vs deferred
**data** writes) — answered and shipped as `#42`.

---

## TL;DR (grug verdict)

microecs is a **good, correct, genuinely useful** little engine for its target — bulk numeric updates over many
entities. Its differentiator — numpy **SoA by archetype** + a **cross-archetype vectorized write-through view**
(`qr.position = qr.position + qr.velocity*dt` lands in *every* matching pool via numpy's array protocols) —
appears **genuinely unique among Python libraries**.

**Now measured (Part 8), re-run 2026-07-26 with seven libraries.** A 7-workload × 7-library × N-sweep confirms
the thesis *and* bounds it honestly: **there is no global winner — the ranking flips by workload and by N.**
microecs wins the **large-N regime broadly** (columnar and per-entity ai from N≈5k, random-access-via-scatter and
the realistic mixed frame from N≈20k) and is the *only* vectorized-numpy lib that can also churn/migrate. Only
below N≈1.5–3k does its fixed per-op overhead lose to leaner libs (xecs, ecs-pattern), and only by ~1.2–1.4×.
Three mechanism findings carry the analysis:

1. **microecs beats the Rust ECS (xecs) at scale precisely because it has no FFI boundary** — xecs does its
   arithmetic in numpy anyway and pays a Rust↔numpy *copy* per op; microecs mutates the pool arrays in place, and
   a single-archetype query is native-C numpy with no per-op object at all. The Rust core is a marshalling tax for
   CPU-vectorized work, not a win.
2. **Binding a native ECS does not help either** (new): EnTT (C++) and flecs (C) hand components back as Python
   objects, so the system body stays a Python loop *plus* a boundary crossing per access — they land **2.3–3.4×
   behind plain esper** on every arithmetic workload and 39–490× behind microecs at 100k. They win only where the
   work *is* the data-structure operation (churn at 100k, migration). In Python the axis that matters is
   **vectorized vs per-entity**, not native vs interpreted.
3. **microecs' one real loss — structural churn — is a validation cost, not a layout cost** (new). Splitting a
   spawn+despawn pair (12.6 µs): storage work (SoA insert + the archetype pop-swap everyone blames) is 43%, while
   **validation is 32% — and half of it runs twice**, because `World.add_entity` validates and then
   `CommandBuffer.append` validates the same command again. ~2 µs/spawn, 16% of a churn pair, is free to reclaim
   ([task 44](../todos/open/44-spawn-path-validates-twice/TASK.md)). The weak spot is bookkeeping, not architecture.

- **The direct peer set is nearly empty**, and both peers have a fatal gap vs microecs:
  - **xecs** — the "serious" attempt (Rust core + numpy). **Effectively dead** (3★, last push 2023-10-30, 14 open
    issues, not archived), **AND-only** queries, and **no archetypes** (per-component columns).
  - **manifoldx / manifold-gfx** — the truest architectural *twin* (pure Python+numpy, SoA-by-archetype,
    deferred commits; created April 2026, **still active — last push 2026-07-18**, 6★). But **welded to a wgpu
    renderer**, requires **Python ≥3.13**, and does **one numpy op *per archetype*** — still **no cross-archetype
    view** (re-checked 2026-07-26), which is exactly microecs's headline.
  - **Native-engine bindings are not peers, they are the control group** (new in this pass): PyEnTT (C++ EnTT)
    and pyflecs11 (C flecs) put a world-class ECS under Python but hand components back as Python objects. Part 8
    experiment 3 measures the result: **slower than plain esper**. Both bindings are also unmaintained hobby repos
    (PyEnTT: 1★, archived one day after creation; pyflecs11: 2★, dormant since 2025-08).
- **Widen the lens to "vectorized updates over many entities"** and there are vibrant *adjacent* clusters
  solving microecs's core problem differently — worth mining for ideas, not competing for users:
  - **GPU batch-ECS**: **Madrona** (Stanford) — a real archetype ECS, but C++ systems + GPU, **static**
    archetypes, batch-over-worlds. The aspirational "north star."
  - **JAX SoA + mask + vmap**: Brax, MJX, PGX, **ABMax/Foragax** — fixed-capacity struct-of-arrays + a boolean
    `active` mask for soft-delete, no dynamic archetypes. The *mirror image* of microecs's design.
  - **Polars/dataframe ABM**: **mesa-frames**, **AMBER** — entities as DataFrame rows, vectorized via Polars,
    one table *per agent type* (no cross-type op).
- **Validated by the big engines**: deferred command buffer (Unity ECB), single archetype-SoA path (Bevy is
  trying to *delete* its second/sparse-set storage), cache-matching-pools + invalidate-on-commit (flecs), and
  "export the column as a view, never copy" (Madrona's zero-copy tensors == microecs's `QueryResult`).
- **Also validated academically** (new): arXiv 2606.14919, *The Essence of Entity Component System* (Tasnim &
  Zhao, 2026-06-12) gives formal semantics for **archetype** ECS — entity creation, component composition, system
  execution, archetype migration as state transitions — and finds archetype ECS beats the alternative designs
  empirically. It says nothing about vectorization/SoA/Python, so it validates microecs's **storage model**, not
  its numpy layer.
- **Ideas evaluated** (Part 5/6): optional/OR, masked soft-delete, and group-by/reduction were **reviewed
  2026-07-15 and set aside** — each is derivable at user-code level or a bad trade for a query-first lib
  (rationale in Part 6). The **low-N `Field` overhead** ([task 26](../todos/done/26-low-n-field-overhead/TASK.md),
  robosim's 500-UAV driver) is now **shipped/done** — `_QRArray` single-pool fast path + cache + lazy cumsum,
  crossover moved ~10k→~1.5–3k; an optional **batch-over-worlds** axis (from Madrona) stays a roadmap candidate.

microecs's slot, stated precisely: **the only Python, CPU, dynamic-archetype ECS with a cross-pool vectorized
write — esper's accessibility with numpy's bulk speed, no compile step, no GPU, ~850 LoC (578 without docstrings
and comments), 2 deps.**

---

## Part 1 — Is microecs efficient / good / nice?

### Efficient — yes, for its target

| Mechanism | Where | Verdict |
|---|---|---|
| SoA storage, one numpy array per field per archetype | `pool.py:26-30` | ✅ cache-friendly, the whole point |
| Bitmask keys; match `(arch & inc) == inc and (arch & exc) == 0` (AND + NOT) | `world.py:129-131` | ✅ simple, correct |
| Query result **cached** between updates (keyed by include+exclude) | `world.py:118`, `world.py:138` | ✅ steady-state queries are O(1) |
| Deferred command buffer (no iterator invalidation) | `world.py:74-93`, `command_buffer.py` | ✅ same approach as flecs/Bevy/Unity |
| Vectorized cross-pool write | `query_result.py:52-61`, `qr_field.py:84-100` | ✅ the differentiator (no Python peer has it) |

For a motion/physics system over **tens of thousands** of same-archetype entities, the work runs in numpy and
beats per-entity Python loops by **1–2 orders of magnitude** (measured 62× vs esper, 150× vs the C++ EnTT binding
at N=100k; Part 8). The win is **N-gated**: below N≈1.5–3k the residual per-op overhead lets leaner libs win, but
only by ~1.2–1.4×, and microecs still *beats* esper/ecs-pattern on columnar there. The headline to publish is
"**microecs is fast at scale** (N≳5k) — only at small N (≲1.5k) does its fixed overhead let leaner libs win." The
SoA-per-field choice is **universally validated** — Brax (`QP`), MJX (`Data`), Madrona (component columns),
mesa-frames (Polars columns) all store one array per field. microecs is on the canonical path.

**Honest inefficiencies** (inherent to the approach, not bugs):

- **Multi-archetype queries still loop over pools in Python** (`QRField._apply_fn_on_parts`, `qr_field.py:34-46`)
  and do `np.broadcast_to` / `np.split` on writes (`qr_field.py:98-99`). Vectorization only wins *within* a pool;
  a query over many small pools degenerates toward per-pool Python overhead. **`#26` removed this cost for the
  common single-archetype case**: a one-pool query returns `_QRArray` (a thin `np.ndarray` subclass,
  `query_result.py:11-20`, chosen at `query_result.py:43-46`) so `+`/`*`/`[:]=` are native-C numpy with no per-op
  `QRField` object; `QRField` is cached per field and its `cumsum` is lazy. The per-pool-loop tax now only applies
  at ≥2 matching archetypes — and it is measurable: 1.34 ns/entity through a 2-pool `QRField` vs 0.84 writing the
  pools directly (benchmark 1), a 1.6× tax for the convenience.
- **Archetype fragmentation** — classic archetype-ECS tax: many component combos → many small pools.
- **`add/remove_component` copies the whole entity** to a new pool (`world.py:188-203`, `pool.py:58-60`). Fine
  occasionally, bad every tick — which is why the bounce task (`todos/open/1`) uses an impulse accumulator
  instead of per-tick component churn. Good instinct. Measured: w7 migrate is ~0.8–1.3× the field, never a rout.
- **The spawn path validates twice** (new, 2026-07-26): `World.add_entity` runs `_validate_components` +
  `_defaults_for` (`world.py:76-77`), then `CommandBuffer.append` runs both again on the same command
  (`command_buffer.py:74-77`). **2.0 µs/spawn redundant — 34% of `add_entity`, 20% of a full spawn** — and this is
  the dominant term in w5 churn, microecs' worst workload. [Task 44](../todos/open/44-spawn-path-validates-twice/TASK.md).
  This one is a *bug-shaped* inefficiency, not an inherent tax.
- **The `Entity` handle costs ~0.3–0.5 µs per field touch** (`entity.py:104-114`), so a per-entity loop over a
  query is a cliff (Part 8, P3/P5). Of a 320 ns read, 144 is the `_locate` guard and ~89 of *that* is
  `fields_set.issuperset([name])`; a plain `name in fields_set` is 40 ns. Cheap 15% available on every read.
- Keys are `2**i` Python ints (`world.py:36`), arbitrary-precision → **no 64-component cap** that bites C
  engines. But a cache miss is a linear scan of all pools (`world.py:129`).

### Good — yes (design quality)

Clean single-responsibility split: `Pool` (SoA dynamic array, no id concept) / `World` (ids + archetypes +
command buffer) / `QueryResult` (cross-pool view) / `Component` (dataclass) / `System` (convention). **846 lines,
578 without docstrings and comments**, 2 deps. Tests are genuinely strong: the 500-op randomized churn invariant
check (`test_world.py:1320`) and the recarray-parity suite (`test_field_numpy_parity.py`) are things most ECS
libraries lack — plus plan 2's shadow-model fuzzer (155,859 random structural ops, zero mismatches). The
`__array_ufunc__` / `__array_function__` impl matches numpy broadcast + recarray write-through semantics, with
sharp edges rejected rather than silently wrong.

**Quality nits** (non-test files — dev's call):
- `QRField.__array_function__` (`qr_field.py:79-81`) applies **any** numpy function per-pool with no whitelist, so
  functions it cannot honour (reductions, sorts) silently return per-pool nonsense — open [task 38].
- `from typing import Callable, T` (`qr_field.py:2`) imports the *private* `typing.T`. Works, fragile.
- `qr.field` is **two types** (`_QRArray` at one pool, `QRField` at 2+) — open [task 37]; the sharpest ergonomic
  edge plan 2 found from inside an app.

[task 37]: ../todos/open/37-qrarray-qrfield-one-contract/TASK.md
[task 38]: ../todos/open/38-array-function-honour-check/TASK.md

### Nice — mostly

The vectorized write idiom is the nicest thing here:

```python
qr.position = qr.position + qr.velocity * DT   # updates every matched entity, across all pools, in numpy
```

One ergonomic tax left: **component defs leak numpy internals** (`shape`/`dtype` metadata in every field) — see
the ergonomics comparison in Part 4. Query expressiveness — once the biggest gap — now does **AND + NOT**
(`exclude=`) and **tags**; `any_of`/optional are deferred (Part 5 has the mechanism if revisited).

---

## Part 2 — The landscape (vectorized + Python-interop only)

Domain filter: interops with Python **and** vectorizes (array math over many entities). Organised by how close
each is to microecs's job, not by popularity.

### Tier 1 — Direct vectorized Python ECS (true peers)

| Project | Stack | Stars / status | Storage | Vectorization | Queries | Verdict |
|---|---|---|---|---|---|---|
| **xecs** (`lukasturcani/xecs`) | Rust core + numpy, ships wheels | **3★**, last push **2023-10-30**, 14 open issues, v0.9.0 → **dead** | **Per-component** flat arrays (numpy-exposed); **no archetypes** | ✅ per-component (`transform.translation += velocity.value`); also `product_2` all-pairs | **AND only** (no exclude/optional/OR) | **PEER (stalled)** — closest intent (Bevy-style numpy ECS) but dead, weaker queries, no archetype model |
| **manifoldx / manifold-gfx** (`apiad/manifoldx`) | **Pure Python + numpy** (+ wgpu/rendercanvas), Py ≥3.13 | **6★**, created **2026-04-03**, last push **2026-07-18** → **active** | **SoA by archetype** + free-list (≈ microecs's Pool) | ✅ within an archetype; **one op *per archetype*** — **NO cross-archetype view** (re-checked 2026-07-26) | AND (component filter); no NOT/optional shown | **PEER (truest twin)** — same SoA-archetype + deferred commits, but renderer-locked + per-archetype-only + heavy deps |

**The two peers bracket microecs.** xecs went the Rust route and stalled; manifoldx went pure-numpy (microecs's
exact stack) but is a *renderer* and crucially **lacks the cross-archetype write-through** — its author's design
writeup explicitly describes "one method invocation per archetype," and its README still shows one query/one
`numpy` op per system. microecs's `QueryResult` is the thing neither has. *No JAX- or torch-backed Python ECS
library exists* (re-confirmed by targeted search 2026-07-26) — that sub-niche is empty.

### Tier 1b — Native ECS engines behind Python bindings (the control group, benchmarked in Part 8)

Not peers by design — they are per-entity engines — but they are the obvious objection ("just bind a real ECS"),
so Part 8 experiment 3 measures them. **Both bindings are unmaintained hobby wrappers**; the engines behind them
are not.

| Binding | Engine | Stars / status | Storage | Component in Python | Verdict |
|---|---|---|---|---|---|
| **PyEnTT** (`ominkk/PyEnTT`, PyPI `entt` 0.1.2) | **EnTT** (C++, sparse-set, ships in real games) | **1★**, created 2026-03-04, **archived 2026-03-05**; no Linux wheel, no sdist → build from source | native sparse-set | a Python object per component | **CONTROL** — fastest lib in the suite on churn/migration at 100k, **2.4–3× slower than esper** on arithmetic |
| **pyflecs11** (`Wesxdz/pyflecs11`, PyPI `flecs` 0.0.2) | **flecs** (C, archetype tables) | **2★**, created 2025-07-16, last push **2025-08-21**; linux wheel is cp39-only → build from sdist; **segfaults** on a re-iterated `Query` without `.reset()` and at interpreter shutdown | native archetype tables | a Python object per component | **CONTROL** — slowest lib in the suite on every workload |

**Why they lose is structural, not a binding bug:** the system body is still a Python loop, and now each component
access also crosses the language boundary. A better-engineered binding shaves constants; it cannot delete the
crossing. This is the same mechanism as xecs' `.numpy()` copy tax, one layer up.

### Tier 2 — Vectorized entity stores that aren't "ECS" (same problem, different shape — ADJACENT)

| Project | Stack | Stars / status | How entities are stored | Cross-type vectorized op? |
|---|---|---|---|---|
| **mesa-frames** (`projectmesa/mesa-frames`) | **Polars** (Arrow/Rust) | ~41★, pre-1.0, active | One **DataFrame per agent type** (rows=agents) | ❌ iterate per type; no cross-type expression |
| **AMBER** (`a11to1n3/AMBER`, arXiv 2601.16292) | Polars | research prototype, 2026 | Single Polars DataFrame, agents=rows | ❌ (doesn't address heterogeneity) |
| **Brax** (`google/brax`) | **JAX** | ~3.2k★, training-only maintained | `QP` struct, **SoA with leading batch dims** `[worlds, bodies, 3]` | via `vmap`; **fixed body count** |
| **MJX** (`google-deepmind/mujoco/mjx`) | JAX | parent ~9k★, active | `Model`/`Data` **pytrees of arrays** | via `vmap`; **shape change ⇒ recompile** |
| **PGX / Gymnax / Jumanji / JaxMARL** | JAX | ~0.6–0.9k★ each, mixed | dataclass-pytree `State` + **boolean masks** (`legal_action_mask`) | via `vmap`; fixed shapes + masks |
| **ABMax / Foragax** (arXiv 2508.16508 / 2409.06345) | JAX | research, small | **SoA + capacity + `num_active` cursor**; `active` flag soft-delete | via `vmap`; padded fixed capacity |
| **EnvPool** (`sail-sg/envpool`) | C++ + pybind11 | ~1.5k★, appears active | C++ batched envs → **numpy/torch** out | CPU threads, not vmap |

These solve microecs's *exact* core problem ("apply one vectorized update to many homogeneous entities") and
have **independently converged** on a single idiom: **fixed-capacity struct-of-arrays + boolean `active` mask
for births/deaths + `vmap`/`jit`** — the mirror image of microecs's *dynamic archetype pools + deferred commit*.
**ABMax/Foragax read almost like "an ECS pool reinvented for JAX"** (capacity, active-cursor, append-at-tail,
compact-by-sort) and are the single highest-value reference for ideas (Part 5).

### Tier 3 — GPU / data-oriented engines with a Python interface (ADJACENT; one real ECS)

| Project | What it is | Stars / status | ECS? | Relevance |
|---|---|---|---|---|
| **Madrona** (`shacklettbp/madrona`) | **GPU batch-ECS** for RL sim; thousands of worlds/GPU | ~495★, active (Stanford) | **Yes** | **PEER (aspirational)** — archetype SoA + columns-as-zero-copy-torch-tensors, but **C++ systems**, **static archetypes** (no dynamic add/remove), batch-over-worlds. The north star. |
| **NVIDIA Warp** (`NVIDIA/warp`) | Python→CUDA **kernel JIT** | ~6.7k★, v1.14 (2026-06), very active | No | ADJACENT — "write Python, get fast array math" via codegen kernels + zero-copy torch; no entity model |
| **Taichi** (`taichi-dev/taichi`) | Python DSL, **SNode SoA fields**, JIT | ~28k★, **maintenance mode** (Genesis forked → Quadrants) | No | ADJACENT (idea: layout decoupled from access) |
| Genesis / NVIDIA Newton | multi-physics sim engines | ~29k★ / LF, active | No | PRUNE (context: this is where the GPU-sim money/attention is) |

### Out of scope — the per-entity Python ECS mainstream (pruned)

Not vectorized; components are Python objects, iteration is a `for` loop. They compete on game-logic ergonomics,
not throughput, and none is adding vectorization. One line each:

- **esper** (696★, **active**, last push 2026-07-09) — the popular pure-Python ECS; per-entity loop. *The
  positioning foil:* "esper's simplicity, numpy's bulk speed." Keep as a benchmark baseline only — and note Part 8
  experiment 3: **esper beats both native-engine bindings on every arithmetic workload.** Being pure Python is not
  the thing that costs you.
- **tcod-ecs** (28★, **active**) — Pythonic sparse-set + **relationships + `IsA` prefabs**; the "if you don't
  need numpy vectorization, here's the other Python design" reference.
- **ecs-pattern** (v1.4.0, 2025) — dataclass AoS, lightly maintained.
- **snecs** (v1.2.2, **2020, dead**) — bitmask + query algebra, the design ancestor of the `& | ~` idea.
- **entitas-python** (**archived 2021**) — Entitas port, reactive groups. Dead.

---

## Part 3 — The central design fork: how the field stores & updates "many entities"

This is the data-structure heart of the analysis. Six strategies exist for "store entities with heterogeneous
component sets and update them in bulk." microecs picks the first.

| Strategy | Who uses it | Heterogeneity handled by | Cross-type vectorized op | Add/remove | Compute target |
|---|---|---|---|---|---|
| **1. Dynamic archetype pools, SoA/field** | **microecs**, manifoldx, Madrona, flecs, Bevy(table), Unity(chunks) | one pool per unique component set | **microecs: ✅ `QueryResult` write-through**; others: per-pool loop / task graph | **dynamic** | CPU numpy *(microecs)* |
| **2. Per-component flat arrays / sparse-set** | xecs, EnTT, Bevy(sparse) | no archetypes; the component is the unit | per-component views (no "type" to span) | fast add/remove | CPU (xecs: Rust) |
| **3. One wide null-padded table** | "DataFrames-as-ECS" pattern | **nullable columns** = optional comps | ✅ one table, but pays null/mask cost | add/drop a column | Polars / cuDF |
| **4. Per-type tables (one DF per class)** | mesa-frames, AMBER | one DataFrame per agent class | ❌ iterate per type | row append/filter | Polars |
| **5. Fixed-cap SoA pytree + `active` mask + vmap** | Brax, MJX, PGX, **ABMax/Foragax** | one schema + masks; pad to capacity | via `vmap`+mask; **fixed capacity** | **soft-delete** (flip mask); no dynamic shapes | GPU/TPU (XLA), **autodiff** |
| **6. Archetype SoA × N worlds (batch-over-worlds)** | Madrona | static archetypes; **batch is a leading axis** | per-world parallel on GPU | static | GPU megakernel |

**Where microecs sits and what's unique.** microecs is strategy **1 on CPU/numpy with no compile step**, plus the
rare property that a *single* op spans all matching pools (the cross-archetype write-through). That combination —
**dynamic archetypes + cross-type vectorized write + zero compile/GPU** — is held by **no one else**:

- Strategy 2 (xecs, sparse-set engines) has no archetype concept at all.
- Strategy 3 (wide table) gets cross-type ops for free but wastes memory on nulls and needs masking.
- Strategy 4 (mesa-frames/AMBER) is the closest *philosophically* (homogeneous per-type stores) but **keys by
  Python class, not by component set**, and **cannot vectorize across types** — microecs's `query(A, …)`
  unions every pool that has `A` regardless of what else it has, in one view.
- Strategy 5 (JAX) trades dynamism for accelerators: every framework must know `max_agents` up front, "delete"
  is `where(active, …)` over padded arrays, and a **structural/shape change triggers recompile** (MJX states
  numpy fields are "structural fields that control JIT output"). microecs gets dynamic structure + instant edits
  for free *because* it stays on CPU/numpy.
- Strategy 6 (Madrona) is the GPU big sibling: same archetype DNA, but systems are compiled C++ and the
  vectorization axis is "N parallel worlds," not "across archetypes."

**The honest tradeoff microecs accepts:** archetype fragmentation + per-pool Python overhead in `_Field`
(strategy 1's tax) and CPU-only throughput. What it buys: dynamic archetypes, ragged-free dense blocks, no
recompile, no warmup, trivial debugging (`print` any array mid-step), and the cross-pool view.

---

## Part 4 — Ergonomics & efficiency, head-to-head

### Define a component / query / bulk-update — same task, each stack

```python
# microecs — numpy fields (metadata leaks; all three keys are required), AND+NOT query, write-through across pools
class HasVel(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})
qr = world.query(HasPos, HasVel, exclude=[Frozen])
qr.position = qr.position + qr.velocity * dt          # imperative, immediate, spans all matching pools

# xecs — typed component, Bevy-style typed query, per-component view (Rust under the hood)
class Velocity(xx.Component): value: xx.Vec2
def sys(q: xx.Query[tuple[xx.Transform2, Velocity]]):
    transform, velocity = q.result(); transform.translation += velocity.value

# manifoldx — decorator components/systems bound to the engine/renderer; one op per archetype
@engine.system
def step(q: mx.Query[Particle, Transform], dt: float): q[Particle].life -= dt

# mesa-frames — agents are Polars rows; bulk update is a Polars expression; one table per type
class MoneyAgents(AgentSet):
    def give(self): self.select(self.wealth > 0); self["active", "wealth"] -= 1   # + group_by / sample / joins

# JAX SoA+mask+vmap — pure function returns a NEW pytree; "delete" = where(active,…) over padded arrays
@struct.dataclass
class Agents: pos: jnp.ndarray; vel: jnp.ndarray; active: jnp.ndarray
def step(a, dt): return a.replace(pos=jnp.where(a.active[:,None], a.pos + a.vel*dt, a.pos))
batched = jax.vmap(step, in_axes=(0, None))              # lift across N worlds; needs jit + fixed capacity

# Warp — author a per-element kernel, JIT to CUDA, launch over the array
@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)):
    i = wp.tid(); pos[i] = pos[i] + vel[i]*dt
```

**Ergonomic read:** microecs's write-through is the **tersest for pure field math** and the only one that spans
heterogeneous types in one expression. It loses on (a) **component verbosity** (xecs/manifoldx typed fields are
cleaner than shape/dtype metadata), (b) **relational ops** — mesa-frames gets `group_by`/`sample`/joins free
from Polars; microecs is field-math only, and (c) **typed query signatures** (xecs's `Query[tuple[A,B]]` is
self-documenting + IDE-friendly).

**Functional vs imperative:** the JAX cluster is *pure* (return a new state pytree; `jit` fuses, `vmap` lifts) —
wins on composability + autodiff + accelerator fusion; microecs's in-place numpy wins on directness, zero
warmup, and debuggability.

### Structural-change safety — microecs is best-in-class

| Library | add/remove entity | add/remove component | mechanism |
|---|---|---|---|
| **microecs** | ✅ deferred | ✅ deferred | command buffer → `update()` (the **only** path; can't mutate eagerly) |
| Unity DOTS | ✅ | ✅ | `EntityCommandBuffer` (opt-in; user must route through it) |
| xecs / manifoldx | ✅ | ✅ | `Commands` / end-of-frame commands (Bevy-style) |
| esper / snecs | ✅ delete | ⚠️ immediate | — |
| tcod-ecs | ⚠️ snapshot yourself | ⚠️ snapshot yourself | — |

Deferring **all four** ops through one commit point, as the *only* path, is a genuine strength — arguably cleaner
than Unity's ECB (no "remember to use the buffer" footgun). Validated by Unity and Bevy.

### Efficiency scorecard (honest — measured Part 8, re-run 2026-07-26 vs seven libraries)

- **microecs wins:** columnar physics/bounce and per-entity ai from N≈5k (**~2.5–3.7× vs xecs at 100k**, 62× vs
  esper, **150× vs the C++ EnTT binding**), random access *if scattered* (9.9× vs xecs at 100k), and the realistic
  mixed frame from N≈20k (1.8× vs xecs at 100k). Plus **dynamic** structure, **zero warmup**, no compile/GPU,
  debuggable, tiny dep surface. It is also the **only vectorized-numpy lib in the benchmark field that can churn or
  migrate at all** (xecs can do neither; manifoldx has a free-list but is untested here), and at 1M entities the
  only library in the suite that steps a frame in single-digit ms.
- **microecs loses (N ≲ 1.5–3k):** the fixed per-op cost dominates below the columnar crossover, so xecs (leaner
  Rust path) wins columnar and ecs-pattern wins random/churn — by only ~1.2–1.4× on columnar. Many small games
  live *here* (hundreds–low-thousands): near-parity, not a rout.
- **microecs loses regardless of N: structural churn** — and the *reason* changed in this pass. It is **not**
  mainly the archetype pop-swap (2.2 µs, 17% of a churn pair): it is the **spawn path (79%)**, of which
  **validation is 32% with half of it duplicated** ([task 44]). ecs-pattern beats it 2–12× below 100k, EnTT by
  1.2× at 100k. On **component migration** it is now *competitive*, not behind: 1.02–1.07× vs EnTT at 5k–20k,
  0.77× at 100k. Large N on accelerators (JAX/Madrona 10⁴–10⁶ via GPU) and **autodiff** stay out of reach by design.
- **The one usage rule that decides microecs' fate: batch *everything*, including random access.** The public
  `get_entity(id).f` per-id loop is a **~2900 ns/hit trap** (483× a scattered `col[rows]-=x`). Any hot loop that
  falls back to it erases the vectorization win (it sank the mixed-frame number by 4–5× until fixed).
- **The measurement rule (new):** absolute ms on this machine drift **±10–25% between runs** — re-running the suite
  11 days apart moved every *unchanged* library by up to +12% median. Ratios inside one run are the only portable
  output; to judge a library change, normalize against the field, never compare ms across days.
- **Honest framing to publish:** **bulk numeric updates at scale = huge win; sub-crossover-N or per-entity-scalar
  logic = loss.** vectorization is a large-N bet (MJX says the same: "10× slower than C MuJoCo for a *single* scene").

[task 44]: ../todos/open/44-spawn-path-validates-twice/TASK.md

---

## Part 5 — Ideas worth stealing (ranked) + design choices validated

### Borrow — ranked cheap → high-value

1. **Per-pool optional / OR (the mechanism that unblocks the deferred `any_of`).** flecs (`?D`) and Bevy
   (`Option<&T>`) resolve optional/OR **per archetype**, never as one flat aligned column — which is *exactly*
   why microecs deferred them. The key realization: **`QueryResult` already stores each field as a list of
   per-pool parts** (`_Field.parts` concatenates them). Optional is therefore not a new data model — it's
   "attach a placeholder part for pools lacking the component, plus a presence mask":
   ```python
   # sketch only — feasibility, not a change request
   if D_bit & arch_key:  part, present = pool.data[f][:len], np.ones(len, bool)     # real column
   else:                 part, present = np.zeros((len, *shape), dtype), np.zeros(len, bool)  # padding
   ```
   `_Field` concatenates as today (alignment preserved by *padding* absent pools); `qr.present(D)` returns the
   mask. **The one hazard:** writes to placeholder rows must be a no-op/error (make optional fields read-only or
   masked-scatter). OR is the same trick on the matching loop (union of pools); filter-only OR needs no
   placeholders. **This is the single most borrowable advanced idea and it fits microecs's grug constraints.**
2. **Masked soft-delete + deferred compaction (from JAX ABM: ABMax/Foragax).** Instead of per-entity pop-swap,
   flip an `active` flag and **compact once per `update()`** (microecs already has that commit boundary). Append
   new entities at the tail; sort-by-active to keep live rows contiguous. Cache-friendly; a natural fit for the
   existing deferred command buffer.
3. **`group_by` / reduction as a query primitive (from mesa-frames / Polars).** microecs has AND+NOT filtering
   but no aggregation. A `qr.group_by(...)` / reduction would be the **highest-value addition if ABM users are a
   target audience** — it's the main thing Polars-backed peers get for free.
4. **Boolean sub-selection inside a view: `qr.where(mask)` (from JAX masks / `legal_action_mask`).** The whole
   JAX cluster shows masks are *the* ergonomic primitive for "operate on some, not all." First-class masked
   write-back lets users vectorize over a subset without leaving the `QueryResult`.
5. **Optional fixed-capacity fast path (ABMax `num_active` cursor).** When a pool's size is stable, pre-reserve
   capacity and write into a tail cursor to kill realloc churn in hot loops. Keep the dynamic path as default.
6. **Batch-over-worlds leading axis (from Madrona).** An optional leading "world/env" axis on the SoA columns so
   **N parallel sims** update in one vectorized pass. Pure-numpy-friendly, directly serves game/sim/physics, and
   is a clean way to scale on CPU. Roadmap-worthy.
7. **Per-column change tick (Bevy-lite change detection).** One `uint64` "changed-tick" per field per pool,
   stamped on a `_Field` write; a system compares against its last-run tick to skip untouched pools. Cheap —
   but be honest: this is **column granularity** ("did anything in this pool's `position` move?"), not
   per-entity. Per-entity dirty would need a parallel bool mask (expensive).
8. **Typed `Query[tuple[A, B]]` sugar (from xecs) + structured-dtype fields (Warp `vec3`).** IDE/mypy support
   and a nicer field ergonomic than raw float columns. Low-risk, pure-Python-doable via typing.

### Don't borrow

- **Relationships / pairs / `IsA`** (flecs, tcod-ecs) — flecs itself reports **5–10% add/remove overhead** +
  table fragmentation; they shine for graph/gameplay data, the opposite axis from dense numeric columns. Point
  users at tcod-ecs if they need it.
- **16KB chunking** (Unity) — numpy already hands you a contiguous block per column; re-tiling fights whole-array
  vectorization for ~zero gain at microecs's scale.
- **Second (sparse-set) storage type** (Bevy/EnTT) — a live Bevy discussion (#19164) argues for **removing**
  sparse-set storage because maintaining two code paths isn't worth it. Strong validation of microecs's single
  archetype-SoA path. *(Direction-of-travel, not a shipped change — flagged.)*
- **GPU / JIT / autodiff** (JAX, Warp, Madrona) — a different engineering universe (tracing, pure-functional
  state, recompile cliffs, GPU deps). Exactly the complexity the "one file per job, 2 deps" thesis rejects.
- **A native core under Python** (new — now measured, not assumed): binding EnTT or flecs makes arithmetic
  **2.3–3.4× slower than plain esper**, because components come back as Python objects and every access crosses
  the boundary. It buys only structural speed. If microecs ever wants faster churn, the answer is to stop doing
  redundant work in Python ([task 44]), not to rewrite the storage in C.

### Validated — microecs already got these right

- **Deferred command buffer for all structural change** — Unity's entire ECB exists to do this; microecs makes
  it the only path. ✅
- **Single archetype-SoA storage** — Bevy maintains a second sparse-set path and now wants to drop it; EnTT's
  "owning groups" are the sparse-set world straining to *recover* the contiguity archetypes give for free. And
  arXiv 2606.14919 (2026-06) formalizes archetype ECS and finds it wins empirically. ✅
- **Cache the matching pools, invalidate on structural commit** — exactly flecs's cached-query model
  ("prematched list of tables, cheap because archetypes are stable"). microecs's `_cache[(inc,exc)]` +
  clear-on-`update()` is the same idea in ~15 lines. ✅
- **Export the column as a view, never copy** — Madrona aliases GPU memory into zero-copy torch tensors; that is
  precisely microecs's `QueryResult` philosophy in numpy. **This is now the measured reason microecs beats both a
  Rust and a C++ engine** (Part 8, P2 + experiment 3): everyone else copies or crosses a boundary. ✅
- **SoA per field** — universal across Brax/MJX/Madrona/mesa-frames. ✅
- **Pure Python, no native core** — validated by experiment 3, where the two native cores lose to esper. ✅

---

## Part 6 — Features we need vs. don't need

### Need

1. ✅ **Query exclusion** — **shipped** (task 8, 2026-06-05). `query(A, B, exclude=[C, D])`; composite cache key;
   a cache-hit bug fixed alongside.
2. ✅ **Zero-size tag components** — **shipped** (task 9, 2026-06-05). Compose with `exclude=`.
3. **Single-component get/set by id** — **effectively answered, still no `get_component`.** The 2026-06 form of
   this gap ("`get_entity` copies all fields") is **gone**: `get_entity` is a dict lookup returning a cached
   `Entity` (74 ns), and single-field read/write go straight at the pool row — `e.f` 320 ns, `e.f = v` 471 ns,
   `set_data(f=v)` 610 ns (Part 8, P5). What remains is naming, not capability: there is no
   `get_component(eid, C)` returning one component's fields. **Two live sub-items** instead:
   (a) `_locate` spends ~89 of its 144 ns on `fields_set.issuperset([name])` where `name in fields_set` is 40 ns
   — a single-name fast path is ~15% of every entity read; (b) `set_data(f=v)` is now **slower** than `e.f = v`
   for the identical effect, so the docs must stop implying `set_data` is the cheap path.
4. ✅ **Low-N `QueryResult`/`Field` overhead** — **DONE ([task 26](../todos/done/26-low-n-field-overhead/TASK.md),
   2026-07-15).** microecs *used to* lose the 500–10k band to xecs on fixed per-op `Field`-allocation overhead.
   Fixed with (a) `_QRArray` — a single-pool query returns a thin `np.ndarray` subclass (native-C `+`/`*`, no
   per-op `QRField`), (b) a per-field `QRField` cache, (c) lazy `cumsum`. Result: microecs now beats xecs across
   the whole 200–20k band on columnar, crossover moved ~10k→~1.5–3k (Part 8 re-run), and the 500-UAV robosim
   driver is met with huge headroom (N=500 ≈ 5µs/frame vs the 16.6ms budget). Archived benchmark:
   `.tracker/todos/done/26-low-n-field-overhead/` (`run.py`, `bench.txt`).

### Evaluated candidates (from Part 5)

- **Optional / `any_of`** — **reviewed 2026-07-15 → not pursuing.** No use case, and it's derivable at
  user-code level: two queries (`query(A, D)` + `query(A, exclude=[D])`). The placeholder-part + presence-mask
  machinery (plus the write-to-placeholder hazard) is real complexity for convenience you already have. The
  mechanism is understood (Part 5 #1) if a concrete need ever lands.
- **`group_by` / reduction** — **reviewed 2026-07-15 → not pursuing.** Adds no *capability* over today's AND +
  numpy on the exported column (`np.add.at` / `np.bincount` over `qr.field.numpy()` with a group key). Per
  "enabler, not solutioner," don't wrap what the user can already do in one numpy call. Reconsider only if
  courting ABM users as a distinct audience.
- **Masked soft-delete + compaction at `update()`** — **reviewed 2026-07-15 → not pursuing; the 2026-07-26 churn
  breakdown confirms it.** Soft-delete would tax **every** query with an active-mask filter to speed up the delete
  path — and the delete path is only **21%** of a churn pair (pop-swap itself 17%), while the spawn path is 79%.
  Optimizing the wrong end. Do [task 44] instead.
- **Batch-over-worlds axis** — still a roadmap candidate for multi-sim/RL throughput on CPU (not reviewed).

### Don't need (scope discipline — matches CLAUDE.md minimalism)

- **Relationships / hierarchy**, **16KB chunking**, **second storage type**, **GPU/JIT/autodiff**, **event
  bus / observers** (the bounce task prefers an impulse accumulator over events), **system scheduler /
  parallelism** (systems are a convention), **serialization** (`object` dtype + pickle is the escape hatch).
  Rationale for each in Part 5 "Don't borrow."

---

## Part 7 — Positioning & recommended follow-ups

**Positioning (honest, evidence-based — now measured against seven libraries):** *"A pure-Python + numpy ECS
that's **alive** (xecs isn't), **standalone** (manifoldx is a renderer), with **AND + NOT + tags** queries
(neither peer has) and a **cross-pool vectorized write-through view** (no Python library has it), in ~850 LoC +
2 deps. esper's accessibility with numpy's bulk speed — no compile step, no GPU."* The honest perf caveat, with
numbers (Part 8, 2026-07-26): microecs wins the **large-N regime** (from N≈5k on every non-structural workload) —
~2.5–3.7× over xecs on columnar at 100k, 150× over a C++ EnTT binding, and it's the only vectorized-numpy lib in
the field that can churn/migrate — but **loses below N≈1.5–3k** to leaner libs (by ~1.2–1.4×), **loses structural
churn at every N** (a fixable validation cost, [task 44], not the layout), and demands you **batch even random
access**
(`get_entity` in a hot loop is a 483× trap). GPU/JAX engines still win at 10⁴–10⁶ + autodiff.

**New, and worth saying out loud:** the strongest evidence for the design is no longer "microecs beats esper" —
it is that **a Rust ECS and a C++ ECS both lose to it in Python**, for the same reason (a per-op copy or a per-access
boundary crossing). That is a claim about the *language*, not about microecs' cleverness, and it is the most
transferable thing in this document.

1. ✅ **[task 8](../todos/done/8-query-exclusion-none-of/TASK.md)** — query exclusion. **Done.**
2. ✅ **[task 9](../todos/done/9-tag-components/TASK.md)** — tag components. **Done.**
3. **README "Comparison / positioning" section** — *(not filed)* use the positioning paragraph above; name the
   real peers (xecs, manifoldx) + the adjacent clusters (Madrona, JAX ABM, mesa-frames); the cross-pool view as
   the headline. Fold in the Part 8 crossover chart + both boundary mechanisms (xecs' copy, entt/flecs' crossing).
   The public docs page (`docs/source/benchmarks.md`) already carries the three experiments; the README still only
   links to it.
4. ✅ **Benchmark** — **DONE (2026-07-15; re-run 2026-07-26 with 7 libs).** Superseded the single-number plan: a
   7-workload × 7-lib × N-sweep with a verified crossover + three mechanisms (Part 8). Code + data:
   `examples/05-benchmark-workloads/` (README, `results.json`, `FINDINGS.md`, `probes/`). *Not yet done:*
   mesa-frames (Polars) and manifoldx (per-archetype) baselines — worthwhile future additions, and manifoldx is
   the one that would test the cross-archetype claim head-on.
5. 🔥 **[Task 44](../todos/open/44-spawn-path-validates-twice/TASK.md) — kill the duplicate spawn-path validation.**
   *(filed 2026-07-26)* The single largest measured win available: ~2 µs/spawn, 20% of a full spawn, on the one
   workload microecs loses at every N.
6. **Entity-path polish** — *(not filed; recorded in [task 36](../todos/done/36-optimize-entity-read-write-path/TASK.md))*
   `_locate` single-name fast path (~15% of every entity read) and the `set_data` vs `e.f = v` cost inversion.
   Task 36 closed this question as "cost accepted" under `#29`'s design; `#42` changed that design, so the
   re-measured numbers live there.
7. **(designed, unbuilt) optional / `any_of`** — *(not filed)* the per-pool mechanism in Part 5 #1 if a use case
   appears.

---

## Part 8 — Empirical multi-workload benchmark (re-run 2026-07-26)

**Why?** The original benchmark measured **one** workload (columnar physics at N=100k) — microecs' best case — and
published a single "189× / 3.4×" headline. True but misleading: it doesn't tell a user what happens on the
workloads a *real* game/sim runs, or at the entity counts they use. This section answers that.

**What?** Seven workloads × **seven** libraries × an N-sweep (200 → 1,000,000), every result verified against a
float64 reference (order-independent fingerprint, so a lib can't look fast by skipping work). Code, raw numbers,
and mechanism probes: **`examples/05-benchmark-workloads/`** (`README.md`, `results.json`, `FINDINGS.md`,
`probes/`, one folder per workload × one file per library). Env: numpy 2.5.1, Py 3.12.12, times = min-over-reps of
mean-over-30-frames, GC off.

**The field now covers all three ways to build an ECS reachable from Python**, which is what makes this more than
a microbenchmark: numpy-vectorized (**microecs**), native core + native columnar store (**xecs**, Rust), native
core handing components back as Python objects (**entt** = C++ EnTT, **flecs** = C flecs), and pure Python
(**esper**, **snecs**, **ecs-pattern**). So the suite tests three *strategies*, not seven products —
see experiment 3.

Workloads (each = the real game system it models): **w1** physics (columnar integrate), **w2** bounce (columnar +
`np.where` branch), **w3** ai (per-entity health FSM), **w4** random (K read-modify-write by id/frame), **w5**
churn (spawn+FIFO-despawn/frame), **w6** mixed (physics+ai+targeted damage — a realistic steady frame), **w7**
migrate (component add/remove → archetype change).

**Fairness:** both columnar libs (microecs, xecs) batch — w4/w6 use the *same* `col[rows]-=x` scatter for each; the
per-entity libs (esper/snecs/ecs-pattern **and** entt/flecs, whose components are Python objects) loop with O(1) id
lookups, their fast path. The naive microecs `get_entity` loop is quantified as "the trap" (P3), not as microecs'
number.

**What this re-run was for.** The previous run predates `#42` (eager `e.field = v` and eager `set_data`), `#43`
(duplicate-component rejection, same-tick idempotent remove) and the `_locate` guard — all on the entity and
structural paths. Question: did they cost anything? **Answer: no, and the attempt to measure it taught the method
lesson below.**

### Experiment 1 — the headline: there is no global winner; it flips by workload AND by N

```
fastest lib     N=200        1k          5k          20k        100k
w1 physics      xecs         xecs        microecs    microecs   microecs
w2 bounce       xecs         xecs        microecs    microecs   microecs
w3 ai           esper        xecs        microecs    microecs   microecs
w4 random       ecs-pattern  ecs-pattern microecs    microecs   microecs
w5 churn        ecs-pattern  ecs-pattern ecs-pattern ecs-pattern entt
w6 mixed        esper        xecs        xecs        microecs   microecs
w7 migrate      snecs        entt        microecs    microecs   entt       (xecs, ecs-pattern: N/A)
```

fastest-competitor / microecs — **>1 = microecs wins by that factor**, and the rival is named because *which*
library it is matters as much as the number:

| workload | N=200 | 1k | 5k | 20k | 100k |
|---|---|---|---|---|---|
| w1 physics | 0.71 xecs | 0.84 xecs | **1.68 xecs** | **3.35 xecs** | **2.51 xecs** |
| w2 bounce | 0.71 xecs | 0.78 xecs | **1.19 xecs** | **2.13 xecs** | **2.42 xecs** |
| w3 ai | 0.56 esper | 0.99 xecs | **1.32 xecs** | **1.65 xecs** | **1.52 xecs** |
| w4 random | 0.81 ecs-pattern | 0.88 ecs-pattern | **2.96 ecs-pattern** | **5.82 xecs** | **9.91 xecs** |
| w5 churn | 0.08 ecs-pattern | 0.22 ecs-pattern | 0.36 ecs-pattern | 0.46 ecs-pattern | 0.82 entt |
| w6 mixed | 0.33 esper | 0.44 xecs | 0.77 xecs | **1.60 xecs** | **1.80 xecs** |
| w7 migrate | 0.24 snecs | 0.83 entt | **1.02 entt** | **1.07 entt** | 0.77 entt |

**Read it as:** microecs owns the **N≳5k regime** on every non-structural workload, plus scattered random access
and the realistic mixed frame from N≈20k. It is a near-tie on migration (1.02–1.07× vs EnTT mid-range) and only
genuinely *loses* structural churn — plus the smallest N (≲1.5–3k), where the fixed per-op cost still shows, by
~1.2–1.4× on columnar.

**Read it also as a map of who the rival is:** on every columnar/branchy/random cell the nearest competitor is
another **vectorized** library (xecs). A native per-entity engine (entt) only ever leads on the two **structural**
workloads. That split *is* the thesis of this document, and it now falls out of the data instead of being argued.

Two caveats on individual cells (both from `FINDINGS.md`): w3 ai @1k is a **tie** (0.0190 vs 0.0188 ms — it was
microecs' by the same margin last run), and w1 @100k reads 2.51 only because that run's min-of-2 for xecs landed
low; an 8-rep re-measure of that cell gives **3.3× on medians, 3.7× on mins** (w2: 2.5–2.7×).

### Experiment 2 — the columnar crossover and the "why is Rust slower?" answer

Columnar physics, ns/entity (microecs vs xecs, main sweep + large-N tail):

| N | 200 | 1k | 5k | 20k | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| microecs | 73 | 15 | **3.7** | **1.6** | **1.7** | **1.6** | **1.7** | **2.7** |
| xecs | 52 | 12 | 6.1 | 5.2 | 4.3 | 5.3 | 5.4 | 5.7 |

**Crossover ≈ N=1.5–3k.** xecs is *flat* at ~5 ns/e everywhere; microecs falls to ~1.7 ns/e as fixed overhead
amortizes. At 1M a physics frame is **2.7 ms (microecs) vs 5.7 ms (xecs)** — microecs is the only library in the
suite that steps 1M entities in single-digit ms with no GPU and no compile step. Why does a **pure-Python+numpy**
lib beat a **Rust** lib ~2–3.7× at scale? Verified with probes (P1–P3 in FINDINGS):

- **Not a dtype artifact (P1).** numpy 2.x (NEP 50) keeps `Float32 * python_float` in float32; forcing float32 dt
  gives 0.92–1.03× (no speedup) at every N from 1k to 1M. Refuted, twice now.
- **It's the Rust↔numpy copy boundary (P2, the key finding).** `xecs.view.x * dt` returns a **plain numpy array** —
  xecs does its arithmetic *in numpy*, not in Rust; the Rust core is just storage. `.numpy()` is a **copy** (its
  own docstring; no zero-copy accessor exists), so every columnar op copies operand columns out of Rust and writes
  results back — ≈6 buffer copies/step, **4–11× raw in-place numpy** (10.8× at 10k, 4.5× at 1M). microecs operates
  **in place on the very pool arrays** (verified zero-copy: `pool.py`, `qr_field.py` in-place ufunc `out=` path) —
  and a single-archetype query returns `_QRArray` (`query_result.py:11-20`), so there isn't even a `QRField` object
  in the loop. So microecs wins *because it has no FFI boundary* — the Rust core is a marshalling **tax** for
  CPU-vectorized work, not an asset. (Ruled out the `get_view()`-cost confound and the x/y-split confound: raw
  2-column numpy ≈ fused `(N,2)` numpy.)
- At small N (≲1.5k), microecs' residual fixed per-op cost (query dict lookup + view construction) still exceeds
  xecs' leaner path → xecs wins, by ~1.2–1.4×.

### Experiment 3 (new) — does a native core help? EnTT and flecs vs pure Python

The obvious objection to a pure-Python ECS is "just bind a real one." So we did: **EnTT** (C++ sparse-set, ships in
real games) and **flecs** (C archetype tables). Both store entities in native memory; both hand components back as
Python objects, so the system body is still a Python loop. That makes the comparison clean —
**entt/flecs and esper/snecs/ecs-pattern do identical Python work per entity; only the storage engine differs.**

best(entt, flecs) / best(esper, snecs, ecs-pattern); **>1 = the native core is slower**:

| workload | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| w1 physics | 3.43 | 3.13 | 2.57 | 2.72 | 2.41 |
| w2 bounce | 3.35 | 3.04 | 2.66 | 2.64 | 2.32 |
| w3 ai | 3.15 | 3.23 | 3.00 | 2.94 | 2.95 |
| w4 random | 2.21 | 2.41 | 2.89 | 2.89 | 2.29 |
| w5 churn | 2.04 | 2.30 | 2.29 | 1.89 | **0.90** |
| w7 migrate | 1.12 | **0.94** | **0.92** | **0.97** | **0.90** |

**A native core costs ~2.3–3.4× on field arithmetic and only pays off on structural work.** Where the work is
per-entity `p.x += p.vx * dt`, you pay Python's loop *plus* a boundary crossing per component access — so binding a
world-class C++ engine lands you *behind plain esper*. Where the work **is** the data-structure operation
(spawn/despawn, add/remove component), it all happens in C++ and EnTT becomes the fastest library in the suite.

Against microecs at N=100k, ns/entity/frame:

| workload | microecs | entt (C++) | flecs (C) | esper (pure py) | microecs vs entt |
|---|--:|--:|--:|--:|--:|
| w1 physics | **1.7** | 255.2 | 829.5 | 105.9 | **150× faster** |
| w2 bounce | **4.0** | 414.9 | 1392.9 | 179.2 | **104× faster** |
| w3 ai | **4.4** | 170.0 | 497.3 | 68.1 | **39× faster** |
| w4 random | **0.1** | 7.2 | 27.6 | 14.6 | **79× faster** |
| w5 churn | 219.2 | **180.0** | 554.1 | 414.8 | 1.2× slower |
| w6 mixed | **7.5** | 498.5 | 1500.3 | 249.8 | **66× faster** |
| w7 migrate | 227.3 | **175.9** | 620.7 | 477.7 | 1.3× slower |

Same mechanism as P2, pushed across three native runtimes: **in Python, the only thing that buys per-entity
arithmetic throughput is vectorization.** A faster storage engine can only make structural ops cheaper, and it
charges a boundary crossing for everything else. Honest caveat: both bindings are unmaintained hobby wrappers
(Part 2, Tier 1b) — better engineering would shave constants, not delete the crossing or the Python loop.

### The random-access trap (P3) — and why it decides the mixed frame

`world.get_entity(id).f` in a hot loop costs **~2900 ns/hit** (`get_entity` lookup + `_locate` guard + numpy row
index, flat in N). The batched SoA idiom `col[rows] -= x` costs **6 ns/hit at 100k — 483× less.** This is not
academic: with the naive loop microecs was **4–5× slower than xecs on the mixed frame at every N**; with the
(fair) scatter it **wins the mixed frame above N≈20k**. The whole vectorization advantage lives or dies on whether
the user batches random access. Caveat: batching needs a static set / a row-map rebuilt after `update()` (pop-swap
reorders rows) — so a *churning* world with per-id edits is genuinely hard.

**P5 (new) prices the same path per operation** (N=20k): raw numpy row read 110 ns → `get_entity` 74 → `_locate`
144 → `e.f` **320** → `e.f = v` **471** → `set_data(f=v)` **610** → `set_data(a=,b=)` 1433 → the composite
read-modify-write **2018**. Two consequences: (a) ~89 ns of `_locate`'s 144 is `fields_set.issuperset([name])`
where `name in fields_set` costs 40 — a single-name fast path is ~15% of every entity read; (b) **`set_data(f=v)`
is now slower than `e.f = v`** (610 vs 471) for the identical effect, so `#29`-era advice to prefer `set_data` no
longer matches the cost. Both recorded on [task 36](../todos/done/36-optimize-entity-read-write-path/TASK.md).

### P4 (new) — churn is a validation cost, not a pop-swap cost

microecs loses w5 churn at every N, and this pass finally split the 12.6 µs of one spawn+despawn pair:

| part | ns | share |
|---|--:|--:|
| full spawn (`add_entity` + `update()`) | 9964 | 79% |
| — `add_entity`, buffered (validates **twice**) | 5843 | 46% |
| — the commit | 4121 | 33% |
| full despawn (`remove_entity` + `update()`) | 2600 | 21% |
| **storage work** = `Pool.add_entity` 3227 + `_pop_from_pool` 2155 | 5382 | **43%** |
| **validation** = `_validate_components` 1627 ×2 + `_defaults_for` 383 ×2 | 4019 | **32%** |

So the archetype **pop-swap this plan blamed for years is 17%** of the pair, while validation is 32% — **and half of
that runs twice**: `World.add_entity` validates and computes defaults, then `CommandBuffer.append` validates and
computes defaults again on the same command. **~2 µs/spawn, 20% of a full spawn, 16% of a churn pair, is free.**
Filed as [task 44](../todos/open/44-spawn-path-validates-twice/TASK.md). This reframes the trade: archetype-SoA
does pay for structural change, but microecs' current gap to ecs-pattern/EnTT is mostly *its own bookkeeping*.

### Capability gaps decide churn & migration (not just speed)

- **xecs cannot despawn OR migrate** — `Commands.spawn` only; fixed-capacity, per-component pools. It is
  **disqualified** from any birth/death or component-toggling sim, regardless of its columnar speed. This is the
  sharpest practical differentiator vs microecs.
- **ecs-pattern cannot migrate** — entities are fixed inheritance-classes; no runtime component add/remove.
- Five libraries migrate: **microecs / entt / flecs / esper / snecs**. With the native engines in the field the
  picture changed from the 5-lib run: **EnTT is the migration and churn champion at 100k** (it is a C++
  data-structure operation), snecs leads at tiny N, and **microecs is competitive mid-range** (1.02–1.07× vs EnTT
  at 5k–20k) rather than "mid". Archetype-SoA buys contiguous columns at the price of structural change; the
  price is real but smaller than assumed — and partly self-inflicted (P4).

### Workload → real game/sim regimes (which of these matters?)

| workload | real systems | typical N | who wins there |
|---|---|--:|---|
| w1/w2/w3 columnar | particles/VFX, bullet-hell, boids, N-body, **UAV/vehicle integrators**, RL rollouts, ABM ticks | 1k–1M | xecs/esper <~1.5–3k, **microecs above** |
| w4 random | ARPG/MOBA damage, hitscan, targeted heals, net delta-apply | hits 1–100 / 100s–few-k live | ecs-pattern tiny-N, **microecs at scale** |
| w5 churn | bullet emitters, TD creep waves, spawn/despawn pools | births 10–1k/frame | ecs-pattern small-N, **EnTT at 100k** (**not xecs**) |
| w7 migrate | state tags (Alive↔Dead), buff/debuff add-remove | K/frame | snecs tiny-N, EnTT at 100k, **microecs competitive mid-range** |

**Reality check:** many *games* live around or below the crossover — RTS units 200–2k, ARPG hundreds, roguelikes
tens. That mid-range (1–3k) is **near-parity**, not a clear xecs/ecs-pattern win, and anything ≳5k is microecs
territory. microecs' large-N advantage is decisive for **particle systems, big bullet-hell, and agent-based /
physics simulation at N≳5k** — its stated niche. For robosim itself (~2 robots + tens of world entities, N≈10²)
microecs is well below its own crossover; here it's chosen for its **API + zero-copy state + dynamic structure**,
not throughput, and at that N *all* libs are sub-millisecond so it doesn't matter.

**Strategic verdict:** microecs' real audience is **vectorizable ABM / physics sim at scale**, and for that
audience the winning competitor is *microecs itself at large N* — xecs only wins the small-N/tight-loop regimes
this audience doesn't operate in and forfeits churn and migration entirely; the native-core bindings lose the
arithmetic they'd be bought for. The one thing microecs users must internalize: **batch everything, including
random access.**

### Did `#42`/`#43` cost anything? No — and the method lesson

This re-run's actual question. Answer, four ways:

- **Absolute ms says "everything got 10% slower" — and it is lying.** Median shift of the **six non-microecs**
  libraries between the two runs: **+7.1% (N=200), +6.5% (1k), +11.7% (5k), −0.3% (20k), +3.3% (100k)**. Those
  libraries did not change a line of code. On this machine, **absolute times drift ±10–25% run-to-run**.
- **Normalized against the field** (microecs ÷ median of the other six, inside each run), microecs moved by
  mixed-sign single digits on six of seven workloads: w1 +2.4%, w2 +4.8%, w3 +1.0%, w4 −2.6%, w5 +4.8%, w6 +3.3%,
  w7 +3.1% (mean over N). Only w7's five cells share a sign, at +3% mean. Nothing here clears the noise floor.
- **Direct measurement of the one added cost:** `#43`'s idempotency bookkeeping (`entity_id not in live_entities`
  + `removed_this_tick.add`) is **50 ns — 1.9% of a full despawn, 0.4% of a churn pair.** That is the honest price
  of same-tick idempotent removal, and it is unmeasurable through the harness.
- **Winner map: one cell moved**, w3 ai @1k, a 1% tie.

**Method fixed as a result** (also in the benchmark README): compare **ratios inside one run**, never ms across
days; normalize against the unchanged field when judging a library change; keep the machine idle while the suite
runs. A first attempt at this re-run was **discarded** because unrelated work (network fetches, file edits) ran
concurrently and inflated every library by 10–30% — which looked exactly like a regression until the control
libraries gave it away.

### Honest limitations of this benchmark (pre-empting the skeptic)

- **Run-to-run drift is ±10–25% on absolutes** (quantified above). Only ratios inside one run, the winner map, and
  the crossover are portable; treat any single cell as ±20%. Where a headline rides on one cell, re-measure it with
  more reps (done for columnar @100k: 8 reps).
- **GC-off + min-over-reps flatters the object-per-entity libs** (a real frame budget cares about p99, not the
  min, and long GC-on sessions expose the object-churn pauses esper/snecs/ecs-pattern generate). A GC-on
  p50/p99 long-run variant would be the fairer "real session" number — worthwhile future work.
- **float32 (SoA) vs float64 (object libs)** is half the bytes — a real bandwidth edge for SoA at large N,
  independent of python overhead. microecs vs xecs is fair (both f32).
- Workloads use **1–2 archetypes** (except w7); heavy **archetype fragmentation** (many tiny pools, where
  `QRField`-concat-across-pools degrades) is untested and is microecs' theoretical soft spot.
- Verification is an **order-independent multiset** fingerprint: catches skipped/wrong-magnitude work, not a
  which-entity permutation. Adequate for "didn't cheat," not a per-entity correctness proof.
- min-16 touch floor → small-N w4/w5 touch ~8%/frame, heavier than the advertised ~2%/1%.
- **The two native bindings are unmaintained hobby wrappers.** The engines aren't on trial; the *strategy* of
  binding one from Python is.

---

## Sources

**Direct peers** — xecs https://github.com/lukasturcani/xecs · https://pypi.org/pypi/xecs/json ·
https://xecs.readthedocs.io · manifoldx https://github.com/apiad/manifoldx · https://pypi.org/pypi/manifold-gfx/json ·
https://blog.apiad.net/p/realtime-3d-in-pure-python-numpy

**GPU / data-oriented** — Madrona https://github.com/shacklettbp/madrona · https://madrona-engine.github.io ·
SIGGRAPH'23 https://madrona-engine.github.io/shacklett_siggraph23.pdf · example
https://github.com/shacklettbp/madrona_escape_room · Warp https://github.com/NVIDIA/warp · Taichi
https://github.com/taichi-dev/taichi · "halted" discussion https://github.com/taichi-dev/taichi/discussions/8506 ·
Genesis https://github.com/Genesis-Embodied-AI/genesis-world · Newton
https://developer.nvidia.com/newton-physics

**JAX / batched-state** — Brax https://github.com/google/brax (paper https://arxiv.org/abs/2106.13281) ·
MJX https://mujoco.readthedocs.io/en/stable/mjx.html · PGX https://github.com/sotetsuk/pgx · Gymnax
https://github.com/RobertTLange/gymnax · Jumanji https://github.com/instadeepai/jumanji · JaxMARL
https://github.com/FLAIROx/JaxMARL · ABMax https://arxiv.org/abs/2508.16508 · Foragax
https://github.com/i-m-iron-man/Foragax (https://arxiv.org/abs/2409.06345) · EnvPool
https://github.com/sail-sg/envpool · flax.struct https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html

**Dataframe / ABM** — mesa-frames https://github.com/projectmesa/mesa-frames ·
https://projectmesa.github.io/mesa-frames/ · AMBER https://arxiv.org/abs/2601.16292 ·
https://github.com/a11to1n3/AMBER · Mesa https://github.com/projectmesa/mesa · "DataFrames as ECS"
https://medium.com/@arsdragonfly/dataframes-might-be-an-underrated-entity-component-system-for-game-development-dfb72b1819fe ·
awkward array https://github.com/scikit-hep/awkward · numpy structured arrays
https://numpy.org/doc/stable/user/basics.rec.html

**Native ECS (ideas only)** — flecs queries https://www.flecs.dev/flecs/md_docs_2Queries.html ·
relationships https://www.flecs.dev/flecs/md_docs_2Relationships.html · Bevy storage PR
https://github.com/bevyengine/bevy/pull/1525 · sparse-set-removal discussion
https://github.com/bevyengine/bevy/discussions/19164 · change detection
https://docs.rs/bevy/latest/bevy/ecs/change_detection/trait.DetectChanges.html · Unity structural changes
https://docs.unity3d.com/Packages/com.unity.entities@1.0/manual/concepts-structural-changes.html · Unity chunk
layout https://rams3s.github.io/blog/2019-01-09-ecs-deep-dive/ · EnTT views/groups
https://github.com/skypjack/entt/wiki/Crash-Course:-entity-component-system · tcod-ecs
https://github.com/HexDecimal/python-tcod-ecs · ecs-faq https://github.com/SanderMertens/ecs-faq

**Per-entity Python (benchmark baselines)** — esper https://github.com/benmoran56/esper · ecs-pattern
https://github.com/ikvk/ecs_pattern (PyPI `ecs-pattern`) · snecs https://github.com/slavfox/snecs · entitas-python
https://github.com/Aenyhm/entitas-python

**Native-engine Python bindings (the control group)** — PyEnTT https://github.com/ominkk/PyEnTT (PyPI `entt`,
nanobind over https://github.com/skypjack/entt) · pyflecs11 https://github.com/Wesxdz/pyflecs11 (PyPI `flecs`,
pybind11 over https://github.com/SanderMertens/flecs)

**ECS theory** — *The Essence of Entity Component System*, Tasnim & Zhao, arXiv 2606.14919 (2026-06-12) — formal
semantics for archetype ECS (creation, composition, system execution, migration as state transitions) + an
empirical result favouring archetype ECS. No vectorization/SoA/Python content.

**Part 8 benchmark** — `examples/05-benchmark-workloads/` (harness, per-workload adapters, `results.json`,
`FINDINGS.md`, `probes/xecs_dtype.py` / `probes/boundary.py` / `probes/microecs_random.py`). xecs numpy-boundary
confirmed in its installed source `xecs/_internal/vec2.py` (`__iadd__`/`__mul__` do `self.numpy()` + `np.*` +
write-back) and `Float32.numpy.__doc__` ("Copy the elements into a NumPy array"). P4/P5 and the run-to-run
comparison came from `test/manual/bench-compare/` (`delta.py`, `relative.py`, `paths.py`, `spawn_breakdown.py`,
`repeat_100k.py`) —
gitignored, so their numbers are inlined in `FINDINGS.md` and above.

**Method note:** landscape facts re-verified **2026-07-26** via GitHub API / PyPI JSON / docs (prev 2026-07-15);
**no material drift** — xecs still dead (3★, last push 2023-10-30, no despawn/migrate), snecs dead, esper 695→696★
& ecs-pattern (`ikvk/ecs_pattern`, 54★) both lightly-active but unvectorized, manifoldx **still active but still**
renderer-locked with no cross-archetype write (README re-read), and **still no JAX/torch-backed Python ECS
*library***. New this pass: both native-binding repos are hobby-grade (PyEnTT 1★ archived 2026-03-05; pyflecs11 2★,
last push 2025-08-21). Part 8 numbers are this-machine point-in-time (numpy 2.5.1, Py 3.12.12) and drift ±10–25%
run-to-run; the *ratios/crossover* are the portable result. Author-run third-party benchmarks (mesa-frames 10×,
AMBER 1.7–93×) remain `[claim]`. "Bevy removing sparse-set" is still a proposal.
