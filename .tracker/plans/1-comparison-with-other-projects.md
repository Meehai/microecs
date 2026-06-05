# microecs vs. the vectorized-entity landscape — analysis & positioning

**Created**: 2026-06-04
**Updated**: 2026-06-05 (refocused on the vectorized + Python-interop niche; added GPU batch-ECS, JAX, and
dataframe-ABM clusters; turned "ideas" into a ranked borrow list)
**Type**: Reference / competitive analysis
**Scope**: Assess microecs (efficient? good? nice?), then compare **only against projects in the same domain**:
must (a) interop with Python and (b) be **vectorized** (numpy / torch / jax / polars — bulk array math, not
per-entity Python loops). The per-entity Python ECS crowd is explicitly **out of scope** (pruned to one line).
Decide what ideas are worth borrowing.

**Confidence**: stars/dates are June 2026 point-in-time snapshots (drift). Author-claimed benchmarks are flagged
`[claim]`. Primary sources (GitHub API, PyPI JSON, docs, papers) in Sources.

---

## TL;DR (grug verdict)

microecs is a **good, correct, genuinely useful** little engine for its target — bulk numeric updates over many
entities. Its differentiator — numpy **SoA by archetype** + a **cross-archetype vectorized write-through view**
(`qr.position[:] = qr.position + qr.velocity*dt` lands in *every* matching pool via numpy's array protocols) —
appears **genuinely unique among Python libraries**.

- **The direct peer set is nearly empty**, and both peers have a fatal gap vs microecs:
  - **xecs** — the "serious" attempt (Rust core + numpy). **Effectively dead** (last commit 2023-10-05, 3★),
    **AND-only** queries, and **no archetypes** (per-component columns).
  - **manifoldx / manifold-gfx** — the truest architectural *twin* (pure Python+numpy, SoA-by-archetype,
    deferred commits, April 2026). But **welded to a wgpu renderer**, requires **Python ≥3.13**, and does
    **one numpy op *per archetype*** — it has **no cross-archetype view**, which is exactly microecs's headline.
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
- **Best ideas to borrow** (Part 5): per-pool **optional/OR** (the mechanism that unblocks the deferred
  `any_of`), **masked soft-delete at the commit boundary** (from JAX ABM), **group-by/reduction** (from
  Polars ABM), and an optional **batch-over-worlds** axis (from Madrona).

microecs's slot, stated precisely: **the only Python, CPU, dynamic-archetype ECS with a cross-pool vectorized
write — esper's accessibility with numpy's bulk speed, no compile step, no GPU, ~300 LoC, 2 deps.**

---

## Part 1 — Is microecs efficient / good / nice?

### Efficient — yes, for its target

| Mechanism | Where | Verdict |
|---|---|---|
| SoA storage, one numpy array per field per archetype | `pool.py:23` | ✅ cache-friendly, the whole point |
| Bitmask keys; match `(arch & inc) == inc and (arch & exc) == 0` (AND + NOT) | `world.py:110` | ✅ simple, correct |
| Query result **cached** between updates (keyed by include+exclude) | `world.py:98`, `world.py:51` | ✅ steady-state queries are O(1) |
| Deferred command buffer (no iterator invalidation) | `world.py:38-45` | ✅ same approach as flecs/Bevy/Unity |
| Vectorized cross-pool write | `query_result.py:56-71` | ✅ the differentiator (no Python peer has it) |

For a motion/physics system over thousands of same-archetype entities, the work runs in numpy and beats
per-entity Python loops by **1–2 orders of magnitude**. That is the headline benchmark to publish. The SoA-per-
field choice is **universally validated** — Brax (`QP`), MJX (`Data`), Madrona (component columns), mesa-frames
(Polars columns) all store one array per field. microecs is on the canonical path.

**Honest inefficiencies** (inherent to the approach, not bugs):

- **`_Field` loops over pools in Python** (`query_result.py:29-34`), allocates a new `_Field` per op, and does
  `np.broadcast_to` / `np.split` on writes (`query_result.py:69-71`). Vectorization only wins *within* a pool;
  a query over many small pools degenerates toward per-pool Python overhead.
- **Archetype fragmentation** — classic archetype-ECS tax: many component combos → many small pools.
- **`add/remove_component` copies the whole entity** to a new pool (`world.py:128-142`, `pool.py:52`). Fine
  occasionally, bad every tick — which is why the bounce task (`todos/open/1`) uses an impulse accumulator
  instead of per-tick component churn. Good instinct.
- **`get_entity` copies *all* fields** into a dict (`world.py:66`). No "read one component for one entity" fast
  path (Part 6, gap #3).
- Keys are `2**i` Python ints (`world.py:20`), arbitrary-precision → **no 64-component cap** that bites C
  engines. But a cache miss is a linear scan of all pools (`world.py:89`).

### Good — yes (design quality)

Clean single-responsibility split: `Pool` (SoA dynamic array, no id concept) / `World` (ids + archetypes +
command buffer) / `QueryResult` (cross-pool view) / `Component` (dataclass) / `System` (convention). ~300 lines
of logic, 2 deps. Tests are genuinely strong: the 500-op randomized churn invariant check (`test_world.py:687`)
and the recarray-parity tests (`test_queryresult.py:458`) are things most ECS libraries lack. The
`__array_ufunc__` / `__array_function__` impl matches numpy broadcast + recarray write-through semantics, with
sharp edges rejected rather than silently wrong.

**Quality nits** (non-test files — dev's call):
- `_Field.__array_function__` whitelists only `{np.where, np.clip}` (`query_result.py:9`); everything else is
  `NotImplemented`. Documented, but hit often → forces `.numpy()`.
- `from typing import Callable, T` (`query_result.py:2`) imports the *private* `typing.T`. Works, fragile.

### Nice — mostly

The vectorized write idiom is the nicest thing here:

```python
qr.position[:] = qr.position + qr.velocity * DT   # updates every matched entity, across all pools, in numpy
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
| **xecs** (`lukasturcani/xecs`) | Rust core + numpy, ships wheels | **3★**, last commit **2023-10-05**, v0.9.0 → **dead** | **Per-component** flat arrays (numpy-exposed); **no archetypes** | ✅ per-component (`transform.translation += velocity.value`); also `product_2` all-pairs | **AND only** (no exclude/optional/OR) | **PEER (stalled)** — closest intent (Bevy-style numpy ECS) but dead, weaker queries, no archetype model |
| **manifoldx / manifold-gfx** (`apiad/manifoldx`) | **Pure Python + numpy** (+ wgpu/rendercanvas), Py ≥3.13 | **6★**, created **2026-04-03**, v0.3.0 → **active/new** | **SoA by archetype** + free-list (≈ microecs's Pool) | ✅ within an archetype; **one op *per archetype*** — **NO cross-archetype view** | AND (component filter); no NOT/optional shown | **PEER (truest twin)** — same SoA-archetype + deferred commits, but renderer-locked + per-archetype-only + heavy deps |

**The two peers bracket microecs.** xecs went the Rust route and stalled; manifoldx went pure-numpy (microecs's
exact stack) but is a *renderer* and crucially **lacks the cross-archetype write-through** — its author's design
writeup explicitly describes "one method invocation per archetype." microecs's `QueryResult` is the thing
neither has. *No JAX- or torch-backed Python ECS library exists* (confirmed by targeted search) — that sub-niche
is empty.

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

- **esper** (688★, **active**) — the popular pure-Python ECS; per-entity loop. *The positioning foil:* "esper's
  simplicity, numpy's bulk speed." Keep as a benchmark baseline only.
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
# microecs — numpy fields (metadata leaks), AND+NOT query, in-place write-through across pools
class HasVel(Component): velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})
qr = world.query(HasPos, HasVel, exclude=[Frozen])
qr.position[:] = qr.position + qr.velocity * dt          # imperative, immediate, spans all matching pools

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

### Efficiency scorecard (honest)

- **microecs wins:** small-to-mid N, **dynamic** structure, **zero warmup**, no compile/GPU, debuggable, tiny
  dep surface. Bulk numeric step over a few thousand same-archetype entities ≈ 1–2 orders over per-entity loops.
- **microecs loses:** large N on accelerators (JAX/Madrona run 10⁴–10⁶ via GPU+fusion), **autodiff** through the
  sim (can't), and per-pool Python overhead when a query spans many tiny pools.
- **The honest framing to publish** (from AMBER's measured 1.7×–93× spread vs Mesa `[claim]`): **bulk numeric
  updates = huge win; irreducibly sequential / branchy per-entity logic = modest win.** Don't over-claim. MJX's
  own "10× slower than C MuJoCo for a *single* scene" is the analogous honesty from the GPU side — vectorization
  is a large-N bet.

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
  state, recompile cliffs, GPU deps). Exactly the complexity the "300 LoC, 2 deps" thesis rejects.

### Validated — microecs already got these right

- **Deferred command buffer for all structural change** — Unity's entire ECB exists to do this; microecs makes
  it the only path. ✅
- **Single archetype-SoA storage** — Bevy maintains a second sparse-set path and now wants to drop it; EnTT's
  "owning groups" are the sparse-set world straining to *recover* the contiguity archetypes give for free. ✅
- **Cache the matching pools, invalidate on structural commit** — exactly flecs's cached-query model
  ("prematched list of tables, cheap because archetypes are stable"). microecs's `_cache[(inc,exc)]` +
  clear-on-`update()` is the same idea at 300 LoC. ✅
- **Export the column as a view, never copy** — Madrona aliases GPU memory into zero-copy torch tensors; that is
  precisely microecs's `QueryResult` philosophy in numpy. ✅
- **SoA per field** — universal across Brax/MJX/Madrona/mesa-frames. ✅

---

## Part 6 — Features we need vs. don't need

### Need

1. ✅ **Query exclusion** — **shipped** (task 8, 2026-06-05). `query(A, B, exclude=[C, D])`; composite cache key;
   a cache-hit bug fixed alongside.
2. ✅ **Zero-size tag components** — **shipped** (task 9, 2026-06-05). Compose with `exclude=`.
3. **Single-component get/set by id** — **open, minor.** No `get_component`/`set_component`; `get_entity` copies
   all fields (`world.py:68`). The only remaining original "need."

### Evaluated candidates (from Part 5 — not committed, just on the table)

- **Optional / `any_of`** — was "deferred because it breaks the aligned column." **Now we know the mechanism**
  (per-pool placeholder parts + presence mask, Part 5 #1). Still no current use case, but it's no longer a
  mystery — promote from "deferred" to "designed, unbuilt" if a need lands.
- **`group_by` / reduction** — new candidate, **conditional on wanting ABM users**. Highest ergonomic ROI.
- **Masked soft-delete + compaction at `update()`** — internal optimization candidate (no API change).
- **Batch-over-worlds axis** — roadmap candidate for multi-sim/RL throughput on CPU.

### Don't need (scope discipline — matches CLAUDE.md minimalism)

- **Relationships / hierarchy**, **16KB chunking**, **second storage type**, **GPU/JIT/autodiff**, **event
  bus / observers** (the bounce task prefers an impulse accumulator over events), **system scheduler /
  parallelism** (systems are a convention), **serialization** (`object` dtype + pickle is the escape hatch).
  Rationale for each in Part 5 "Don't borrow."

---

## Part 7 — Positioning & recommended follow-ups

**Positioning (honest, evidence-based):** *"A pure-Python + numpy ECS that's **alive** (xecs isn't),
**standalone** (manifoldx is a renderer), with **AND + NOT + tags** queries (neither peer has) and a
**cross-pool vectorized write-through view** (no Python library has it), in ~300 LoC + 2 deps. esper's
accessibility with numpy's bulk speed — no compile step, no GPU."* The honest perf caveat: vectorization wins on
bulk numeric updates (large-N), modest on sequential/branchy logic; GPU/JAX engines win at 10⁴–10⁶ + autodiff;
per-entity Python libs win on churn/branchy game logic.

1. ✅ **[task 8](../todos/done/8-query-exclusion-none-of/TASK.md)** — query exclusion. **Done.**
2. ✅ **[task 9](../todos/done/9-tag-components/TASK.md)** — tag components. **Done.**
3. **README "Comparison / positioning" section** — *(not filed)* use the positioning paragraph above; name the
   real peers (xecs, manifoldx) + the adjacent clusters (Madrona, JAX ABM, mesa-frames); the cross-pool view as
   the headline.
4. **Benchmark task** — *(not filed)* publish the headline number against honest baselines: **esper** (per-entity
   loop), **mesa-frames** (Polars per-type), and **manifoldx** (per-archetype). The story is the 1–2 orders over
   esper and the cross-pool-view advantage over manifoldx.
5. **Single-component get/set by id** — *(not filed; minor)* Part 6 #3.
6. **(designed, unbuilt) optional / `any_of`** — *(not filed)* the per-pool mechanism in Part 5 #1 if a use case
   appears.

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

**Per-entity Python (pruned)** — esper https://github.com/benmoran56/esper · ecs-pattern
https://pypi.org/pypi/ecs-pattern/json · snecs https://pypi.org/pypi/snecs/json · entitas-python
https://github.com/Aenyhm/entitas-python

**Method note:** facts verified June 2026 via GitHub API / PyPI JSON / docs / arXiv. Star counts are
point-in-time and drift. Author-run benchmarks (mesa-frames 10×, AMBER 1.7–93×, DataFrames-as-ECS 9ns) are
`[claim]` — credible but unreplicated. "Bevy removing sparse-set" is a maintainer proposal, not shipped. No
JAX/torch-backed Python ECS *library* was found as of June 2026.
