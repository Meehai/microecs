# microecs vs. the vectorized-entity landscape — analysis & positioning

**Created**: 2026-06-04
**Updated**: 2026-07-15 (added **Part 8 — empirical multi-workload benchmark**: 6 workloads × 5 libs ×
N-sweep, verified crossover + copy-boundary mechanism; corrected the efficiency scorecard with measured
numbers; competitor status re-verified — no material drift since June)
**Prev update**: 2026-06-05 (refocused on the vectorized + Python-interop niche; added GPU batch-ECS, JAX,
and dataframe-ABM clusters; turned "ideas" into a ranked borrow list)
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

**Now measured (Part 8).** A 6-workload × 5-library × N-sweep confirms the thesis *and* bounds it honestly:
**there is no global winner — the ranking flips by workload and by N.** microecs wins the **large-N regime broadly**
(N≥~20k: columnar physics, random-access-via-scatter, and a realistic mixed frame) and is the *only* vectorized-numpy
lib that can also churn/migrate. Below N≈10k its fixed per-op query/Field overhead loses to leaner libs (xecs,
ecs-pattern). The single most important mechanism finding: **microecs beats the Rust ECS (xecs) at scale precisely
because it has no FFI boundary** — xecs does its arithmetic in numpy anyway and pays a Rust↔numpy *copy* per op;
microecs mutates the pool arrays in place. The Rust core is a marshalling tax for CPU-vectorized work, not a win.

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
- **Ideas evaluated** (Part 5/6): optional/OR, masked soft-delete, and group-by/reduction were **reviewed
  2026-07-15 and set aside** — each is derivable at user-code level or a bad trade for a query-first lib
  (rationale in Part 6). The live perf need is **low-N `Field` overhead** ([task 26](../todos/open/26-low-n-field-overhead/TASK.md),
  robosim's 500-UAV driver); an optional **batch-over-worlds** axis (from Madrona) stays a roadmap candidate.

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

For a motion/physics system over **tens of thousands** of same-archetype entities, the work runs in numpy and
beats per-entity Python loops by **1–2 orders of magnitude** (measured 30–60× vs esper at N=100k; Part 8). But the
win is **N-gated**: below N≈10k the per-op overhead (query + `_Field` construction) dominates and per-entity libs
are actually *faster* — at N=200 microecs is ~4× *slower* than xecs and neck-and-neck with esper. The headline to
publish is therefore not "microecs is fast" but "**microecs is fast at scale** (N≳10k) — at small N its fixed
overhead makes leaner libs win." The SoA-per-field choice is **universally validated** — Brax (`QP`), MJX
(`Data`), Madrona (component columns), mesa-frames (Polars columns) all store one array per field. microecs is on
the canonical path.

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

### Efficiency scorecard (honest — now measured, Part 8)

- **microecs wins (N ≳ 20k):** columnar physics/bounce/AI (3× vs xecs at 100k, 30–60× vs per-entity libs),
  random access *if scattered* (10× vs the field at 100k), and the realistic mixed frame (1.7× vs xecs). Plus
  **dynamic** structure, **zero warmup**, no compile/GPU, debuggable, tiny dep surface. It is also the **only
  vectorized-numpy lib that can churn or migrate at all** (xecs can do neither).
- **microecs loses (N ≲ 10k):** the fixed per-op cost (query + `_Field` alloc + per-pool python loop) dominates,
  so xecs (leaner Rust path) wins columnar and ecs-pattern/esper win random/churn. Real games often live *here*
  (hundreds–few thousand entities) — this is not microecs' regime.
- **microecs loses regardless of N:** structural **churn** (ecs-pattern/snecs beat it; the archetype pop-swap +
  realloc tax) and is *mid* on **component migration** (snecs' sparse-set beats archetype migration — copying the
  whole entity to a new pool, exactly the cost flagged in Part 1). Large N on accelerators (JAX/Madrona 10⁴–10⁶
  via GPU) and **autodiff** are out of reach by design.
- **The one usage rule that decides microecs' fate: batch *everything*, including random access.** The public
  `get_entity(id).f` per-id loop is a **~2600 ns/hit trap** (400× a scattered `col[rows]-=x`). Any hot loop that
  falls back to it erases the vectorization win (it sank the mixed-frame number by 4–5× until fixed).
- **Honest framing to publish:** **bulk numeric updates at scale = huge win; small-N or per-entity-scalar logic =
  loss.** vectorization is a large-N bet (MJX says the same: "10× slower than C MuJoCo for a *single* scene").

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
4. **Low-N `QueryResult`/`Field` overhead** — **open ([task 26](../todos/open/26-low-n-field-overhead/TASK.md)).**
   microecs loses the **500–10k** band to xecs on fixed per-op `Field`-allocation overhead (Part 8 crossover
   ~10k). **Concrete driver:** robosim needs ~500 UAVs at 60 fps and can't today. Focused tracker:
   `examples/06-benchmark-vs-xecs-low-entities/` (the per-frame breakdown shows the step at N=500 is ~21× the
   raw-numpy floor, and it's `Field` churn — `query()` is already cheap/cached).

### Evaluated candidates (from Part 5)

- **Optional / `any_of`** — **reviewed 2026-07-15 → not pursuing.** No use case, and it's derivable at
  user-code level: two queries (`query(A, D)` + `query(A, exclude=[D])`). The placeholder-part + presence-mask
  machinery (plus the write-to-placeholder hazard) is real complexity for convenience you already have. The
  mechanism is understood (Part 5 #1) if a concrete need ever lands.
- **`group_by` / reduction** — **reviewed 2026-07-15 → not pursuing.** Adds no *capability* over today's AND +
  numpy on the exported column (`np.add.at` / `np.bincount` over `qr.field.numpy()` with a group key). Per
  "enabler, not solutioner," don't wrap what the user can already do in one numpy call. Reconsider only if
  courting ABM users as a distinct audience.
- **Masked soft-delete + compaction at `update()`** — **reviewed 2026-07-15 → not pursuing.** The current
  deferred pop-swap already keeps live rows dense with no per-query tax; soft-delete would tax **every** query
  with an active-mask filter to speed up only the (cold, already-batched) delete path — the wrong trade for a
  query-first lib. Revisit only if churn benchmarks show pop-swap is the bottleneck (w5 has microecs mid-pack;
  the cost there is build/add, not remove).
- **Batch-over-worlds axis** — still a roadmap candidate for multi-sim/RL throughput on CPU (not reviewed).

### Don't need (scope discipline — matches CLAUDE.md minimalism)

- **Relationships / hierarchy**, **16KB chunking**, **second storage type**, **GPU/JIT/autodiff**, **event
  bus / observers** (the bounce task prefers an impulse accumulator over events), **system scheduler /
  parallelism** (systems are a convention), **serialization** (`object` dtype + pickle is the escape hatch).
  Rationale for each in Part 5 "Don't borrow."

---

## Part 7 — Positioning & recommended follow-ups

**Positioning (honest, evidence-based — now measured):** *"A pure-Python + numpy ECS that's **alive** (xecs
isn't), **standalone** (manifoldx is a renderer), with **AND + NOT + tags** queries (neither peer has) and a
**cross-pool vectorized write-through view** (no Python library has it), in ~300 LoC + 2 deps. esper's
accessibility with numpy's bulk speed — no compile step, no GPU."* The honest perf caveat, now with numbers
(Part 8): microecs wins the **large-N regime** (N≳20k) — 3× over xecs on columnar, and it's the only
vectorized-numpy lib that can churn/migrate — but **loses below N≈10k** to leaner libs, is **mid on churn/migration**
(sparse-set snecs / object ecs-pattern beat it), and demands you **batch even random access** (`get_entity` in a
hot loop is a 400× trap). GPU/JAX engines still win at 10⁴–10⁶ + autodiff.

1. ✅ **[task 8](../todos/done/8-query-exclusion-none-of/TASK.md)** — query exclusion. **Done.**
2. ✅ **[task 9](../todos/done/9-tag-components/TASK.md)** — tag components. **Done.**
3. **README "Comparison / positioning" section** — *(not filed)* use the positioning paragraph above; name the
   real peers (xecs, manifoldx) + the adjacent clusters (Madrona, JAX ABM, mesa-frames); the cross-pool view as
   the headline. Fold in the Part 8 crossover chart + the copy-boundary mechanism.
4. ✅ **Benchmark** — **DONE (2026-07-15).** Superseded the single-number plan: not "1–2 orders over esper" but a
   full 6-workload × 5-lib × N-sweep with a verified crossover + mechanism (Part 8). Code + data:
   `examples/05-benchmark-workloads/` (README, `results.json`, `FINDINGS.md`, `probes/`). *Not yet done:*
   mesa-frames (Polars) and manifoldx (per-archetype) baselines — worthwhile future additions.
5. **Single-component get/set by id** — *(not filed; minor)* Part 6 #3. **Newly urgent:** Part 8 shows
   `get_entity(id).f` is a ~2600 ns/hit trap; a real `get_component`/`set_component` (or a public
   `entity_row(id)`) would give users an O(1) fast path and let them avoid the batching gymnastics.
6. **(designed, unbuilt) optional / `any_of`** — *(not filed)* the per-pool mechanism in Part 5 #1 if a use case
   appears.

---

## Part 8 — Empirical multi-workload benchmark (July 2026)

**Why?** The old benchmark measured **one** workload (columnar physics at N=100k) — microecs' best case — and
published a single "189× / 3.4×" headline. That's true but misleading: it doesn't tell a user what happens on the
workloads a *real* game/sim actually runs, or at the entity counts they actually use. This section answers that.

**What?** Six workloads × five libraries × an N-sweep (200 → 1,000,000), every result verified against a float64
reference (order-independent fingerprint, so a lib can't look fast by skipping work). Code, raw numbers, and
mechanism probes: **`examples/05-benchmark-workloads/`** (`README.md`, `results.json`, `FINDINGS.md`,
`probes/`, organized one folder per workload × one file per library). Env: numpy 2.5.1, Py 3.12, times =
min-over-reps of mean-over-30-frames, GC off.

Workloads (each = the real game system it models): **w1** physics (columnar integrate), **w2** bounce (columnar +
`np.where` branch), **w3** ai (per-entity health FSM), **w4** random (K read-modify-write by id/frame), **w5**
churn (spawn+FIFO-despawn/frame), **w6** mixed (physics+ai+targeted damage — a realistic steady frame), **w7**
migrate (component add/remove → archetype change). Libs: **microecs** (numpy SoA), **xecs** (Rust SoA, no
archetypes), **esper**/**snecs**/**ecs-pattern** (pure-python per-entity).

**Fairness:** both SoA libs (microecs, xecs) batch — w4/w6 use the *same* `col[rows]-=x` scatter for each; the
AoS libs loop with O(1) id lookups (their fast path). The naive microecs `get_entity` loop is quantified as "the
trap" (P3), not as microecs' number.

### The headline: there is no global winner — it flips by workload AND by N

```
fastest lib     N=200        1k          5k          20k        100k
w1 physics      xecs         xecs        xecs        microecs   microecs
w2 bounce       xecs         xecs        xecs        microecs   microecs
w3 ai           esper        xecs        xecs        microecs   microecs
w4 random       ecs-pattern  ecs-pattern microecs    microecs   microecs
w5 churn        ecs-pattern  ecs-pattern ecs-pattern ecs-pattern snecs
w6 mixed        esper        xecs        xecs        microecs   microecs
w7 migrate      snecs        snecs       microecs    microecs   snecs      (xecs, ecs-pattern: N/A)
```

microecs / fastest-competitor (>1 = microecs slower):

| workload | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| w1 physics | 4.10 | 3.25 | 1.52 | **0.56** | **0.32** |
| w3 ai | 6.11 | 3.01 | 1.64 | **0.96** | **0.66** |
| w4 random | 1.45 | 1.10 | **0.34** | **0.16** | **0.10** |
| w5 churn | 13.3 | 4.51 | 2.54 | 2.07 | 1.04 |
| w6 mixed | 4.45 | 3.32 | 1.72 | **0.85** | **0.58** |
| w7 migrate | 4.51 | 1.32 | **0.87** | **0.89** | 1.22 |

**Read it as:** microecs owns the **N≳20k regime** (columnar, scattered random access, and the realistic mixed
frame — all *faster* than the whole field), is competitive on migration, and only genuinely *loses* structural
churn. Below N≈10k it loses almost everything to the leaner libs — its per-op query/`_Field` overhead isn't
amortized yet.

### The columnar crossover and the "why is Rust slower?" answer

Columnar physics, ns/entity (microecs vs xecs, main sweep + large-N tail):

| N | 200 | 1k | 5k | 20k | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| microecs | 178 | 40 | 8.9 | **2.8** | **1.7** | **1.7** | **1.8** | **2.8** |
| xecs | 44 | 12 | 5.8 | 5.1 | 5.3 | 5.4 | 5.5 | 5.6 |

**Crossover ≈ N=10k.** xecs is *flat* at ~5–6 ns/e everywhere; microecs falls to ~1.7 ns/e as fixed overhead
amortizes. Why does a **pure-Python+numpy** lib beat a **Rust** lib 3× at scale? Verified with probes (P1–P3 in
FINDINGS):

- **Not a dtype artifact (P1).** numpy 2.x (NEP 50) keeps `Float32 * python_float` in float32; forcing float32 dt
  gives 0 speedup. Refuted.
- **It's the Rust↔numpy copy boundary (P2, the key finding).** `xecs.view.x * dt` returns a **plain numpy array** —
  xecs does its arithmetic *in numpy*, not in Rust; the Rust core is just storage. `.numpy()` is a **copy** (its
  own docstring; no zero-copy accessor exists), so every columnar op copies operand columns out of Rust and writes
  results back — ≈6 buffer copies/step, **5–7× raw in-place numpy**. microecs' `QueryResult`/`_Field` operate
  **in place on the very pool arrays** (verified zero-copy: `pool.py:67`, `query_result.py:106`, in-place write
  `query_result.py:55`). So microecs wins *because it has no FFI boundary* — the Rust core is a marshalling **tax**
  for CPU-vectorized work, not an asset. (Ruled out the `get_view()`-cost confound and the x/y-split confound: raw
  2-column numpy ≈ fused `(N,2)` numpy.)
- At small N, microecs' fixed per-op cost (query dict lookup + `_Field` alloc + per-pool python loop) exceeds
  xecs' leaner path → xecs wins < N≈10k.

### The random-access trap (P3) — and why it decides the mixed frame

`world.get_entity(id).f` in a hot loop costs **~2600 ns/hit** (Entity view + `__getattr__` dict lookup, flat in
N). The batched SoA idiom `col[rows] -= x` costs **6 ns/hit at 100k — 400× less.** This is not academic: with the
naive loop microecs was **4–5× slower than xecs on the mixed frame at every N**; with the (fair) scatter it
**wins the mixed frame above N≈20k**. The whole vectorization advantage lives or dies on whether the user batches
random access. Caveat: batching needs a static set / a row-map rebuilt after `update()` (pop-swap reorders rows) —
so a *churning* world with per-id edits is genuinely hard, and argues for the `get_component(id)` fast-path
follow-up (Part 7 #5).

### Capability gaps decide churn & migration (not just speed)

- **xecs cannot despawn OR migrate** — `Commands.spawn` only; fixed-capacity, per-component pools. It is
  **disqualified** from any birth/death or component-toggling sim, regardless of its columnar speed. This is the
  sharpest practical differentiator vs microecs.
- **ecs-pattern cannot migrate** — entities are fixed inheritance-classes; no runtime component add/remove.
- Only **microecs / esper / snecs** do component migration. **snecs (sparse-set) is the migration champ**;
  microecs is *mid* — the archetype pop-swap + whole-entity copy to a new pool (flagged in Part 1) is a real cost.
  Honest: archetype-SoA buys contiguous columns at the price of expensive structural change; that's the trade.

### Workload → real game/sim regimes (which of these matters?)

| workload | real systems | typical N | who wins there |
|---|---|--:|---|
| w1/w2/w3 columnar | particles/VFX, bullet-hell, boids, N-body, **UAV/vehicle integrators**, RL rollouts, ABM ticks | 1k–1M | xecs <10k, **microecs >10k** |
| w4 random | ARPG/MOBA damage, hitscan, targeted heals, net delta-apply | hits 1–100 / 100s–few-k live | ecs-pattern tiny-N, **microecs/xecs at scale** |
| w5 churn | bullet emitters, TD creep waves, spawn/despawn pools | births 10–1k/frame | ecs-pattern / snecs (**not xecs**) |
| w7 migrate | state tags (Alive↔Dead), buff/debuff add-remove | K/frame | snecs; microecs mid (**not xecs/ecs-pattern**) |

**Reality check:** most *games* live **below** the crossover (RTS units 200–2k, ARPG hundreds, roguelikes tens) —
that's xecs/ecs-pattern territory. microecs' large-N advantage is real only for **particle systems, big
bullet-hell, and agent-based / physics simulation at N≳10k** — which is exactly its stated niche. For robosim
itself (~2 robots + tens of world entities, N≈10²) microecs is three orders of magnitude below its own crossover;
here it's chosen for its **API + zero-copy state + dynamic structure**, not throughput, and at that N *all* libs
are sub-millisecond so it doesn't matter.

**Strategic verdict:** microecs' real audience is **vectorizable ABM / physics sim at scale**, and for that
audience the winning competitor is *microecs itself at large N* — xecs only wins the small-N/tight-loop regimes
this audience doesn't operate in, and forfeits churn and migration entirely. The one thing microecs users must
internalize: **batch everything, including random access.**

### Honest limitations of this benchmark (pre-empting the skeptic)

- **GC-off + min-over-reps flatters the object-per-entity libs** (a real frame budget cares about p99, not the
  min, and long GC-on sessions expose the object-churn pauses esper/snecs/ecs-pattern generate). A GC-on
  p50/p99 long-run variant would be the fairer "real session" number — worthwhile future work.
- **float32 (SoA) vs float64 (object libs)** is half the bytes — a real bandwidth edge for SoA at large N,
  independent of python overhead. microecs vs xecs is fair (both f32).
- Workloads use **1–2 archetypes** (except w7); heavy **archetype fragmentation** (many tiny pools, where
  `_Field`-concat-across-pools degrades) is untested and is microecs' theoretical soft spot.
- Verification is an **order-independent multiset** fingerprint: catches skipped/wrong-magnitude work, not a
  which-entity permutation. Adequate for "didn't cheat," not a per-entity correctness proof.
- min-16 touch floor → small-N w4/w5 touch ~8%/frame, heavier than the advertised ~2%/1%.

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

**Part 8 benchmark** — `examples/05-benchmark-workloads/` (harness, per-workload adapters, `results.json`,
`FINDINGS.md`, `probes/xecs_dtype.py` / `probes/boundary.py` / `probes/microecs_random.py`). xecs numpy-boundary confirmed in its
installed source `xecs/_internal/vec2.py` (`__iadd__`/`__mul__` do `self.numpy()` + `np.*` + write-back) and
`Float32.numpy.__doc__` ("Copy the elements into a NumPy array").

**Method note:** landscape facts re-verified **2026-07-15** via GitHub API / PyPI JSON / docs (prev June 2026);
**no material drift** — xecs still dead (3★, last commit 2023-10-05, no despawn/migrate), snecs dead, esper 688→695★
& ecs-pattern (`ikvk/ecs_pattern`, 54★) both lightly-active but unvectorized, manifoldx still renderer-locked with
no cross-archetype write, and **still no JAX/torch-backed Python ECS *library***. Part 8 numbers are this-machine
point-in-time (numpy 2.5.1, Py 3.12); the *ratios/crossover* are the portable result. Author-run third-party
benchmarks (mesa-frames 10×, AMBER 1.7–93×) remain `[claim]`. "Bevy removing sparse-set" is still a proposal.
