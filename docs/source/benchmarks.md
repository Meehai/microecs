# Benchmarks

Two benchmarks: microecs against plain OOP (why vectorize at all), and microecs against other Python ECS libraries (how it compares). Both verify every method against a numpy reference before timing it.

## Benchmark 1: microecs vs OOP on a simple physics step

We run the same physics step `pos += vel*dt` over N=100k entities split across 2 pools in various ways (ECS or OOP). All methods are verified to produce the identical result. Reproduce with `python examples/04-benchmark-ecs-vs-oop.py` (it prints `{mode: avg_seconds_per_step}`).

| pattern | ns/entity | vs OOP-scalar |
|---|---:|---:|
| `micro-ecs-pool-vectorized` — `for pool: pool.f[:] = pool.f + …` | 0.9 | **52× faster** |
| `micro-ecs-vectorized` — `qr.f = qr.f + …` (the `Field`) | 1.8 | **27× faster** |
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

## Benchmark 2: microecs vs other Python ECS libraries

Not one workload — **seven** × **seven** libraries × an N-sweep (200 → 1,000,000), every result
verified against a float64 reference. The field spans all three ways to build an ECS reachable from
Python: numpy-vectorized (microecs), a native core with a native columnar store (xecs), a native core
with per-entity Python components (**EnTT**, **flecs**), and pure Python (esper, snecs, ecs-pattern).
Full setup, fairness notes, and raw data in `examples/05-benchmark-workloads/` (one folder per workload;
`FINDINGS.md` has every number; run `./run_benchmark.sh` to regenerate). Environment: numpy 2.5.1,
Python 3.12; times are min-over-reps of the mean over 30 frames, GC off.

### The seven workloads — what each models and why it's here

Each workload is a real game/sim *system*, chosen to stress a different data-access shape so the
suite covers what a real frame actually does — not just microecs' best case.

| id | what it computes | real systems | shape it stresses | why it's in the suite |
|---|---|---|---|---|
| **w1 physics** | `vel += acc·dt; pos += vel·dt` over all | particles, bullets, boids, UAV/vehicle integrators, RL rollouts | pure columnar (vectorizable) | the canonical bulk-numeric case — microecs' home turf; the "does vectorizing pay?" baseline |
| **w2 bounce** | w1 + reflect at walls via `np.where` | anything with a data-parallel branch | vectorizable-with-branch | does a branch stay in numpy (`np.where`) or force a per-entity fallback? |
| **w3 ai** | per-entity health FSM: drain → die → respawn | NPC/AI ticks, status effects | branchy row logic (per-entity `if`) | the case per-entity libs *should* win — tests microecs' masked-update idiom against real loops |
| **w4 random** | read-modify-write K entities picked by id/frame | targeted damage, hitscan, heals, net delta-apply | random access by id | the case SoA is *supposed* to be bad at — tests batched column scatter vs per-entity lookup |
| **w5 churn** | spawn B + FIFO-despawn B per frame + integrate | bullet-hell emitters, TD creep waves, spawn/die pools | structural churn (births/deaths) | archetype-SoA's known weak spot (pop-swap + realloc); also a capability test (xecs can't despawn) |
| **w6 mixed** | w1 physics + w3 ai + K targeted hits, fixed set | a realistic steady-state game frame | composite (columnar + branchy + random) | the most representative "real frame" — several systems per tick, not one microbenchmark |
| **w7 migrate** | integrate all + rolling component add/remove | buff/debuff on-off, state tags (Alive↔Dead) | archetype migration (whole-entity copy) | the migration tax; capability test (xecs and ecs-pattern can't migrate at all) |

`k = max(16, n//50)` entities touched/frame (w4/w6); `b = max(16, n//100)` churned/frame (w5);
`2·max(4, n//200)` migrations/frame (w7).

### The seven libraries

| library | core | model | how each workload is driven (idiomatic, best-case) |
|---|---|---|---|
| **microecs** | pure python | numpy SoA by archetype | columnar via `QueryResult` write-through; branch via `np.where`; random via batched column scatter; churn/migrate via `add/remove_entity` + `update()` |
| **xecs** | **Rust** (pyo3) | SoA, per-component columns, no archetypes | columnar via in-place `view.x += …`; random scatters a column; **no despawn/migration → w5/w7 N/A** |
| **entt** | **C++** (nanobind) | [EnTT](https://github.com/skypjack/entt) sparse-set; components are python objects | `registry.view(A, B)` yields `(entity, *comps)`; `registry.get(e, T)`; `create`/`destroy`/`emplace`/`remove` |
| **flecs** | **C** (pybind11) | [flecs](https://www.flecs.dev) archetype tables; components are python objects | `world.query(A, B)`; `entity.get/set/remove`; `entity.destroy()` |
| **esper** | pure python | per-entity objects | `get_components` loop; `if` per entity; O(1) `component_for_entity(id)`; `create/delete_entity` |
| **snecs** | pure python | sparse-set | compiled `Query` loop; per-entity `if`; `entity_component(id)`; sparse-set migration (the migration champ) |
| **ecs-pattern** | pure python | dataclass AoS | `get_with_component` loop; direct object ref (fastest random access); **fixed inheritance classes → w7 N/A** |

Getting the two native engines to run at all is part of the finding — neither is a `pip install`:
**PyEnTT** publishes no Linux wheel *and* no sdist (build from the repo with its `entt` submodule);
the PyPI **flecs** Linux wheel is cp39-only (build from sdist), and that binding **segfaults** when a
`Query` is iterated twice without `.reset()` in between, and again at interpreter shutdown. The
adapters call `.reset()` before every iteration. Install notes are in `requirements.txt`.

### Experiment 1 — the full field: N=200 → 100k, all seven libraries

There is no single winner — the fastest library flips by workload *and* by N:

| fastest library | N=200 | 1k | 5k | 20k | 100k |
|---|---|---|---|---|---|
| w1 physics | xecs | xecs | **microecs** | **microecs** | **microecs** |
| w2 bounce | xecs | xecs | **microecs** | **microecs** | **microecs** |
| w3 ai | esper | **microecs** | **microecs** | **microecs** | **microecs** |
| w4 random | ecs-pattern | ecs-pattern | **microecs** | **microecs** | **microecs** |
| w5 churn | ecs-pattern | ecs-pattern | ecs-pattern | ecs-pattern | entt |
| w6 mixed | esper | xecs | xecs | **microecs** | **microecs** |
| w7 migrate | snecs | entt | **microecs** | **microecs** | entt |

*w7 migrate: xecs and ecs-pattern can't migrate (N/A). Every other library does all seven.*

**How close is the race, and against whom?** Each cell below is **how many times faster microecs is**
than the fastest other library (that library's time ÷ microecs's), with that rival named. `>1` →
microecs is faster; `<1` → slower (by `1/x`). So `3.31 (xecs)` = microecs **3.3× faster** than xecs;
`0.79 (xecs)` = microecs **1.27× slower** than xecs; `0.94 (snecs)` = a near-tie, snecs just ahead.
The named lib is microecs's nearest rival — the one it beats, or the one beating it. **Bold** =
microecs wins that cell (ratio `>1`).

| workload | N=200 | 1k | 5k | 20k | 100k |
|---|---|---|---|---|---|
| w1 physics | 0.78 (xecs) | 0.92 (xecs) | **1.76 (xecs)** | **3.10 (xecs)** | **3.70 (xecs)** |
| w2 bounce | 0.66 (xecs) | 0.79 (xecs) | **1.35 (xecs)** | **2.53 (xecs)** | **2.57 (xecs)** |
| w3 ai | 0.59 (esper) | **1.09 (xecs)** | **1.26 (xecs)** | **1.84 (xecs)** | **1.51 (xecs)** |
| w4 random | 0.72 (ecs-pattern) | 0.95 (ecs-pattern) | **3.05 (ecs-pattern)** | **6.31 (xecs)** | **10.69 (xecs)** |
| w5 churn | 0.09 (ecs-pattern) | 0.24 (ecs-pattern) | 0.40 (ecs-pattern) | 0.49 (ecs-pattern) | 0.84 (entt) |
| w6 mixed | 0.35 (esper) | 0.43 (xecs) | 0.81 (xecs) | **1.45 (xecs)** | **1.87 (xecs)** |
| w7 migrate | 0.24 (snecs) | 0.86 (entt) | **1.14 (snecs)** | **1.08 (entt)** | 0.80 (entt) |

Note who the rival *is*: on every columnar/branchy/random workload it is another **vectorized** library
(xecs), never a native per-entity one. The only cells microecs loses to a C/C++ engine are the two
structural ones (w5 churn, w7 migrate) — which is exactly Experiment 3's point.

### Experiment 2 — columnar scaling to 1M (microecs vs xecs)

Past 100k only the two vectorized libraries stay in the race, so experiment 2 pits just those two on
the columnar workloads. xecs stays flat at its copy-bound rate; microecs holds near its in-place
floor. Columnar step, **ns/entity per frame** (lower is better):

| N | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|
| w1 physics — microecs | **1.60ns** | **1.52ns** | **1.67ns** | **2.74ns** |
| w1 physics — xecs | 5.91ns | 5.17ns | 5.08ns | 5.55ns |
| w2 bounce — microecs | **3.96ns** | **4.00ns** | **5.05ns** | **7.03ns** |
| w2 bounce — xecs | 10.16ns | 10.12ns | 10.29ns | 11.95ns |

At **1M entities** a physics frame is **2.7 ms (microecs) vs 5.6 ms (xecs)**, a bounce frame
**7.0 ms vs 12.0 ms** — a steady ~2× lead. It holds because microecs mutates the pool arrays in
place while xecs copies ~6 buffers across the Rust↔numpy boundary every step (the copy-boundary
mechanism explained in the takeaways below). microecs is the only library in the suite that steps 1M
entities per frame in low-single-digit milliseconds with no GPU and no compile step.

### Experiment 3 — does a C/C++ core help? (EnTT and flecs vs pure Python)

The obvious objection to a pure-Python ECS is "just bind a real one." So we did: **EnTT** (the C++
sparse-set engine behind a lot of shipped games) and **flecs** (the C archetype engine). Both store
entities in native memory; both hand back Python objects as components, so the *system body* is still
a Python loop.

That makes the comparison clean: **entt/flecs and esper/snecs/ecs-pattern do identical Python work per
entity — only the storage engine differs.** Ratio = best(entt, flecs) ÷ best(esper, snecs, ecs-pattern);
`>1` means the native core is **slower**:

| workload | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| w1 physics | 3.16 | 2.99 | 2.83 | 2.72 | 2.44 |
| w2 bounce | 2.66 | 2.78 | 2.79 | 2.86 | 2.36 |
| w3 ai | 3.54 | 3.21 | 3.01 | 2.81 | 2.62 |
| w4 random | 2.43 | 2.34 | 2.64 | 2.78 | 2.40 |
| w5 churn | 1.96 | 2.18 | 2.16 | 1.69 | **0.92** |
| w7 migrate | 1.07 | 1.00 | 1.01 | **0.94** | **0.89** |

**A native core costs ~2.4–2.6× on field arithmetic and only pays off on structural work.** Where the
work is per-entity `p.x += p.vx * dt`, you pay Python's loop *plus* a boundary crossing per component
access — so binding a world-class C++ engine lands you *behind* plain esper. Where the work **is** the
data-structure operation (spawn/despawn, add/remove component), the whole thing happens in C++ and EnTT
becomes the fastest library in the suite.

Against microecs at N=100,000, ns/entity/frame:

| workload | microecs | entt (C++) | flecs (C) | esper (pure py) | microecs vs entt |
|---|--:|--:|--:|--:|--:|
| w1 physics | **1.6** | 277.9 | 873.6 | 113.7 | **174× faster** |
| w2 bounce | **4.0** | 455.5 | 1405.0 | 192.9 | **115× faster** |
| w3 ai | **4.6** | 156.7 | 472.8 | 69.5 | **34× faster** |
| w4 random | **0.1** | 6.4 | 27.2 | 13.2 | **67× faster** |
| w5 churn | 201.7 | **169.3** | 522.8 | 386.4 | 1.2× slower |
| w6 mixed | **6.6** | 446.1 | 1347.6 | 243.5 | **68× faster** |
| w7 migrate | 204.9 | **163.2** | 539.8 | 468.7 | 1.3× slower |

This is the same mechanism that makes microecs beat Rust-backed xecs (takeaway 2 below), pushed to its
conclusion across three different native runtimes. **In Python, the only thing that buys per-entity
arithmetic throughput is vectorization.** A faster storage engine cannot — it can only make structural
operations cheaper, and it charges you a boundary crossing for everything else.

Five things to take from the three experiments:

1. **microecs owns the large-N regime.** On the columnar tail it runs at **~1.5 ns/entity** and holds
   flat to 1M; on random access and the mixed frame it wins from N≈5–20k up (and by 100k it's
   ~3–10× the runner-up). For N≥20k simulation, the numpy-SoA design is the fastest thing here.
2. **The columnar crossover is now ≈ N=1.5–3k** (was ~10k — the `microecs #26` low-N optimization
   moved it down ~5×). Below it, xecs (Rust SoA) still wins, but by only ~1.3–1.4× now, not ~4×.
   Above it microecs wins ~3×. The surprise — a pure-Python+numpy lib beating a Rust lib at scale —
   is real and mechanistic: xecs does its arithmetic *in numpy anyway* (`view.x * dt` returns an
   ndarray) and `.numpy()` is a **copy**, so every columnar op copies buffers out of Rust and back
   (~4–8× raw numpy). microecs mutates the pool arrays **in place, zero-copy** — it wins *because it
   has no FFI boundary*. (Confirmed in `examples/05-benchmark-workloads/probes/`.)
3. **What `#26` bought:** a single-archetype query now returns a thin numpy view instead of building
   a `QueryResult.Field` per op, so the fixed per-op cost that used to sink low-N is gone. microecs
   got faster in *every* cell; w3 ai @1k, w1/w2 @5k all flipped from xecs to microecs.
4. **Batch random access; capability gaps decide churn/migration.** `get_entity(id).f` in a hot loop
   is a ~2900 ns/hit trap — up to **459× slower** than a batched `col[rows] -= …` scatter (6 ns/hit),
   which is what the benchmark uses. And xecs can't despawn *or* migrate; ecs-pattern can't migrate —
   for spawn/die churn and buff-on/off migration the qualifying set is microecs/entt/flecs/esper/snecs.
   Structural **churn** (w5) is microecs' one genuine loss at every N: below 100k to ecs-pattern, at
   100k to EnTT. If your update loop *isn't* vectorizable and stays small, a per-entity python ECS is
   simpler and faster — see the microbenchmark above.
5. **Binding a native ECS is not the shortcut it looks like.** Experiment 3: EnTT and flecs are
   ~2.4–2.6× *slower* than plain esper on every arithmetic workload, and 34–547× slower than microecs.
   They win only churn and migration, where the work happens entirely inside the native engine. The
   axis that matters in Python is vectorized-vs-per-entity, **not** native-vs-interpreted.
