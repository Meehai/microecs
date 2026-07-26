# Benchmarks

Two benchmarks: microecs against plain OOP (why vectorize at all), and microecs against other Python ECS libraries (how it compares). Both verify every method against a numpy reference before timing it.

## Benchmark 1: microecs vs OOP on a simple physics step

We run the same physics step `pos += vel*dt` over N=100k entities split across 2 pools in various ways (ECS or OOP). All methods are verified to produce the identical result. Reproduce with `python examples/04-benchmark-ecs-vs-oop.py` (it prints `{mode: avg_seconds_per_step}`).

| pattern | ns/entity | vs OOP-scalar |
|---|---:|---:|
| `micro-ecs-pool-vectorized` — `for pool: pool.f[:] = pool.f + …` | 0.84 | **64× faster** |
| `micro-ecs-vectorized` — `qr.f = qr.f + …` (the `Field`) | 1.34 | **40× faster** |
| **`oop-scalar`** — `for o: o.x += o.vx*dt` (python floats) | 54 | 1× (baseline) |
| `oop-numpy` — objects holding `(2,)` numpy arrays | 719 | 13× slower |
| `micro-ecs-zip-rows` — `for p, v in zip(qr.pos, qr.vel)` | 838 | 16× slower |
| `micro-ecs-pool-loop` — `for pool: for i: pool.f[i]` | 1023 | 19× slower |
| `micro-ecs-get-entity` — `world.get_entity(eid)` per entity | 2163 | 40× slower |

Three things to take from it:

1. **Vectorized wins big.** Batched ops (`Field` or per-pool) run at ~1 ns/entity — **40–64×
   faster** than the *fastest* OOP loop. A data-parallel branch stays in that regime: w2 bounce below
   (`np.where` wall reflection) runs at 4.0 ns/entity against 179 for the fastest per-entity library.
2. **Per-entity loops are a cliff, not a tie.** Every per-entity microecs path is **16–40× slower**
   than idiomatic float-based OOP — because microecs is numpy-backed, so a per-entity step pays
   numpy's tiny-array overhead (`oop-numpy` shows the same ~13× tax). One unavoidable per-entity
   pass (~840 ns/entity) costs ~600× a vectorized op (~1.3 ns) and will dominate the frame.
3. **If you must loop, loop right.** `zip`-rows (16×) < pool-loop (19×) < `get_entity` (40×).
   For random single-entity access use `world.get_entity(qr.entity_ids[i])` — there's no `qr.f[i]`
   shortcut (the entity axis is off-limits on a query; it raises).

### What one entity operation costs

The `Entity` handle is the OOP-shaped escape hatch, and the table above prices the whole loop. Per
single operation (N=20k, ns, this machine):

| operation | ns | what it is |
|---|--:|---|
| `pool.data['position'][ix]` | 110 | the raw numpy floor — no ECS at all |
| `world.get_entity(eid)` | 74 | a dict lookup; the `Entity` object is cached per id |
| `ent.position` (read) | 320 | locate the pool + check the field exists + index |
| `ent.position = v` (write) | 471 | same, then write the row |
| `ent.set_data(position=v)` | 610 | the kwargs form of the same write |
| `ent.set_data(position=v, velocity=v)` | 1433 | two fields, with the shape/dtype pre-checks |
| `e.position = e.position + e.velocity*dt` | 2018 | read + read + write = one "entity tick" |

Read it as **~0.3–0.5 µs per field touch**. That is fine for tens of entities (a UAV, a player, a
handful of pickups) and a disaster for thousands — 2 µs × 10k entities is a 20 ms frame. Same rule as
above: address *one* entity by id with `Entity`, address *many* with a query.

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
Python 3.12; times are min-over-reps of the mean over 30 frames, GC off. Run 2026-07-26.

**How precise is this?** Absolute times drift **±10–25% between runs** on the same machine — re-running
the suite 11 days apart moved *every* library, including the six microecs cannot affect, by up to +12%
median. So read the **ratios within a run**, the winner map and the crossover; treat a single cell as
±20%. Where a headline depends on one cell, it was re-measured with 8 reps (noted inline).

### The seven workloads — what each models and why it's here

Each workload is a real game/sim *system*, chosen to stress a different data-access shape so the
suite covers what a real frame actually does — not just microecs' best case.

| id | what it computes | real systems | shape it stresses | why it's in the suite |
|---|---|---|---|---|
| **w1 physics** | `vel += acc·dt; pos += vel·dt` over all | particles, bullets, boids, UAV/vehicle integrators, RL rollouts | pure columnar (vectorizable) | the canonical bulk-numeric case — microecs' home turf; the "does vectorizing pay?" baseline |
| **w2 bounce** | w1 + reflect at walls via `np.where` | anything with a data-parallel branch | vectorizable-with-branch | does a branch stay in numpy (`np.where`) or force a per-entity fallback? |
| **w3 ai** | per-entity health FSM: drain → die → respawn | NPC/AI ticks, status effects | branchy row logic (per-entity `if`) | the case per-entity libs *should* win — tests microecs' masked-update idiom against real loops |
| **w4 random** | read-modify-write K entities picked by id/frame | targeted damage, hitscan, heals, net delta-apply | random access by id | the case SoA is *supposed* to be bad at — tests batched column scatter vs per-entity lookup |
| **w5 churn** | spawn B + FIFO-despawn B per frame + integrate | bullet-hell emitters, TD creep waves, spawn/die pools | structural churn (births/deaths) | microecs' weak spot — and takeaway 3 shows the cost is mostly spawn-path validation, not the pop-swap; also a capability test (xecs can't despawn) |
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
| w3 ai | esper | xecs | **microecs** | **microecs** | **microecs** |
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
| w1 physics | 0.71 (xecs) | 0.84 (xecs) | **1.68 (xecs)** | **3.35 (xecs)** | **2.51 (xecs)** |
| w2 bounce | 0.71 (xecs) | 0.78 (xecs) | **1.19 (xecs)** | **2.13 (xecs)** | **2.42 (xecs)** |
| w3 ai | 0.56 (esper) | 0.99 (xecs) | **1.32 (xecs)** | **1.65 (xecs)** | **1.52 (xecs)** |
| w4 random | 0.81 (ecs-pattern) | 0.88 (ecs-pattern) | **2.96 (ecs-pattern)** | **5.82 (xecs)** | **9.91 (xecs)** |
| w5 churn | 0.08 (ecs-pattern) | 0.22 (ecs-pattern) | 0.36 (ecs-pattern) | 0.46 (ecs-pattern) | 0.82 (entt) |
| w6 mixed | 0.33 (esper) | 0.44 (xecs) | 0.77 (xecs) | **1.60 (xecs)** | **1.80 (xecs)** |
| w7 migrate | 0.24 (snecs) | 0.83 (entt) | **1.02 (entt)** | **1.07 (entt)** | 0.77 (entt) |

Note who the rival *is*: on every columnar/branchy/random workload it is another **vectorized** library
(xecs), never a native per-entity one. The only cells microecs loses to a C/C++ engine are the two
structural ones (w5 churn, w7 migrate) — which is exactly Experiment 3's point.

Two cells are ties rather than results: w3 ai @1k is 0.99 (0.0190 vs 0.0188 ms — it was microecs' by
the same margin last run), and w7 migrate @5k is 1.02. The w1 @100k cell reads 2.51 here because that
run's min-of-2 for xecs landed low; an 8-rep re-measure of that one cell gives **3.3× on medians,
3.7× on mins** (w2: 2.5–2.7×). Nothing else moved from the previous run.

### Experiment 2 — columnar scaling to 1M (microecs vs xecs)

Past 100k only the two vectorized libraries stay in the race, so experiment 2 pits just those two on
the columnar workloads. xecs stays flat at its copy-bound rate; microecs holds near its in-place
floor. Columnar step, **ns/entity per frame** (lower is better):

| N | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|
| w1 physics — microecs | **1.70ns** | **1.62ns** | **1.72ns** | **2.66ns** |
| w1 physics — xecs | 4.26ns | 5.34ns | 5.42ns | 5.72ns |
| w2 bounce — microecs | **3.99ns** | **4.11ns** | **5.20ns** | **6.91ns** |
| w2 bounce — xecs | 9.65ns | 10.20ns | 11.34ns | 11.99ns |

At **1M entities** a physics frame is **2.7 ms (microecs) vs 5.7 ms (xecs)**, a bounce frame
**6.9 ms vs 12.0 ms** — a steady ~2× lead. It holds because microecs mutates the pool arrays in
place while xecs copies ~6 buffers across the Rust↔numpy boundary every step (the copy-boundary
mechanism explained in the takeaways below). microecs is the only library in the suite that steps 1M
entities per frame in low-single-digit milliseconds with no GPU and no compile step.

(The 100k column is the main-matrix cell — min-of-2 reps, and xecs' 4.26 is the low end of its spread;
8 reps put it at 5.2–5.4ns, i.e. the ~3.3× lead the 200k/500k columns show. microecs is the noisier of
the two here: 1.4–1.7ns across 8 reps versus xecs' 5.21–5.38.)

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
| w1 physics | 3.43 | 3.13 | 2.57 | 2.72 | 2.41 |
| w2 bounce | 3.35 | 3.04 | 2.66 | 2.64 | 2.32 |
| w3 ai | 3.15 | 3.23 | 3.00 | 2.94 | 2.95 |
| w4 random | 2.21 | 2.41 | 2.89 | 2.89 | 2.29 |
| w5 churn | 2.04 | 2.30 | 2.29 | 1.89 | **0.90** |
| w7 migrate | 1.12 | **0.94** | **0.92** | **0.97** | **0.90** |

**A native core costs ~2.3–3.4× on field arithmetic and only pays off on structural work.** Where the
work is per-entity `p.x += p.vx * dt`, you pay Python's loop *plus* a boundary crossing per component
access — so binding a world-class C++ engine lands you *behind* plain esper. Where the work **is** the
data-structure operation (spawn/despawn, add/remove component), the whole thing happens in C++ and EnTT
becomes the fastest library in the suite.

Against microecs at N=100,000, ns/entity/frame:

| workload | microecs | entt (C++) | flecs (C) | esper (pure py) | microecs vs entt |
|---|--:|--:|--:|--:|--:|
| w1 physics | **1.7** | 255.2 | 829.5 | 105.9 | **150× faster** |
| w2 bounce | **4.0** | 414.9 | 1392.9 | 179.2 | **104× faster** |
| w3 ai | **4.4** | 170.0 | 497.3 | 68.1 | **39× faster** |
| w4 random | **0.1** | 7.2 | 27.6 | 14.6 | **79× faster** |
| w5 churn | 219.2 | **180.0** | 554.1 | 414.8 | 1.2× slower |
| w6 mixed | **7.5** | 498.5 | 1500.3 | 249.8 | **66× faster** |
| w7 migrate | 227.3 | **175.9** | 620.7 | 477.7 | 1.3× slower |

This is the same mechanism that makes microecs beat Rust-backed xecs (takeaway 2 below), pushed to its
conclusion across three different native runtimes. **In Python, the only thing that buys per-entity
arithmetic throughput is vectorization.** A faster storage engine cannot — it can only make structural
operations cheaper, and it charges you a boundary crossing for everything else.

One caveat, stated plainly: the *engines* here are world-class, the *Python bindings* are not
maintained — PyEnTT is a 1★ repo archived the day after it was created (2026-03-05); pyflecs11 is 2★
with no commit since 2025-08-21. A better-engineered binding would shave constant factors. It cannot
remove the per-access boundary crossing or the Python-level loop, and that is what the ~2.3–3.4× above
is made of.

Five things to take from the three experiments:

1. **microecs owns the large-N regime.** On the columnar tail it runs at **~1.7 ns/entity** and holds
   near that to 1M; on random access and the mixed frame it wins from N≈5–20k up (and by 100k it's
   ~2.5–10× the runner-up). For N≥20k simulation, the numpy-SoA design is the fastest thing here.
2. **The columnar crossover is ≈ N=1.5–3k.** Below it, xecs (Rust SoA) still wins, but only by
   ~1.2–1.4×. Above it microecs wins by ~1.7× at 5k, ~3× at 20k–500k, ~2.1× at 1M. The surprise — a
   pure-Python+numpy lib beating a Rust lib at scale — is real and mechanistic: xecs does its
   arithmetic *in numpy anyway* (`view.x * dt` returns an ndarray) and `.numpy()` is a **copy**, so
   every columnar op copies buffers out of Rust and back (~4–11× raw numpy). microecs mutates the pool
   arrays **in place, zero-copy** — it wins *because it has no FFI boundary*. (Confirmed in
   `examples/05-benchmark-workloads/probes/`.)
3. **The one genuine loss — churn — is a validation cost, not a layout cost.** w5 churn is microecs'
   weakest workload at every N (0.08× ecs-pattern at 200, 0.82× EnTT at 100k). Splitting one
   spawn+despawn pair (12.6 µs) shows why: the **storage** work — SoA insert plus the archetype
   pop-swap usually blamed for this — is 43%, while **validation is 32%, and half of that is done
   twice** (`World.add_entity` validates, then `CommandBuffer.append` validates the same command
   again). ~2 µs per spawn, 16% of a churn pair, is redundant. That is microecs' largest available win
   and it is bookkeeping, not architecture.
4. **Batch random access; capability gaps decide churn/migration.** `get_entity(id).f` in a hot loop
   is a ~2900 ns/hit trap — up to **483× slower** than a batched `col[rows] -= …` scatter (6 ns/hit),
   which is what the benchmark uses. And xecs can't despawn *or* migrate; ecs-pattern can't migrate —
   for spawn/die churn and buff-on/off migration the qualifying set is microecs/entt/flecs/esper/snecs.
   If your update loop *isn't* vectorizable and stays small, a per-entity python ECS is simpler and
   faster — see the microbenchmark above.
5. **Binding a native ECS is not the shortcut it looks like.** Experiment 3: EnTT and flecs are
   ~2.3–3.4× *slower* than plain esper on every arithmetic workload, and 39–490× slower than microecs.
   They win only churn and migration, where the work happens entirely inside the native engine. The
   axis that matters in Python is vectorized-vs-per-entity, **not** native-vs-interpreted.
