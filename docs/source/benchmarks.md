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
| `micro-ecs-get-entity` — `world.get_entity(eid)` per entity | 1560 | 29× slower |

Three things to take from it:

1. **Vectorized wins big.** Batched ops (`Field` or per-pool) run at ~1 ns/entity — **40–64×
   faster** than the *fastest* OOP loop. A data-parallel branch stays in that regime: w2 bounce below
   (`np.where` wall reflection) runs at 4.0 ns/entity against 179 for the fastest per-entity library.
2. **Per-entity loops are a cliff, not a tie.** Every per-entity microecs path is **16–29× slower**
   than idiomatic float-based OOP — because microecs is numpy-backed, so a per-entity step pays
   numpy's tiny-array overhead (`oop-numpy` shows the same ~13× tax). One unavoidable per-entity
   pass (~840 ns/entity) costs ~600× a vectorized op (~1.3 ns) and will dominate the frame.
3. **If you must loop, loop right.** `zip`-rows (16×) < pool-loop (19×) < `get_entity` (29×).
   For random single-entity access use `world.get_entity(qr.entity_ids[i])` — there's no `qr.f[i]`
   shortcut (the entity axis is off-limits on a query; it raises).
   **~19× is the floor for any id-addressed access, not 16×** — `zip` uses numpy's array iterator
   (718 ns/row) while any indexed access pays `col[i]` getitem (885 ns/row) before microecs does
   anything at all. `pool-loop` already sits on that floor; `get_entity` is now ~1.5× off it.

### What one entity operation costs

The `Entity` handle is the OOP-shaped escape hatch, and the table above prices the whole loop. Per
single operation (N=20k, ns, this machine; median of 3 interleaved runs of
`test/manual/get-entity-perf/per_operation.py`):

| operation | ns | ×floor | what it is |
|---|--:|--:|---|
| `pool.data['position'][ix]` | 89 | 1.0 | the raw numpy floor — no ECS at all |
| `world.get_entity(eid)` | 39 | 0.4 | a dict lookup; the `Entity` object is cached per id |
| `ent.position` (read) | 172 | 1.9 | resolve `(pool, row)` by id, then index the column |
| `ent.position = v` (write) | 247 | 2.8 | same, then write the row |
| `ent.set_data(position=v)` | 505 | 5.7 | the kwargs form of the same write |
| `ent.set_data(position=v, velocity=v)` | 1486 | 16.7 | two fields, with the shape/dtype pre-checks |
| `e.position = e.position + e.velocity*dt` | 1521 | 17.1 | read + read + write = one "entity tick" |

Read it as **~0.2–0.25 µs per field touch** — i.e. **~2× the raw numpy floor**, which is about as
close as a python-level handle gets. That is fine for tens of entities (a UAV, a player, a handful of
pickups) and still a problem for thousands: 1.5 µs × 10k entities is a 15 ms frame. Same rule as
above: address *one* entity by id with `Entity`, address *many* with a query.

`set_data(f=v)` remains **~2× the cost of `e.f = v`** for the same effect — it pays a kwargs dict plus
the multi-name `issuperset` check. Use it when you genuinely set several fields at once.

**Rule of thumb:** keep systems vectorized and push branches into `np.where` / `np.clip`. If a
workload is *irreducibly* per-entity (data-dependent control flow), plain python objects beat
microecs ~15× — use them there. microecs is the right tool for **vectorizable** simulation.

### What a query costs

Everything above prices the *step*. Building the `QueryResult` is a separate cost, and it is flat in N —
a query is O(pools), not O(entities). Cold means the cache was dropped, which happens on any tick that had a
structural change (spawn, despawn, add/remove component); a warm query is a dict hit.

| N | cold query | `+ len(qr)` | `+ qr.entity_ids` |
|---|--:|--:|--:|
| 1 000 | 4.2 µs | 4.7 µs | 28.5 µs |
| 10 000 | 4.1 µs | 4.8 µs | 219.6 µs |
| 100 000 | 3.6 µs | 4.4 µs | **2133 µs** |

**`qr.entity_ids` is the one part that scales with N**, so it is built lazily on first read. Ask for it and you
pay for it — at N=100k that is ~500× the rest of the query put together. A system that iterates fields
(`qr.position += qr.velocity * dt`) never touches it; one that goes by id (`world.get_entity(qr.entity_ids[i])`)
pays it every tick the cache was dropped. `len(qr)` counts pools and does not materialize it.

## Benchmark 2: microecs vs other Python ECS libraries

Not one workload — **seven** × **seven** libraries × an N-sweep (200 → 1,000,000), every result
verified against a float64 reference. The field spans all three ways to build an ECS reachable from
Python: numpy-vectorized (microecs), a native core with a native columnar store (xecs), a native core
with per-entity Python components (**EnTT**, **flecs**), and pure Python (esper, snecs, ecs-pattern).
Full setup, fairness notes, and raw data in `examples/05-benchmark-workloads/` (one folder per workload;
`FINDINGS.md` has every number; run `./run_benchmark.sh` to regenerate). Environment: numpy 2.5.1,
Python 3.12; times are min-over-reps of the mean over 30 frames, GC off. Run 2026-07-27 (microecs with
**#44 + #49** merged); baseline run 2026-07-26 (`975097c`, neither change).

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
| w3 ai | ecs-pattern | **microecs** | **microecs** | **microecs** | **microecs** |
| w4 random | ecs-pattern | ecs-pattern | **microecs** | **microecs** | **microecs** |
| w5 churn | ecs-pattern | ecs-pattern | ecs-pattern | ecs-pattern | **microecs** |
| w6 mixed | esper | xecs | xecs | **microecs** | **microecs** |
| w7 migrate | snecs | **microecs** | **microecs** | **microecs** | **microecs** |

*w7 migrate: xecs and ecs-pattern can't migrate (N/A). Every other library does all seven.*

**How close is the race, and against whom?** Each cell below is **how many times faster microecs is**
than the fastest other library (that library's time ÷ microecs's), with that rival named. `>1` →
microecs is faster; `<1` → slower (by `1/x`). So `3.31 (xecs)` = microecs **3.3× faster** than xecs;
`0.79 (xecs)` = microecs **1.27× slower** than xecs; `0.94 (snecs)` = a near-tie, snecs just ahead.
The named lib is microecs's nearest rival — the one it beats, or the one beating it. **Bold** =
microecs wins that cell (ratio `>1`).

| workload | N=200 | 1k | 5k | 20k | 100k |
|---|---|---|---|---|---|
| w1 physics | 0.90 (xecs) | 0.90 (xecs) | **1.50 (xecs)** | **3.15 (xecs)** | **3.10 (xecs)** |
| w2 bounce | 0.76 (xecs) | 0.69 (xecs) | **1.28 (xecs)** | **1.82 (xecs)** | **2.63 (xecs)** |
| w3 ai | 0.63 (ecs-pattern) | **1.05 (xecs)** | **1.21 (xecs)** | **1.58 (xecs)** | **1.73 (xecs)** |
| w4 random | 0.82 (ecs-pattern) | 0.90 (ecs-pattern) | **2.91 (ecs-pattern)** | **5.95 (xecs)** | **9.90 (xecs)** |
| w5 churn | 0.17 (ecs-pattern) | 0.46 (ecs-pattern) | 0.77 (ecs-pattern) | 0.96 (ecs-pattern) | **1.65 (entt)** |
| w6 mixed | 0.34 (esper) | 0.40 (xecs) | 0.75 (xecs) | **1.50 (xecs)** | **1.85 (xecs)** |
| w7 migrate | 0.32 (snecs) | **1.23 (snecs)** | **1.83 (entt)** | **1.72 (entt)** | **1.01 (entt)** |

Note who the rival *is*: on every columnar/branchy/random workload it is another **vectorized** library
(xecs), never a native per-entity one. The only cell microecs still loses to a C/C++ engine is **one**
structural one (w7 migrate @100k, and by 1.04×) — which is exactly Experiment 3's point.

**What moved since the 2026-07-26 baseline: the two structural rows, and only those.** `#44` made
`World.add_entity` the sole validator of a spawn (the command buffer used to repeat the same pass); `#49`
made the pool's per-field dtype check cheap (`np.issubdtype` → `==`, 8× faster *and* a stricter check),
dropped a row copy the despawn path threw away, and stopped tearing down archetypes that empty and refill
within one tick. Normalized to the field: **w5 churn −50%**, **w7 migrate −30%**, everything else mixed-sign
single digits. A churn pair is **1.92× cheaper**. The two changes were measured separately first (−28% and
−29% on w5) and **compound**: 0.72 × 0.71 = 0.51 against the −50% here, because they cut different halves —
the buffered call and the commit. w7 was nobody's target; component migration just happens to run through
the same two pool functions churn does. **microecs now wins all seven workloads at N=100k**, structural ones
included, for the first time. The w1 @100k cell reads 3.10, in line with the 8-rep re-measure of **3.3× on
medians, 3.7× on mins** (w2: 2.5–2.7×). Nothing else moved from the previous run.

### Experiment 2 — columnar scaling to 1M (microecs vs xecs)

Past 100k only the two vectorized libraries stay in the race, so experiment 2 pits just those two on
the columnar workloads. xecs stays flat at its copy-bound rate; microecs holds near its in-place
floor. Columnar step, **ns/entity per frame** (lower is better):

| N | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|
| w1 physics — microecs | **1.59ns** | **1.66ns** | **1.82ns** | **2.81ns** |
| w1 physics — xecs | 4.94ns | 5.58ns | 5.72ns | 6.02ns |
| w2 bounce — microecs | **3.76ns** | **4.31ns** | **5.82ns** | **7.05ns** |
| w2 bounce — xecs | 9.91ns | 9.83ns | 11.36ns | 12.65ns |

At **1M entities** a physics frame is **2.8 ms (microecs) vs 6.0 ms (xecs)**, a bounce frame
**7.1 ms vs 12.7 ms** — a steady ~2× lead. It holds because microecs mutates the pool arrays in
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
| w1 physics | 2.72 | 3.69 | 2.72 | 2.68 | 2.55 |
| w2 bounce | 2.43 | 3.29 | 2.71 | 2.57 | 2.33 |
| w3 ai | 3.42 | 3.15 | 3.13 | 2.94 | 2.67 |
| w4 random | 2.44 | 2.24 | 2.82 | 2.38 | 2.30 |
| w5 churn | 1.82 | 2.26 | 2.15 | 1.65 | **0.94** |
| w6 mixed | 3.06 | 3.09 | 2.78 | 2.63 | 2.58 |
| w7 migrate | 1.05 | 1.05 | **0.94** | **0.95** | 1.00 |

**A native core costs ~2.3–3.4× on field arithmetic and only pays off on structural work.** Where the
work is per-entity `p.x += p.vx * dt`, you pay Python's loop *plus* a boundary crossing per component
access — so binding a world-class C++ engine lands you *behind* plain esper. Where the work **is** the
data-structure operation (spawn/despawn, add/remove component), the whole thing happens in C++ and EnTT
becomes the fastest library in the suite.

**w6 decides which of those two a real frame is: 2.42–3.11×, squarely in the arithmetic band.** The
mixed frame is the most representative workload in the suite (physics + ai + K targeted hits), and it
behaves like arithmetic, not like structure — so the structural win at 100k buys back a slice of a
frame, not the frame. That is the whole trade in one row.

Against microecs at N=100,000, ns/entity/frame:

| workload | microecs | entt (C++) | flecs (C) | esper (pure py) | microecs vs entt |
|---|--:|--:|--:|--:|--:|
| w1 physics | **1.6** | 291.0 | 864.5 | 113.9 | **183× faster** |
| w2 bounce | **3.8** | 450.5 | 1370.1 | 196.2 | **120× faster** |
| w3 ai | **4.2** | 162.6 | 491.9 | 73.3 | **39× faster** |
| w4 random | **0.1** | 10.3 | 29.8 | 15.6 | **103× faster** |
| w5 churn | **119.2** | 196.3 | 580.5 | 450.8 | **1.65× faster** |
| w6 mixed | **7.3** | 511.3 | 1626.1 | 279.7 | **70× faster** |
| w7 migrate | **192.3** | 193.6 | 583.8 | 499.1 | **1.01× faster** |

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
   ~1.1–1.4×. Above it microecs wins by ~1.5× at 5k, ~3× at 20k–500k, ~2.1× at 1M. The surprise — a
   pure-Python+numpy lib beating a Rust lib at scale — is real and mechanistic: xecs does its
   arithmetic *in numpy anyway* (`view.x * dt` returns an ndarray) and `.numpy()` is a **copy**, so
   every columnar op copies buffers out of Rust and back (~4–11× raw numpy). microecs mutates the pool
   arrays **in place, zero-copy** — it wins *because it has no FFI boundary*. (Confirmed in
   `examples/05-benchmark-workloads/probes/`.)
3. **The structural workloads were paying per-entity Python, not archetype layout.** w5 churn used to be
   microecs' weakest workload at every N (0.08× ecs-pattern at 200, 0.82× EnTT at 100k). Splitting one
   spawn+despawn pair showed why: the **storage** work — SoA insert plus the archetype pop-swap usually
   blamed for this — was 43%, while **validation was 32%, half of it duplicated**, and the rest was
   per-entity bookkeeping. Two changes cleared most of it: `#44` removed the duplicate validation pass,
   and `#49` made the pool's per-field dtype check cheap (`np.issubdtype` → `==`, 8× faster and a
   stricter check), dropped a row copy `update()` discarded, and stopped tearing down archetypes that
   empty and refill in one tick. Together: **w5 −50% and w7 migrate −30% against the field**, a churn pair
   **1.92× cheaper**, and microecs takes **every workload at 100k**. The w7 half was unplanned and is the
   more general lesson — component migration runs through the *same* `_pop_from_pool`/`_add_to_pool` pair
   as churn, so anything charged per entity there is charged twice per migration. It was bookkeeping, not
   architecture. Confirmed outside the benchmark: robosim's own physics tick is **~11% faster on average
   (up to 17% at 100 robots)**, while its render tick (draw-bound, no ECS mutation) stayed flat — the control.
   What is left on the spawn path is genuinely per-entity work that only a batch API could collapse; that is
   measured (a 92× ceiling) and deliberately deferred until a non-synthetic workload asks for it.
4. **Batch random access; capability gaps decide churn/migration.** `get_entity(id).f` in a hot loop
   is a ~3000 ns/hit trap — up to **503× slower** than a batched `col[rows] -= …` scatter (6 ns/hit),
   which is what the benchmark uses. Making the accessor itself ~1.8× cheaper did **not** move this:
   at scale the cost is cache-missing random reads into the column, not API overhead. It is the access
   *pattern* that is the trap. And xecs can't despawn *or* migrate; ecs-pattern can't migrate —
   for spawn/die churn and buff-on/off migration the qualifying set is microecs/entt/flecs/esper/snecs.
   If your update loop *isn't* vectorizable and stays small, a per-entity python ECS is simpler and
   faster — see the microbenchmark above.
5. **Binding a native ECS is not the shortcut it looks like.** Experiment 3: EnTT and flecs are
   ~2.3–3.4× *slower* than plain esper on every arithmetic workload — **including the realistic mixed
   frame (2.4–3.1×)** — and 39–490× slower than microecs. They win only churn and migration, where the
   work happens entirely inside the native engine, and those are a slice of a frame, not the frame. The
   axis that matters in Python is vectorized-vs-per-entity, **not** native-vs-interpreted.
