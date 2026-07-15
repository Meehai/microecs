# Benchmarks

Two benchmarks: microecs against plain OOP (why vectorize at all), and microecs against other Python ECS libraries (how it compares). Both verify every method against a numpy reference before timing it.

## Benchmark 1: microecs vs OOP on a simple physics step

We run the same physics step `pos += vel*dt` over N=100k entities split across 2 pools in various ways (ECS or OOP). All methods are verified to produce the identical result. Reproduce with `python examples/04-benchmark-ecs-vs-oop.py` (it prints `{mode: avg_seconds_per_step}`).

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

## Benchmark 2: microecs vs other Python ECS libraries

Not one workload — **seven** × **five** libraries × an N-sweep (200 → 1,000,000), every result
verified against a float64 reference. Full setup, fairness notes, and raw data in
`examples/05-benchmark-workloads/` (one folder per workload; `FINDINGS.md` has every number; run
`./run_benchmark.sh` to regenerate). Environment: numpy 2.5.1, Python 3.12; times are min-over-reps
of the mean over 30 frames, GC off.

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

### The five libraries

| library | model | how each workload is driven (idiomatic, best-case) |
|---|---|---|
| **microecs** | numpy SoA by archetype | columnar via `QueryResult` write-through; branch via `np.where`; random via batched column scatter; churn/migrate via `add/remove_entity` + `update()` |
| **xecs** | Rust SoA, per-component columns, no archetypes | columnar via in-place `view.x += …`; random scatters a column; **no despawn/migration → w5/w7 N/A** |
| **esper** | pure-python, per-entity objects | `get_components` loop; `if` per entity; O(1) `component_for_entity(id)`; `create/delete_entity` |
| **snecs** | pure-python sparse-set | compiled `Query` loop; per-entity `if`; `entity_component(id)`; sparse-set migration (the migration champ) |
| **ecs-pattern** | pure-python, dataclass AoS | `get_with_component` loop; direct object ref (fastest random access); **fixed inheritance classes → w7 N/A** |

### Experiment 1 — the full field: N=200 → 100k, all five libraries

There is no single winner — the fastest library flips by workload *and* by N:

| fastest library | N=200 | 1k | 5k | 20k | 100k |
|---|---|---|---|---|---|
| w1 physics | xecs | xecs | **microecs** | **microecs** | **microecs** |
| w2 bounce | xecs | xecs | **microecs** | **microecs** | **microecs** |
| w3 ai | esper | **microecs** | **microecs** | **microecs** | **microecs** |
| w4 random | ecs-pattern | ecs-pattern | **microecs** | **microecs** | **microecs** |
| w5 churn | ecs-pattern | ecs-pattern | ecs-pattern | ecs-pattern | snecs |
| w6 mixed | esper | xecs | xecs | **microecs** | **microecs** |
| w7 migrate | snecs | snecs | **microecs** | **microecs** | snecs |

*w7 migrate: xecs and ecs-pattern can't migrate (N/A).*

**How close is the race, and against whom?** Each cell below is **how many times faster microecs is**
than the fastest other library (that library's time ÷ microecs's), with that rival named. `>1` →
microecs is faster; `<1` → slower (by `1/x`). So `3.31 (xecs)` = microecs **3.3× faster** than xecs;
`0.79 (xecs)` = microecs **1.27× slower** than xecs; `0.94 (snecs)` = a near-tie, snecs just ahead.
The named lib is microecs's nearest rival — the one it beats, or the one beating it. **Bold** =
microecs wins that cell (ratio `>1`).

| workload | N=200 | 1k | 5k | 20k | 100k |
|---|---|---|---|---|---|
| w1 physics | 0.79 (xecs) | 0.90 (xecs) | **1.78 (xecs)** | **3.41 (xecs)** | **3.31 (xecs)** |
| w2 bounce | 0.69 (xecs) | 0.73 (xecs) | **1.38 (xecs)** | **2.45 (xecs)** | **2.51 (xecs)** |
| w3 ai | 0.64 (esper) | **1.12 (xecs)** | **1.43 (xecs)** | **1.64 (xecs)** | **1.67 (xecs)** |
| w4 random | 0.74 (ecs-pattern) | 0.85 (ecs-pattern) | **3.18 (ecs-pattern)** | **6.55 (xecs)** | **9.91 (xecs)** |
| w5 churn | 0.08 (ecs-pattern) | 0.23 (ecs-pattern) | 0.40 (ecs-pattern) | 0.50 (ecs-pattern) | 0.94 (snecs) |
| w6 mixed | 0.35 (esper) | 0.40 (xecs) | 0.85 (xecs) | **1.52 (xecs)** | **2.04 (xecs)** |
| w7 migrate | 0.30 (snecs) | 0.93 (snecs) | **1.19 (snecs)** | **1.06 (snecs)** | 0.79 (snecs) |

### Experiment 2 — columnar scaling to 1M (microecs vs xecs)

Past 100k only the two vectorized libraries stay in the race, so experiment 2 pits just those two on
the columnar workloads. xecs stays flat at its copy-bound rate; microecs holds near its in-place
floor. Columnar step, **ns/entity per frame** (lower is better):

| N | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|
| w1 physics — microecs | **1.51ns** | **1.44ns** | **1.65ns** | **2.76ns** |
| w1 physics — xecs | 5.02ns | 4.54ns | 5.33ns | 5.55ns |
| w2 bounce — microecs | **3.51ns** | **3.91ns** | **5.65ns** | **6.93ns** |
| w2 bounce — xecs | 8.82ns | 9.08ns | 10.77ns | 11.79ns |

At **1M entities** a physics frame is **2.8 ms (microecs) vs 5.5 ms (xecs)**, a bounce frame
**6.9 ms vs 11.8 ms** — a steady ~2× lead. It holds because microecs mutates the pool arrays in
place while xecs copies ~6 buffers across the Rust↔numpy boundary every step (the copy-boundary
mechanism explained in the takeaways below). microecs is the only library in the suite that steps 1M
entities per frame in low-single-digit milliseconds with no GPU and no compile step.

Four things to take from the two experiments:

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
   for spawn/die churn and buff-on/off migration only microecs/esper/snecs qualify (snecs, sparse-set,
   is the migration champ; microecs is mid — the archetype whole-entity copy). Structural **churn**
   (w5) is microecs' one genuine loss at every N but 100k. If your update loop *isn't* vectorizable
   and stays small, a per-entity python ECS is simpler and faster — see the microbenchmark above.
