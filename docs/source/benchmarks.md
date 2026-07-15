# Benchmarks

Two benchmarks: microecs against plain OOP (why vectorize at all), and microecs against other Python ECS libraries (how it compares). Both verify every method against a numpy reference before timing it.

## microecs vs OOP on a simple physics step

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

## microecs vs other Python ECS libraries

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

### Headline: there is no single winner — the fastest library flips by workload *and* by N

```
fastest library   N=200        1k          5k          20k        100k
w1 physics        xecs         xecs        microecs    microecs   microecs
w2 bounce         xecs         xecs        microecs    microecs   microecs
w3 ai             esper        microecs    microecs    microecs   microecs
w4 random         ecs-pattern  ecs-pattern microecs    microecs   microecs
w5 churn          ecs-pattern  ecs-pattern ecs-pattern ecs-pattern snecs
w6 mixed          esper        xecs        xecs        microecs   microecs
w7 migrate        snecs        snecs       microecs    microecs   snecs      (xecs/ecs-pattern N/A)
```

### The large-N tail — columnar throughput to 1M entities (microecs vs xecs)

Past 100k the two vectorized libraries separate cleanly: xecs stays flat at its copy-bound rate,
microecs holds near its in-place floor. Columnar step, **ns/entity per frame** (lower is better):

| N | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|
| w1 physics — microecs | 1.51 | 1.44 | 1.65 | 2.76 |
| w1 physics — xecs | 5.02 | 4.54 | 5.33 | 5.55 |
| w2 bounce — microecs | 3.51 | 3.91 | 5.65 | 6.93 |
| w2 bounce — xecs | 8.82 | 9.08 | 10.77 | 11.79 |

At **1M entities** a physics frame is **2.8 ms (microecs) vs 5.5 ms (xecs)**, a bounce frame
**6.9 ms vs 11.8 ms** — a steady ~2× lead. It holds because microecs mutates the pool arrays in
place while xecs copies ~6 buffers across the Rust↔numpy boundary every step (the copy-boundary
mechanism the ratio notes below explain). microecs is the only library in the suite that steps 1M
entities per frame in low-single-digit milliseconds with no GPU and no compile step.

### Who's 2nd, and by how much — winner ▸ runner-up (margin)

The winner alone hides how *close* the race is. `A▸B ×m` reads "A won, B was runner-up, A is m×
faster than B." Where microecs isn't first it's usually the runner-up (xecs on columnar/composite,
snecs on migration); the margin is how much the winner beats that runner-up.

```
workload    N=200                  1k                    5k                    20k                  100k
w1 physics  xecs▸microecs 1.27     xecs▸microecs 1.11    microecs▸xecs 1.78    microecs▸xecs 3.41   microecs▸xecs 3.31
w2 bounce   xecs▸microecs 1.44     xecs▸microecs 1.37    microecs▸xecs 1.38    microecs▸xecs 2.45   microecs▸xecs 2.51
w3 ai       esper▸ecs-pat 1.22     microecs▸xecs 1.12    microecs▸xecs 1.43    microecs▸xecs 1.64   microecs▸xecs 1.67
w4 random   ecs-pat▸microecs 1.36  ecs-pat▸microecs 1.17 microecs▸ecs-pat 3.18 microecs▸xecs 6.55   microecs▸xecs 9.91
w5 churn    ecs-pat▸snecs 2.03     ecs-pat▸snecs 2.36    ecs-pat▸snecs 2.35    ecs-pat▸snecs 1.89   snecs▸ecs-pat 1.06
w6 mixed    esper▸xecs 1.05        xecs▸microecs 2.48    xecs▸microecs 1.18    microecs▸xecs 1.52   microecs▸xecs 2.04
w7 migrate  snecs▸esper 1.76       snecs▸microecs 1.08   microecs▸snecs 1.19   microecs▸snecs 1.06  snecs▸microecs 1.26
```

### microecs vs its nearest rival — the ratio (microecs / fastest competitor)

`>1` = microecs slower by that factor; `<1` = microecs faster by `1/x`. E.g. w1 physics @100k = 0.30
means microecs is **3.3× faster** than the next-best lib (xecs).

```
workload    N=200   1k     5k     20k    100k
w1 physics  1.27   1.11   0.56   0.29   0.30
w2 bounce   1.44   1.37   0.72   0.41   0.40
w3 ai       1.57   0.89   0.70   0.61   0.60
w4 random   1.36   1.17   0.31   0.15   0.10
w5 churn   12.32   4.42   2.52   2.00   1.06
w6 mixed    2.89   2.48   1.18   0.66   0.49
w7 migrate  3.38   1.08   0.84   0.95   1.26
```

Four things to take from it:

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
