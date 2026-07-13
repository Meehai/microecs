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

The same batched physics step (`vel += acc*dt` then `pos += vel*dt`) over N=100k entities, run
across the most popular Python ECS libraries. One script per library, every result verified
against a float64 numpy reference. Full setup, fairness notes, and analysis in
`examples/05-benchmark-vs-similar/`.

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
