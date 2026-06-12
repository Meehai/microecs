# microecs vs popular Python ECS libraries — batch-update benchmark

Same physics, same scene, same verification — run across the most popular Python ECS
libraries plus microecs. One script per library, one shared workload.

## The workload

A classic ECS physics step. The scene has **N entities**; half also carry an
`Acceleration`. One frame runs two batched systems (semi-implicit / symplectic Euler):

```
1. integrate velocity:  vel += acc * dt     # only the half with Acceleration
2. integrate position:  pos += vel * dt     # every entity
```

This is the "velocity / acceleration / position" batch update that an ECS is supposed to
make fast. Every library simulates the **same** entities from the **same** seed for the
**same** number of frames, and each result is checked against a float64 numpy ground
truth via an order-independent fingerprint — so a library cannot look fast by skipping work.

## The contestants

| library | model | why it's here |
|---|---|---|
| **microecs** | numpy struct-of-arrays, vectorized | the project under test |
| **xecs** | Rust struct-of-arrays, vectorized | the one *other* vectorized ECS — the real rival |
| **esper** | pure-python, per-entity objects | the most popular Python ECS |
| **snecs** | pure-python, sparse-set, typed | popular, type-hinted |
| **ecs-pattern** | pure-python, dataclass + entity-inheritance | popular, dataclass-style |

## Fairness (read this before quoting numbers)

Each library uses its **own idiomatic, best-case** layout and hot path:

- **Pure-python libs (esper/snecs/ecs-pattern)** store position/velocity as plain
  python-`float` scalars — their *fastest* representation. (Per-entity numpy `(2,)` arrays
  are ~13× slower for them; see the microecs README microbenchmark.) snecs queries are
  `.compile()`d outside the hot loop, its documented fast path.
- **Vectorized libs (microecs/xecs)** store `(2,)` / columnar `float32` arrays — their
  natural layout — and update with batched ops, no per-entity python.
- **Every library is driven at the data level** (direct query / component views), with the
  per-library *scheduler* bypassed (no `esper.process`, no xecs app loop, etc.). That
  isolates the batch-update cost; adding any scheduler back adds fixed per-frame overhead to
  *all* of them.

We compare the **computation**, not the storage. float32 vs float64 is reconciled by a
tolerance loose enough for float32 accumulation; the fingerprints all match.

## Results

`python run_all.py 100000 30` — N=100,000 entities, 30 timed frames, on this machine
(numpy 2.3, Python 3.12). Your numbers will differ; the *ratios* are the point.

| rank | library | build | step / frame | ns / entity | M updates/s | step speedup |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **microecs** | 848 ms | **0.160 ms** | **1.6** | 624 | **189×** |
| 2 | **xecs** | 14 ms | 0.547 ms | 5.5 | 183 | 56× |
| 3 | esper | 186 ms | 9.33 ms | 93.3 | 10.7 | 3.3× |
| 4 | ecs-pattern | 112 ms | 10.74 ms | 107.4 | 9.3 | 2.8× |
| 5 | snecs | 262 ms | 30.38 ms | 303.8 | 3.3 | 1.0× |

("step speedup" is versus the slowest, snecs. All five verified against the numpy reference.)

## Takeaways

1. **Vectorized SoA wins batch updates, hard.** microecs and xecs run the step at 2–6
   ns/entity; the per-entity python ECSs run at 93–304 ns/entity — **19–189× slower**.
   For vectorizable simulation this is the whole game, and it's exactly microecs' thesis.
2. **microecs has the fastest step; build is now its heaviest phase, not a cliff.** Building
   100k entities takes **~0.85 s** — ~4× the pure-python libs and ~60× xecs' **14 ms** bulk
   Rust spawn, because microecs creates entities one at a time in python (the `add_entity`
   loop + the `update()` migration) instead of bulk-spawning. (It used to be ~16 s — that was
   a stray per-entity debug-log call, since removed; the fix made `add_entity` ~45× faster.)
3. **Per-step champion catches up fast.** microecs' faster step repays its bigger build versus
   xecs after only **~2 200 frames** (~36 s at 60 fps), and versus esper after **~70 frames**.
   For any long-lived sim microecs wins total time; only spawn-heavy churn favors xecs' near-free build.
4. **If your update loop isn't vectorizable, none of this applies** — a per-entity python
   ECS like esper is simpler and, for data-dependent per-entity logic, faster than a
   numpy-backed one (again, see the microecs README).

## Running it

```bash
pip install -r requirements.txt          # esper, snecs, ecs-pattern, xecs, numpy
pip install -e ../../                     # microecs itself, from the repo root

python run_all.py                         # all libs, N=100000, 30 frames (default)
python run_all.py 20000 15                # smaller/faster: N, frames as args
python bench_microecs.py                  # any single library standalone
```

## Files

- `_common.py` — shared workload: data generation, the float64 reference, fingerprint
  check, timing harness, and the `run_bench(...)` driver every script funnels through.
- `bench_microecs.py`, `bench_xecs.py`, `bench_esper.py`, `bench_snecs.py`,
  `bench_ecs_pattern.py` — one library each; each exposes `run(n, warmup, measure)` and is
  runnable standalone.
- `run_all.py` — runs them all and prints the comparison table.
- `requirements.txt` — the third-party libraries (microecs is installed separately).
