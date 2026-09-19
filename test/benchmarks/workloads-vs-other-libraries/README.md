# microecs vs the field — multi-workload ECS benchmark

The older single-workload benchmark measured **one** thing: a pure columnar physics step at N=100k.
That is microecs' best case, and it wins there. This suite asks the harder question: **how does
microecs do across the workloads a real game actually runs, and at the entity counts games actually
use?**

**Seven workloads × seven libraries × an N-sweep (200 → 1,000,000)**, every result verified against a
float64 reference. The field covers all three ways to build an ECS you can drive from Python:
numpy-vectorized (microecs), a native core with a native columnar store (xecs), a native core handing
back Python component objects (**EnTT** C++, **flecs** C), and pure Python (esper, snecs, ecs-pattern).
The headline finding: *there is no single winner* — the fastest library flips depending on **both** the
workload **and** N.

## The seven workloads (one subfolder each — map to real game systems)

| id | folder | what it models | shape |
|---|---|---|---|
| **w1** | `w1_physics/` | integrate `vel+=acc·dt; pos+=vel·dt` over all | particles, bullets, boids — **columnar** |
| **w2** | `w2_bounce/` | w1 + wall reflection (`np.where` branch) | anything with a data-parallel branch |
| **w3** | `w3_ai/` | per-entity health state machine (branch per entity) | NPC/AI ticks — **branchy row logic** |
| **w4** | `w4_random/` | read-modify-write K entities picked by id each frame | targeted damage, hit resolution — **random access** |
| **w5** | `w5_churn/` | spawn B + despawn B (FIFO) per frame + integrate | bullet-hell spawn/die — **structural churn** |
| **w6** | `w6_mixed/` | w1 physics + w3 ai + K targeted hits, fixed set | a realistic steady-state frame — **composite** |
| **w7** | `w7_migrate/` | integrate all + rolling component add/remove | buff/debuff on/off — **archetype migration** |

`k = max(16, n//50)` entities touched/frame (w4/w6); `b = max(16, n//100)` churned/frame (w5);
`2·max(4, n//200)` migrations/frame (w7).

## The seven libraries

| library | core | model | how each workload is driven (idiomatic, best-case) |
|---|---|---|---|
| **microecs** | pure python | numpy SoA by archetype | columnar via `QueryResult` write-through; branches via `np.where`; random via batched column scatter; churn/migrate via `add/remove_entity`+`update()` |
| **xecs** | **Rust** (pyo3) | SoA, per-component columns, no archetypes | columnar via in-place `view.x += …`; branches via `.numpy()`+`.fill()`; random scatters a column; **no despawn/migration → w5/w7 N/A** |
| **entt** | **C++** (nanobind) | [EnTT](https://github.com/skypjack/entt) sparse-set; components are python objects | `registry.view(A, B)` yields `(entity, *comps)`; `registry.get(e, T)`; `create`/`destroy`/`emplace`/`remove` |
| **flecs** | **C** (pybind11) | [flecs](https://www.flecs.dev) archetype tables; components are python objects | `world.query(A, B)`; `entity.get/set/remove`; `entity.destroy()` |
| **esper** | pure python | per-entity objects | `get_components` loop; `if` per entity; `component_for_entity(id)` (O(1) dict); `create/delete_entity`; `add/remove_component` |
| **snecs** | pure python | sparse-set | compiled `Query` loop; per-entity `if`; `entity_component(id)`; `delete_entity_immediately`; sparse-set migration |
| **ecs-pattern** | pure python | dataclass AoS | `get_with_component` loop; per-entity `if`; direct object reference (fastest random access); `em.add/delete`; **fixed inheritance classes → w7 N/A** |

Getting the two native engines to run at all is part of the finding — neither is a `pip install`:
**PyEnTT** publishes no Linux wheel *and* no sdist (build from the repo with its `entt` submodule); the
PyPI **flecs** Linux wheel is cp39-only (build from the sdist), and that binding **segfaults** when a
`Query` is iterated twice without `.reset()` in between, and again at interpreter shutdown (harmless —
`results.json` is written before exit). The adapters call `.reset()` before every iteration. Install
notes are in `requirements.txt`.

## Fairness & verification

- Every library gets its **own best-case layout**: the per-entity libs (esper/snecs/ecs-pattern and the
  native-core entt/flecs, whose components are python objects) use `__slots__`/dataclass + python floats,
  their fast path; the two columnar libs (microecs, xecs) use float32 columns.
- **SoA vs SoA is symmetric.** w4/w6 give microecs the **same** columnar scatter idiom xecs gets
  (`col[rows] -= DMG`) — both are columnar libs, so both batch. The naive `get_entity(id)` loop is
  *not* used in the hot path; it is quantified separately as "the trap" in `probes/microecs_random.py`.
  The AoS libs (esper/snecs/ecs-pattern/entt/flecs) legitimately loop with O(1) id lookups / direct
  object refs.
- Every library runs at the **data level** (no per-library scheduler), isolating the actual work.
- Every run is verified against a **float64 numpy/python reference** (`common.py`) via an
  order-independent fingerprint (pool all values, sort) — a library cannot look fast by skipping work.
- **Capability gaps are reported N/A, never faked** (per "no stub-that-lies"): xecs has no despawn
  (w5) and no component migration (w7); ecs-pattern has fixed inheritance-class entities (w7).
- Times are the **min over reps** (6 for N≤5k, 4 for N≤30k, 2 above) of the mean over 30 timed
  frames after 3 warmup frames, GC disabled during timing.

## Running it

```bash
./run_benchmark.sh                 # sets up .venv, installs deps, runs full matrix + tail
./run_benchmark.sh 200 1000 5000   # custom N list (main matrix only)
```

or manually:

```bash
pip install -r requirements.txt    # esper, snecs, ecs-pattern, xecs, numpy (+ entt/flecs: see the file)
pip install -e ../../              # microecs, from the repo root
python run_benchmark.py            # full matrix + columnar tail -> results.json + tables
```

A library that is not installed is reported `N/A no adapter`-style per cell, so a partial field still
runs — but the winner map is only meaningful with all seven present.

## Layout

```
common.py            deterministic scene, event streams, float64 references, verify, timing harness
run_benchmark.py     the driver: sweeps <workload>/<lib> x N, verifies, prints tables + winner map
run_benchmark.sh     venv setup + deps + run
wN_*/                one folder per workload; one file per library (each: build/step/collect)
                     a missing <lib>.py = that library can't express this workload (reported N/A)
probes/              mechanism probes behind the analysis (the crossover, the copy boundary, the trap)
FINDINGS.md          full result matrix + winner map + the "why" + honest limitations
```

## Reading the numbers honestly

Absolute ms drift **±10–25% between runs** on the same machine (thermals/governor): re-running this
suite 11 days apart moved *every* library — including the six microecs cannot affect — by up to +12%
median. So the portable results are the **ratios inside one run**, the winner map, and the crossover;
a single cell is ±20%. To compare two runs of a library change, normalize against the field
(microecs ÷ median of the other six) rather than comparing ms to ms, and keep the machine idle while
the suite runs.

See **FINDINGS.md** for the numbers and the analysis (and the microecs plan
`.tracker/plans/1-comparison-with-other-projects.md`, Part 8, for the full write-up).
