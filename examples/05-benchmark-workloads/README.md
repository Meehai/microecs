# microecs vs the field — multi-workload ECS benchmark

The older single-workload benchmark measured **one** thing: a pure columnar physics step at N=100k.
That is microecs' best case, and it wins there. This suite asks the harder question: **how does
microecs do across the workloads a real game actually runs, and at the entity counts games actually
use?**

**Seven workloads × five libraries × an N-sweep (200 → 1,000,000)**, every result verified against a
float64 reference. The headline finding: *there is no single winner* — the fastest library flips
depending on **both** the workload **and** N.

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

## The five libraries

| library | model | how each workload is driven (idiomatic, best-case) |
|---|---|---|
| **microecs** | numpy SoA by archetype | columnar via `QueryResult` write-through; branches via `np.where`; random via batched column scatter; churn/migrate via `add/remove_entity`+`update()` |
| **xecs** | Rust SoA, per-component columns, no archetypes | columnar via in-place `view.x += …`; branches via `.numpy()`+`.fill()`; random scatters a column; **no despawn/migration → w5/w7 N/A** |
| **esper** | pure-python, per-entity objects | `get_components` loop; `if` per entity; `component_for_entity(id)` (O(1) dict); `create/delete_entity`; `add/remove_component` |
| **snecs** | pure-python sparse-set | compiled `Query` loop; per-entity `if`; `entity_component(id)`; `delete_entity_immediately`; sparse-set migration |
| **ecs-pattern** | pure-python, dataclass AoS | `get_with_component` loop; per-entity `if`; direct object reference (fastest random access); `em.add/delete`; **fixed inheritance classes → w7 N/A** |

## Fairness & verification

- Every library gets its **own best-case layout**: pure-python libs use `__slots__`/dataclass +
  python floats (their fast path); the two vectorized libs (microecs, xecs) use float32 columns.
- **SoA vs SoA is symmetric.** w4/w6 give microecs the **same** columnar scatter idiom xecs gets
  (`col[rows] -= DMG`) — both are columnar libs, so both batch. The naive `get_entity(id)` loop is
  *not* used in the hot path; it is quantified separately as "the trap" in `probes/microecs_random.py`.
  The AoS libs (esper/snecs/ecs-pattern) legitimately loop with O(1) id lookups / direct object refs.
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
pip install -r requirements.txt    # esper, snecs, ecs-pattern, xecs, numpy
pip install -e ../../              # microecs, from the repo root
python run_benchmark.py            # full matrix + columnar tail -> results.json + tables
```

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

See **FINDINGS.md** for the numbers and the analysis (and the microecs plan
`.tracker/plans/1-comparison-with-other-projects.md`, Part 8, for the full write-up).
