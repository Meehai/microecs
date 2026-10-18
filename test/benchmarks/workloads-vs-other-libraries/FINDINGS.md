# Findings — multi-workload ECS benchmark (raw data + the "why")

Environment: numpy 2.5.1, Python 3.12.12, Linux x86_64. Times = min-over-reps of mean-over-30-frames,
GC off. Every cell verified against a float64 reference. Full data in `results.json` (regenerate with
`./run_benchmark.sh`).

**Run: 2026-10-18**, microecs `b49b28d` (0.8.6). **Prev run:** 2026-07-27c, `6fc0d21` (merge of #44 + #49).
No adapter needed changes: every microecs change since was additive or internal.

What is in it, perf-relevant: **#50** (`_convert_components_data`: array-like field data is converted at the
call, a per-field Python pass on every `add_entity`/`add_component`), **#39** (staged writes snapshot the
caller's arrays: `CommandBuffer.append` now `.copy()`s every ndarray arg, so the value at the call is what
lands), **#53** (repeat `remove_component` in one tick is a no-op). Not perf-relevant: #51 `qr.f[mask]`,
world reserved names, assert→raise, #55 test formalism.

**Field: seven libraries**, i.e. all three ways to build an ECS you can drive from Python — numpy-vectorized
(microecs), native core + native columnar store (xecs), native core handing back Python component
objects (entt = C++ EnTT, flecs = C flecs), and pure Python (esper, snecs, ecs-pattern).

**Headline: spawning is ~25% dearer, everything else is flat, and the winner map did not move.** In-process
A/B old tree vs new tree (`test/manual/bench-compare/ab_versions.py`, ABBA, median of 4): **w5 churn step
+24–31%**, **spawn (`build`) +17–32% on every workload**, w7 migrate step +5–16%, columnar steps (w1/w4/w6)
0.92–1.14 (noise). The cost is the two new per-spawn passes: of a ~6.3 µs `add_entity`, **#50's conversion
is ~860 ns (14%) and #39's snapshot copy ~680 ns (11%)** (`spawn_split.py`). microecs still wins all seven
workloads at 100k; the churn lead over EnTT there shrank **1.65× → 1.33×**.

**Update — #56 landed (`befc0f5`), w5/w7 extrapolated, not re-swept.** #56 folds conversion + the one snapshot
copy into `World._validate_component`'s per-field loop (`np.array` for array-likes, `.copy()` for ndarrays) and
deletes `_convert_components_data` and `append`'s copy loop. An in-process A/B of microecs alone, published
`b49b28d` vs `befc0f5` (`ab_versions.py`, ABBA, median of 4, all five N), gives step ratios **w5 0.80–0.93
(median 0.91)** and **w7 0.92–1.02 (median 0.93)**; `build` (= N × `add_entity`) 0.78–0.95 on every workload.
The w1 control step (never calls `add_entity`) wandered 0.92–1.18, so per-cell ratios are ±15% noise: the
microecs w5/w7 cells below are the measured cells × the **per-workload median**. Columnar workloads don't spawn
in `step`, so they are unchanged. Result: churn lead at 100k **1.47× EnTT**, winner map unchanged.

**Fairness note (SoA vs SoA):** w4/w6 give microecs the SAME columnar scatter idiom xecs gets
(`col[rows] -= DMG`) — both are columnar SoA libs, so both batch. The naive `get_entity(id)` loop is
quantified separately as "the trap" (probe P3), never used in the hot path. The AoS libs
(esper/snecs/ecs-pattern/entt/flecs) legitimately loop with O(1) id lookups / direct object refs.

## Matrix — step ms/frame (lower is better)

### w1 physics (columnar integrate)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0116 | 0.0128 | **0.0174** | **0.0260** | **0.1347** |
| xecs | **0.0092** | **0.0122** | 0.0292 | 0.0960 | 0.4408 |
| entt | 0.0451 | 0.2187 | 1.0755 | 4.2729 | 22.1092 |
| flecs | 0.1112 | 0.6112 | 2.9617 | 12.3264 | 73.6852 |
| esper | 0.0142 | 0.0731 | 0.3664 | 1.6098 | 9.9049 |
| snecs | 0.0440 | 0.2153 | 1.0798 | 4.4921 | 25.7404 |
| ecs-pattern | 0.0232 | 0.0916 | 0.4354 | 1.7544 | 9.7056 |

### w1 physics — columnar TAIL, ns/entity (microecs vs xecs)
| N | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|
| microecs | **1.35** | **1.42** | **1.57** | **2.67** |
| xecs | 4.41 | 4.34 | 4.76 | 5.07 |

xecs is flat at ~4.3–5.1 ns/e everywhere; microecs falls to ~1.4 ns/e as fixed overhead amortizes.
**Crossover ≈ N=1.5–3k.** (w2 bounce tail, same shape: microecs 3.6→6.7 ns/e vs xecs 8.3→11.2 over
200k→1M.)

### w2 bounce (integrate + np.where wall reflection)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0258 | 0.0307 | **0.0401** | **0.0727** | **0.3339** |
| xecs | **0.0166** | **0.0229** | 0.0537 | 0.1748 | 0.8586 |
| entt | 0.0759 | 0.3589 | 1.7408 | 6.9935 | 37.1397 |
| flecs | 0.1885 | 0.9898 | 4.7055 | 20.3654 | 113.4302 |
| esper | 0.0242 | 0.1239 | 0.6168 | 2.5101 | 16.1353 |
| snecs | 0.0682 | 0.3597 | 1.7941 | 7.5458 | 40.2704 |
| ecs-pattern | 0.0380 | 0.1547 | 0.7054 | 2.8856 | 15.2448 |

### w3 ai (per-entity health state machine)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0127 | **0.0176** | **0.0298** | **0.0775** | **0.3636** |
| xecs | 0.0130 | 0.0191 | 0.0411 | 0.1287 | 0.5831 |
| entt | 0.0274 | 0.1308 | 0.6428 | 2.6294 | 12.9486 |
| flecs | 0.0664 | 0.3663 | 1.7105 | 7.2539 | 39.1100 |
| esper | **0.0085** | 0.0416 | 0.2144 | 0.8678 | 6.2902 |
| snecs | 0.0242 | 0.1266 | 0.6541 | 2.6641 | 14.6887 |
| ecs-pattern | 0.0096 | 0.0488 | 0.2453 | 0.9752 | 5.1687 |

### w4 random (K=max(16,n//50) distinct hits/frame; SoA libs scatter, AoS libs loop)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0015 | 0.0017 | **0.0018** | **0.0028** | **0.0083** |
| xecs | 0.0028 | 0.0036 | 0.0066 | 0.0196 | 0.0941 |
| entt | 0.0027 | 0.0035 | 0.0168 | 0.0641 | 0.5792 |
| flecs | 0.0071 | 0.0099 | 0.0498 | 0.2475 | 2.3856 |
| esper | 0.0018 | 0.0022 | 0.0117 | 0.0684 | 1.0731 |
| snecs | 0.0020 | 0.0025 | 0.0133 | 0.0962 | 1.1357 |
| ecs-pattern | **0.0012** | **0.0014** | 0.0060 | 0.0249 | 0.2605 |

### w5 churn (spawn B + FIFO-despawn B/frame + integrate; B=max(16,n//100))
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.1749 | 0.1687 | 0.4963 | 2.0205 | **10.8858** |
| xecs | N/A | N/A | N/A | N/A | N/A  (no despawn) |
| entt | 0.0437 | 0.1510 | 0.7219 | 2.9081 | 15.9585 |
| flecs | 0.1080 | 0.3912 | 1.9498 | 8.1680 | 49.0827 |
| esper | 0.0718 | 0.2892 | 1.5613 | 6.2943 | 37.9337 |
| snecs | 0.0462 | 0.1580 | 0.7850 | 3.0418 | 17.2808 |
| ecs-pattern | **0.0229** | **0.0667** | **0.3514** | **1.7014** | 18.5872 |

### w6 mixed (physics + ai + K targeted damage, fixed set — a realistic frame)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0774 | 0.0774 | 0.1022 | **0.1734** | **0.6299** |
| xecs | 0.0272 | **0.0342** | **0.0796** | 0.2450 | 1.1573 |
| entt | 0.0774 | 0.3689 | 1.8007 | 7.1567 | 40.9269 |
| flecs | 0.2039 | 0.9345 | 5.3102 | 21.6903 | 131.3288 |
| esper | **0.0258** | 0.1203 | 0.6704 | 2.8867 | 23.7649 |
| snecs | 0.0704 | 0.3438 | 1.8850 | 7.8938 | 48.3183 |
| ecs-pattern | 0.0391 | 0.1431 | 0.7005 | 2.8673 | 16.7114 |

### w7 migrate (component add/remove → archetype migration; 2·max(4,n//200) migrations/frame)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0940 | **0.1069** | **0.4108** | **1.7757** | **13.6982** |
| xecs | N/A | N/A | N/A | N/A | N/A  (no component add/remove) |
| entt | 0.0320 | 0.1425 | 0.7112 | 2.8342 | 15.1191 |
| flecs | 0.0759 | 0.3640 | 1.8256 | 8.3166 | 50.1574 |
| esper | 0.0592 | 0.2802 | 1.4785 | 6.2121 | 42.8056 |
| snecs | **0.0311** | 0.1410 | 0.7245 | 2.8824 | 16.6167 |
| ecs-pattern | N/A | N/A | N/A | N/A | N/A  (fixed inheritance-class entities) |

## Winner map (fastest library)
```
workload      N=200        1k          5k          20k        100k
w1 physics    xecs         xecs        microecs    microecs   microecs
w2 bounce     xecs         xecs        microecs    microecs   microecs
w3 ai         esper        microecs    microecs    microecs   microecs
w4 random     ecs-pattern  ecs-pattern microecs    microecs   microecs
w5 churn      ecs-pattern  ecs-pattern ecs-pattern ecs-pattern microecs
w6 mixed      esper        xecs        xecs        microecs   microecs
w7 migrate    snecs        microecs    microecs    microecs   microecs
```
**Unchanged from 07-27c:** microecs takes the entire 100k column, all seven workloads. The only flip is w3 ai
@200 (ecs-pattern → esper), a rival-vs-rival tie inside drift that says nothing about microecs.


## microecs / fastest-competitor, and who that competitor is
`>1` = microecs faster by that factor; `<1` = slower by `1/x`.
```
workload     N=200               1k                  5k                  20k            100k
w1 physics   0.80 xecs           0.95 xecs           1.68 xecs           3.70 xecs      3.27 xecs
w2 bounce    0.64 xecs           0.75 xecs           1.34 xecs           2.40 xecs      2.57 xecs
w3 ai        0.67 esper          1.08 xecs           1.38 xecs           1.66 xecs      1.60 xecs
w4 random    0.78 ecs-pattern    0.84 ecs-pattern    3.32 ecs-pattern    7.10 xecs      11.31 xecs
w5 churn     0.13 ecs-pattern    0.40 ecs-pattern    0.71 ecs-pattern    0.84 ecs-pat   1.47 entt
w6 mixed     0.33 esper          0.44 xecs           0.78 xecs           1.41 xecs      1.84 xecs
w7 migrate   0.33 snecs          1.32 snecs          1.73 entt           1.60 entt      1.10 entt
```
(w5/w7 rows: extrapolated after #56, see the update at the top; measured 10-18 values were w5 0.12 / 0.36 /
0.64 / 0.77 / 1.33 and w7 0.31 / 1.23 / 1.61 / 1.48 / 1.03.)

**w5 churn is the one row that moved, and it moved back.** 07-27c → 10-18: 0.17→0.12, 0.46→0.36, 0.77→0.64,
0.96→0.77, **1.65→1.33** — the #50 + #39 spawn tax (see the headline). w7 dips at 5k/20k (1.83→1.61,
1.72→1.48), consistent with the A/B's +5–16% on migration. Every other row is flat within drift.
The rival on every columnar/branchy/random workload is another **vectorized** library (xecs), never a
native per-entity one. microecs only loses to a C/C++ engine on the two **structural** workloads.

**Repeatability of the headline cell** (8 reps each, `test/manual/bench-compare/repeat_100k.py`):
w1@100k microecs 0.1415–0.1694 ms (spread 1.20×), xecs 0.5210–0.5377 (spread 1.03×) → ratio **3.3×
on medians, 3.7× on mins** (07-27 measurement; this run's min-of-2 gives 3.27×). Same check on w2@100k: **2.5–2.7×**. Read the columnar-at-100k lead as **~2.5–3.7×**, and
prefer the tail rows (200k–1M) where both libs are stable.

## ns/entity per frame @ N=100,000 (the large-N regime, all seven)
| workload | microecs | xecs | entt (C++) | flecs (C) | esper | snecs | ecs-pattern |
|---|--:|--:|--:|--:|--:|--:|--:|
| w1 physics | **1.3** | 4.4 | 221.1 | 736.9 | 99.0 | 257.4 | 97.1 |
| w2 bounce | **3.3** | 8.6 | 371.4 | 1134.3 | 161.4 | 402.7 | 152.4 |
| w3 ai | **3.6** | 5.8 | 129.5 | 391.1 | 62.9 | 146.9 | 51.7 |
| w4 random | **0.1** | 0.9 | 5.8 | 23.9 | 10.7 | 11.4 | 2.6 |
| w5 churn | **108.9** | N/A | 159.6 | 490.8 | 379.3 | 172.8 | 185.9 |
| w6 mixed | **6.3** | 11.6 | 409.3 | 1313.3 | 237.6 | 483.2 | 167.1 |
| w7 migrate | **137.0** | N/A | 151.2 | 501.6 | 428.1 | 166.2 | N/A |

## Does a native core help? best(entt, flecs) / best(pure python)
Both groups run **identical Python work per entity** — components are Python objects either way — so
this isolates the storage engine. `>1` = the native core is **slower**.
| workload | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| w1 physics | 3.17 | 2.99 | 2.94 | 2.65 | 2.28 |
| w2 bounce | 3.13 | 2.90 | 2.82 | 2.79 | 2.44 |
| w3 ai | 3.23 | 3.14 | 3.00 | 3.03 | 2.51 |
| w4 random | 2.22 | 2.53 | 2.79 | 2.58 | 2.22 |
| w5 churn | 1.91 | 2.26 | 2.05 | 1.71 | **0.92** |
| w6 mixed | 3.00 | 3.07 | 2.69 | 2.50 | 2.45 |
| w7 migrate | 1.03 | 1.01 | **0.98** | **0.98** | **0.91** |

A native core costs **~2.2–3.2× on field arithmetic** and only pays off where the work *is* the
data-structure operation (churn at 100k, migration). Binding a world-class C++ ECS lands you *behind*
plain esper on every arithmetic workload — you pay Python's loop **plus** a boundary crossing per
component access.

**w6 settles which side a real frame lands on: 2.45–3.07×, inside the arithmetic band.** A realistic
frame is physics + ai + a few targeted hits, so it is mostly arithmetic — the structural win at 100k
(w5/w7) does not carry it. Where the native core wins, it wins a small slice of the frame.

**Binding maturity, stated honestly** (checked 2026-07-26): the engines are world-class, the Python
bindings are not maintained. **PyEnTT** — 1★, created 2026-03-04, **archived 2026-03-05**. **pyflecs11**
— 2★, last push 2025-08-21. A better binding would shave constant factors; it cannot remove the
per-access crossing or the Python loop, which is what the ~2.4–3× is made of.

## Mechanism probes (verified — see `probes/` and `test/manual/bench-compare/`)

P1–P3 live in the committed `probes/`. P4/P5 and the run-comparison scripts live in
`test/manual/bench-compare/` (`delta.py`, `relative.py`, `paths.py`, `spawn_breakdown.py`,
`repeat_100k.py`), which is **gitignored** — so every number they produced is inlined below and this
file is the record. Promote any of them into `probes/` with a single `mv` if a claim needs to ship.

**P1 — the columnar crossover is NOT a dtype artifact → CONFIRMED (`probes/xecs_dtype.py`).**
`Float32 * python_float` stays float32 in numpy 2.x (NEP 50); forcing a float32 dt gives 0.92–1.03×
for xecs at every N from 1k to 1M. Not upcast memory traffic.

**P2 — xecs' large-N slowdown is the Rust↔numpy copy boundary → CONFIRMED (`probes/boundary.py`).**
`view.x * dt` returns a plain numpy array — xecs does its arithmetic in numpy, not Rust; the Rust core
is storage. `.numpy()` is a COPY (docstring: "Copy the elements into a NumPy array"; no zero-copy
accessor). Per position-integrate step:
| N | xecs (ms) | raw-numpy fused (ms) | `.numpy()` one copy (ms) | xecs / numpy |
|---|--:|--:|--:|--:|
| 10k | 0.0530 | 0.0049 | 0.0046 | 10.8× |
| 100k | 0.3604 | 0.0538 | 0.0514 | 6.7× |
| 500k | 1.7285 | 0.4372 | 0.1761 | 4.0× |
| 1M | 3.6076 | 0.7980 | 0.3529 | 4.5× |

xecs ≈ **4–11× raw numpy** (~6 buffer copies/step). microecs mutates the pool arrays **in place**
(zero-copy), and a single-pool query returns `_QRArray`, so a columnar step is native-C numpy on the
pool array with no per-op object at all. **The pure-Python lib beats the Rust lib at scale because it
has no FFI boundary.**

**P3 — `get_entity` in a hot loop is the trap; batched scatter is the fix → CONFIRMED
(`probes/microecs_random.py`).** Re-measured after `#45`, medians of 3 runs (the pre-`#45` column is
the original measurement):
| N | get_entity ns/hit (pre-`#45`) | get_entity ns/hit | batched-scatter ns/hit | speedup |
|---|--:|--:|--:|--:|
| 1k | 3291 | 2055 | 154 | 13× |
| 20k | 2615 | 2407 | 11 | 216× |
| 100k | 2913 | 3050 | 6 | **503×** |

**`#45` did not move this, and that is the point.** Making the accessor ~1.8× cheaper bought ~1.2 µs at
N=1k and **nothing at N=100k** — at scale this workload is dominated by cache-missing random reads into
the pool column, not by accessor overhead. The trap is the access *pattern*, not the API. Batch it.

Unchanged in character by `#42`/`#43`. Batching needs a static set / a row-map rebuilt after `update()`
(pop-swap reorders rows).

**P4 — churn was per-entity Python, not archetype layout. Both halves are now fixed**
(`test/manual/bench-compare/spawn_breakdown.py`, 2-field archetype, same script across all three runs).

The 07-26 run split one spawn+despawn pair and found the archetype pop-swap everyone blamed was 17% of it,
while **validation was 32% with half of that a literal duplicate**. `#44` deleted the duplicate; `#49` then
went after what was left of the per-entity work in the commit.

| part | 07-26 (neither) | now (#44 + #49) | share of the pair now |
|---|--:|--:|--:|
| full spawn (`add_entity` + `update()`) | 9964 | **5037** | 77% |
| — of which `add_entity` (buffered) | 5843 | **3434** | 52% |
| — of which the commit | 4121 | **1603** | 24% |
| full despawn (`remove_entity` + `update()`) | 2600 | **1492** | 23% |
| **storage work** (`Pool.add_entity` + the pop-swap) | 5382 | 2699 | **41%** |
| **validation** (`_validate_components` + `_defaults_for`) | 4019 (×2) | **1660** (×1) | **25%** |
| **one churn pair** | **12564** | **6556** | 100% |

**A churn pair is 1.92× cheaper than at the 07-26 baseline.** Isolated in-process A/Bs, so machine drift
cancels: `#44` alone is 1.97× on `add_entity` and 1.42× on a pair (`test/manual/churn/task44_ab.py`); `#49`
alone is 1.13× on a spawn, 1.70× on the despawn bookkeeping, 1.20× on a pair
(`test/manual/churn/task49_ab.py`).

Note the shape change: the commit went from 4121 ns (41% of a spawn) to 1603 (32%), and `Pool.add_entity`
from 3227 to 1490 — the dtype check really was most of the "storage" cost, not the numpy row write. What is
left is genuinely irreducible per-entity work: one validation pass, one `Command`, two dict writes, one row
copy. Collapsing *that* needs a batch API, which is [microecs #49 item 3] — measured at a 92× ceiling and
**deliberately deferred** until a non-synthetic workload asks for it.

**P5 — the `Entity` path, post-`#45`** (`test/manual/get-entity-perf/per_operation.py`, N=20k, ns/op,
median of 3 interleaved runs). The `pre-#45` column is P5's original measurement; compare **×floor**,
not absolutes — the harness changed (interleaved, min-of-30, no list building) and it reads ~20% lower
across every row, including rows whose code never changed:

| op | pre-`#45` | ns | ×floor |
|---|--:|--:|--:|
| `pool.data['position'][ix]` (raw numpy floor, read) | 110 | 89 | 1.0 |
| `w.get_entity(eid)` (dict lookup, Entity is cached) | 74 | 39 | 0.4 |
| `ent._locate(['position'])` (**no longer on the read/write path**) | 144 | 140 | 1.6 |
| `ent.position` (read) | 320 | 172 | 1.9 |
| `ent.position = v` (write) | 471 | 247 | 2.8 |
| `ent.set_data(position=v)` (1-field "fast path") | 610 | 505 | 5.7 |
| `ent.set_data(position=v, velocity=v)` (2-field) | 1433 | 1486 | 16.7 |
| composite `e.position = e.position + e.velocity*dt` | 2018 | 1521 | 17.1 |

What changed and what did not:

- **(a) resolved.** `_locate`'s `fields_set.issuperset([name])` is gone from the accessors — `#45` inlined
  the `(pool, row)` lookup into `__getattr__`/`__setattr__` and let the `pool.data[name]` dict lookup *be*
  the field check. Read went 2.9× → **1.9× the raw floor**, write 4.3× → **2.8×**. `_locate` itself is
  unchanged and still serves `set_data`, `get_fields`, `get_components` — where the multi-name
  `issuperset` is the right tool.
- **(b) still true, and now worse in relative terms.** `set_data(f=v)` costs **~2× `e.f = v`** (505 vs 247)
  for the same effect, because only the attribute path got cheaper. The "use `set_data`" advice from the
  `#29` era still does not match the cost for the single-field case.

## What changed since 07-27c — and how it was attributed

- **The field is the control, and it says the machine moved.** The six non-microecs libraries changed no
  code, yet their median shifted **−9% (N=200) to −18% (100k)**. So raw ms deltas are useless here: microecs
  "got 11–21% faster" on w1 purely because the box did.
- **Normalized to the field it is too noisy to attribute alone.** w4's microecs `step` is a bare numpy scatter
  that never calls microecs, yet it moved **+11.5%** vs the field (pure-Python rivals sped up more than numpy
  did). That is the noise band for this pair of runs, and w5's **+24%** sits at its edge.
- **So the attribution is an in-process A/B**, old tree `6fc0d21` vs new tree, same adapters, each cell in a
  fresh subprocess, ABBA order, median of 4 rounds (`test/manual/bench-compare/ab_versions.py`):

| workload | N | step new/old | build new/old |
|---|--:|--:|--:|
| w1 physics | 1k / 20k / 100k | 0.92 / 1.14 / 1.01 | 1.21 / 1.25 / 1.25 |
| w4 random | 1k / 20k / 100k | 1.02 / 1.02 / 0.96 | 1.19 / 1.28 / 1.17 |
| **w5 churn** | 1k / 20k / 100k | **1.24 / 1.31 / 1.26** | 1.26 / 1.28 / 1.25 |
| w6 mixed | 1k / 20k / 100k | 0.98 / 1.02 / 1.02 | 1.21 / 1.26 / 1.28 |
| w7 migrate | 1k / 20k / 100k | 1.08 / 1.16 / 1.05 | 1.22 / 1.32 / 1.26 |

- **Read it as: every spawn is ~25% dearer, nothing else moved.** `build` is N × `add_entity` on every
  workload, and it is +17–32% everywhere; churn is spawn-bound so its step follows. Columnar steps are flat.
- **Where the 25% goes** (`test/manual/bench-compare/spawn_split.py`, 2 components / 3 fields, 3 runs):
  `add_entity` ≈ 6.3 µs; **#50's `_convert_components_data` ≈ 860 ns (14%)** — a Python loop over every
  field even when all data is already an ndarray and nothing converts; **#39's snapshot copy in
  `CommandBuffer.append` ≈ 680 ns (11%)** — a dict comprehension `.copy()`ing every ndarray arg. Sum ≈ 1.5 µs ≈
  the measured delta.
- **Both are deliberate correctness/ergonomics features, not accidents.** #39 fixes a real aliasing bug (a
  scratch array reused in a spawn loop committed the last value N times). The cost is the *shape*: two more
  per-field passes on top of `_validate_components` + `_defaults_for` — exactly the pattern P4 found in July
  (per-entity Python passes, not storage, dominate a spawn). Fusing convert + snapshot + validate into one
  per-field pass is the obvious lever; the #49 batch-spawn API (92× ceiling, deferred) remains the big one.
- **w7 migrate's +5–16%** is #50's conversion running again on `add_component`, plus #39's copy on its args;
  migration is half adds, so it pays about half the churn tax.

## Honest limitations (from the hostile fairness review)
- **Run-to-run drift is ±10–25% on absolutes** (measured above). Only **ratios inside one run**, the
  winner map, and the crossover are portable. Treat any single cell as ±20%.
- GC-off + min-over-reps flatters the object-per-entity libs (a real game cares p99, not min);
  min-of-2 reps at 100k is thin — see the 8-rep repeatability check for the headline cell.
- Object libs use float64 python floats; SoA libs float32 (half the bytes) — a real edge for SoA at
  bandwidth-bound large N, independent of python overhead. microecs vs xecs is fair (both f32).
- min-16 touch floor means small-N w4/w5 touch ~8%/frame, not the advertised ~2%/1%.
- Workloads use 1–2 archetypes (except w7). Heavy **archetype fragmentation** (many tiny pools, where
  `QRField` concat-across-pools degrades) is untested — microecs' theoretical soft spot.
- Verification is an order-independent multiset fingerprint (catches skipped work, not a which-entity
  permutation or a compensating cross-field error). Adequate for "didn't cheat", not a per-entity proof.
- The two native bindings are unmaintained hobby wrappers (above). The *engines* are not on trial here;
  the **"bind a native ECS from Python" strategy** is.

## One-line takeaways
1. No global winner — the fastest library flips by workload AND by N.
2. Columnar crossover ≈ N=1.5–3k: xecs wins below, microecs (in-place numpy) above by ~2.5–3.7× at
   100k and ~1.9× at 1M.
3. **microecs wins broadly at N≥5k** (columnar from 5k, ai from 5k, random access and the mixed frame
   by 20k) — and it wins **all seven workloads at 100k**, structural ones included. At N=200 it still
   loses on fixed per-op cost, by ~1.1–1.6× on columnar and ~8× on churn.
4. Random access: batch it (`col[rows]`), never `get_entity` in a hot loop (up to 503× trap — and a
   cheaper accessor does not fix it; the access pattern is the cost).
5. **The structural workloads were paying per-entity Python, not archetype layout — and both halves are
   fixed.** P4 found 32% of a churn pair was validation with half of it redundant; `#44` removed the
   duplicate and `#49` made the surviving pool check cheap, dropped a copy `update()` threw away, and
   stopped rebuilding pools that empty and refill in one tick. Combined: **w5 −50% and w7 −30% against the
   field**, a churn pair **1.92× cheaper**, and microecs wins all seven workloads at 100k. The
   lesson generalizes past churn — both structural workloads share `_pop_from_pool`/`_add_to_pool`, so
   anything charged per entity there is charged twice on a migration, which is why w7 moved without
   anyone aiming at it. **The 10-18 run is the same lesson in reverse:** #50 + #39 added two per-field passes
   to every spawn and gave back ~25% of it (churn lead at 100k 1.65× → 1.33× EnTT). #56 folded them back into
   one pass with one copy, keeping both features: 1.47× (extrapolated).
6. Capability gaps still decide churn/migration: xecs can't despawn OR migrate; ecs-pattern can't
   migrate. Only microecs/entt/flecs/esper/snecs do all seven — and microecs is no longer paying a
   speed penalty for the privilege at large N.
7. Binding a native ECS is not the shortcut it looks like: entt/flecs are ~2.2–3.2× *slower* than plain
   esper on arithmetic — including the realistic mixed frame (w6, 2.45–3.07×) — and 36–547× slower than
   microecs at 100k. In Python the axis that matters is **vectorized vs per-entity**, not native vs
   interpreted.
