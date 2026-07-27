# Findings — multi-workload ECS benchmark (raw data + the "why")

Environment: numpy 2.5.1, Python 3.12.12, Linux x86_64. Times = min-over-reps of mean-over-30-frames,
GC off. Every cell verified against a float64 reference. Full data in `results.json` (regenerate with
`./run_benchmark.sh`).

**Run: 2026-07-26**, microecs at `975097c` (v0.8.2 — after `#42` eager `e.field = v` / `set_data`,
`#43` duplicate-component rejection + idempotent remove, and the `_locate` guard). **Field: seven
libraries**, i.e. all three ways to build an ECS you can drive from Python — numpy-vectorized
(microecs), native core + native columnar store (xecs), native core handing back Python component
objects (entt = C++ EnTT, flecs = C flecs), and pure Python (esper, snecs, ecs-pattern).

**Headline: nothing moved.** The library changes since the previous run (2026-07-15/25) are not visible
in this benchmark — see "What changed" below, which is mostly a lesson about machine drift. What *is*
new is the **mechanism** behind microecs' one genuine loss (w5 churn): it is **not** mainly the
archetype pop-swap. It is the **spawn path, and a quarter of that spawn is validation done twice**
(probe P4).

**Fairness note (SoA vs SoA):** w4/w6 give microecs the SAME columnar scatter idiom xecs gets
(`col[rows] -= DMG`) — both are columnar SoA libs, so both batch. The naive `get_entity(id)` loop is
quantified separately as "the trap" (probe P3), never used in the hot path. The AoS libs
(esper/snecs/ecs-pattern/entt/flecs) legitimately loop with O(1) id lookups / direct object refs.

## Matrix — step ms/frame (lower is better)

### w1 physics (columnar integrate)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0145 | 0.0145 | **0.0183** | **0.0313** | **0.1697** |
| xecs | **0.0104** | **0.0122** | 0.0306 | 0.1049 | 0.4263 |
| entt | 0.0499 | 0.2333 | 1.2510 | 5.0580 | 25.5217 |
| flecs | 0.1158 | 0.6875 | 3.6167 | 14.2127 | 82.9456 |
| esper | 0.0145 | 0.0745 | 0.4859 | 1.8569 | 10.5916 |
| snecs | 0.0473 | 0.2321 | 1.3042 | 5.3456 | 28.6224 |
| ecs-pattern | 0.0242 | 0.0942 | 0.5241 | 2.1787 | 10.8736 |

### w1 physics — columnar TAIL, ns/entity (microecs vs xecs)
| N | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|
| microecs | **1.70** | **1.62** | **1.72** | **2.66** |
| xecs | 4.26 | 5.34 | 5.42 | 5.72 |

xecs is flat at ~4.3–5.7 ns/e everywhere; microecs falls to ~1.7 ns/e as fixed overhead amortizes.
**Crossover ≈ N=1.5–3k.** (w2 bounce tail, same shape: microecs 4.0→6.9 ns/e vs xecs 9.6→12.0 over
100k→1M.)

### w2 bounce (integrate + np.where wall reflection)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0264 | 0.0323 | **0.0471** | **0.0931** | **0.3990** |
| xecs | **0.0187** | **0.0253** | 0.0561 | 0.1984 | 0.9645 |
| entt | 0.0796 | 0.4077 | 2.0493 | 7.9879 | 41.4860 |
| flecs | 0.2122 | 1.1848 | 5.5804 | 25.6873 | 139.2927 |
| esper | 0.0238 | 0.1341 | 0.7701 | 3.0253 | 17.9182 |
| snecs | 0.0729 | 0.3990 | 2.1539 | 8.7209 | 46.8259 |
| ecs-pattern | 0.0423 | 0.1614 | 0.8246 | 3.5846 | 18.2284 |

### w3 ai (per-entity health state machine)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0150 | 0.0190 | **0.0335** | **0.0838** | **0.4373** |
| xecs | 0.0144 | **0.0188** | 0.0441 | 0.1380 | 0.6634 |
| entt | 0.0267 | 0.1410 | 0.7544 | 3.1011 | 17.0003 |
| flecs | 0.0768 | 0.4208 | 2.1207 | 9.5509 | 49.7280 |
| esper | **0.0085** | 0.0437 | 0.2514 | 1.0530 | 6.8075 |
| snecs | 0.0249 | 0.1455 | 0.7559 | 3.2486 | 16.8864 |
| ecs-pattern | 0.0105 | 0.0511 | 0.2849 | 1.1176 | 5.7713 |

### w4 random (K=max(16,n//50) distinct hits/frame; SoA libs scatter, AoS libs loop)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0016 | 0.0018 | **0.0021** | **0.0032** | **0.0091** |
| xecs | 0.0030 | 0.0035 | 0.0071 | 0.0187 | 0.0899 |
| entt | 0.0028 | 0.0037 | 0.0180 | 0.0792 | 0.7233 |
| flecs | 0.0070 | 0.0106 | 0.0567 | 0.3081 | 2.7624 |
| esper | 0.0020 | 0.0027 | 0.0132 | 0.0763 | 1.4553 |
| snecs | 0.0022 | 0.0030 | 0.0152 | 0.1013 | 1.4873 |
| ecs-pattern | **0.0013** | **0.0016** | 0.0062 | 0.0274 | 0.3165 |

### w5 churn (spawn B + FIFO-despawn B/frame + integrate; B=max(16,n//100))
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.3079 | 0.3267 | 1.0139 | 4.2622 | 21.9194 |
| xecs | N/A | N/A | N/A | N/A | N/A  (no despawn) |
| entt | 0.0492 | 0.1626 | 0.8404 | 3.6742 | **17.9959** |
| flecs | 0.1190 | 0.4503 | 2.1575 | 9.3525 | 55.4085 |
| esper | 0.0821 | 0.3030 | 1.6672 | 7.6156 | 41.4774 |
| snecs | 0.0492 | 0.1620 | 0.8740 | 3.6650 | 20.0050 |
| ecs-pattern | **0.0241** | **0.0707** | **0.3677** | **1.9468** | 20.9481 |

### w6 mixed (physics + ai + K targeted damage, fixed set — a realistic frame)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0792 | 0.0849 | 0.1044 | **0.1795** | **0.7500** |
| xecs | 0.0291 | **0.0378** | **0.0805** | 0.2869 | 1.3462 |
| entt | 0.0824 | 0.3392 | 2.1506 | 8.0821 | 49.8469 |
| flecs | 0.2202 | 1.1197 | 5.9010 | 25.5605 | 150.0273 |
| esper | **0.0265** | 0.1242 | 0.7687 | 3.3342 | 24.9795 |
| snecs | 0.0706 | 0.3616 | 2.0893 | 9.5374 | 54.8804 |
| ecs-pattern | 0.0413 | 0.1525 | 0.7673 | 3.4731 | 19.6554 |

### w7 migrate (component add/remove → archetype migration; 2·max(4,n//200) migrations/frame)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.1270 | 0.1749 | **0.7395** | **3.4353** | 22.7338 |
| xecs | N/A | N/A | N/A | N/A | N/A  (no component add/remove) |
| entt | 0.0342 | **0.1457** | 0.7578 | 3.6888 | **17.5890** |
| flecs | 0.0830 | 0.4267 | 2.1348 | 10.0613 | 62.0715 |
| esper | 0.0635 | 0.2933 | 1.7713 | 8.2190 | 47.7665 |
| snecs | **0.0306** | 0.1555 | 0.8256 | 3.7979 | 19.4444 |
| ecs-pattern | N/A | N/A | N/A | N/A | N/A  (fixed inheritance-class entities) |

## Winner map (fastest library)
```
workload      N=200        1k          5k          20k        100k
w1 physics    xecs         xecs        microecs    microecs   microecs
w2 bounce     xecs         xecs        microecs    microecs   microecs
w3 ai         esper        xecs        microecs    microecs   microecs
w4 random     ecs-pattern  ecs-pattern microecs    microecs   microecs
w5 churn      ecs-pattern  ecs-pattern ecs-pattern ecs-pattern entt
w6 mixed      esper        xecs        xecs        microecs    microecs
w7 migrate    snecs        entt        microecs    microecs   entt
```
Exactly **one cell** differs from the previous run: w3 ai @1k (microecs → xecs), and that cell is a tie
— 0.0190 vs 0.0188 ms, a 1% gap. Everything else is identical.

## microecs / fastest-competitor, and who that competitor is
`>1` = microecs faster by that factor; `<1` = slower by `1/x`.
```
workload     N=200               1k                  5k                  20k            100k
w1 physics   0.71 xecs           0.84 xecs           1.68 xecs           3.35 xecs      2.51 xecs
w2 bounce    0.71 xecs           0.78 xecs           1.19 xecs           2.13 xecs      2.42 xecs
w3 ai        0.56 esper          0.99 xecs           1.32 xecs           1.65 xecs      1.52 xecs
w4 random    0.81 ecs-pattern    0.88 ecs-pattern    2.96 ecs-pattern    5.82 xecs      9.91 xecs
w5 churn     0.08 ecs-pattern    0.22 ecs-pattern    0.36 ecs-pattern    0.46 ecs-pat   0.82 entt
w6 mixed     0.33 esper          0.44 xecs           0.77 xecs           1.60 xecs      1.80 xecs
w7 migrate   0.24 snecs          0.83 entt           1.02 entt           1.07 entt      0.77 entt
```
The rival on every columnar/branchy/random workload is another **vectorized** library (xecs), never a
native per-entity one. microecs only loses to a C/C++ engine on the two **structural** workloads.

**Repeatability of the headline cell** (8 reps each, `test/manual/bench-compare/repeat_100k.py`):
w1@100k microecs 0.1415–0.1694 ms (spread 1.20×), xecs 0.5210–0.5377 (spread 1.03×) → ratio **3.3×
on medians, 3.7× on mins**. The 2.51× in the table above is that run's min-of-2 for xecs landing low
(0.4263). Same check on w2@100k: **2.5–2.7×**. Read the columnar-at-100k lead as **~2.5–3.7×**, and
prefer the tail rows (200k–1M) where both libs are stable.

## ns/entity per frame @ N=100,000 (the large-N regime, all seven)
| workload | microecs | xecs | entt (C++) | flecs (C) | esper | snecs | ecs-pattern |
|---|--:|--:|--:|--:|--:|--:|--:|
| w1 physics | **1.7** | 4.3 | 255.2 | 829.5 | 105.9 | 286.2 | 108.7 |
| w2 bounce | **4.0** | 9.6 | 414.9 | 1392.9 | 179.2 | 468.3 | 182.3 |
| w3 ai | **4.4** | 6.6 | 170.0 | 497.3 | 68.1 | 168.9 | 57.7 |
| w4 random | **0.1** | 0.9 | 7.2 | 27.6 | 14.6 | 14.9 | 3.2 |
| w5 churn | 219.2 | N/A | **180.0** | 554.1 | 414.8 | 200.1 | 209.5 |
| w6 mixed | **7.5** | 13.5 | 498.5 | 1500.3 | 249.8 | 548.8 | 196.6 |
| w7 migrate | 227.3 | N/A | **175.9** | 620.7 | 477.7 | 194.4 | N/A |

## Does a native core help? best(entt, flecs) / best(pure python)
Both groups run **identical Python work per entity** — components are Python objects either way — so
this isolates the storage engine. `>1` = the native core is **slower**.
| workload | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| w1 physics | 3.43 | 3.13 | 2.57 | 2.72 | 2.41 |
| w2 bounce | 3.35 | 3.04 | 2.66 | 2.64 | 2.32 |
| w3 ai | 3.15 | 3.23 | 3.00 | 2.94 | 2.95 |
| w4 random | 2.21 | 2.41 | 2.89 | 2.89 | 2.29 |
| w5 churn | 2.04 | 2.30 | 2.29 | 1.89 | **0.90** |
| w7 migrate | 1.12 | **0.94** | **0.92** | **0.97** | **0.90** |

A native core costs **~2.3–3.4× on field arithmetic** and only pays off where the work *is* the
data-structure operation (churn at 100k, migration). Binding a world-class C++ ECS lands you *behind*
plain esper on every arithmetic workload — you pay Python's loop **plus** a boundary crossing per
component access.

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

**P4 (new) — w5 churn is NOT mainly the pop-swap; it is the spawn path, and a quarter of the spawn is
validation done twice** (`test/manual/bench-compare/spawn_breakdown.py`). One churn pair costs
**12.6 µs**:
| part | ns | share of the pair |
|---|--:|--:|
| full spawn (`add_entity` + `update()`) | 9964 | 79% |
| — of which `add_entity` (buffered) | 5843 | 46% |
| — of which the commit | 4121 | 33% |
| full despawn (`remove_entity` + `update()`) | 2600 | 21% |
| **storage work** (`Pool.add_entity` 3227 + `_pop_from_pool` 2155) | 5382 | **43%** |
| **validation** (`_validate_components` 1627 ×2 + `_defaults_for` 383 ×2) | 4019 | **32%** |

`World.add_entity` validates and computes defaults, then `CommandBuffer.append` **validates and
computes defaults again** for the same command (plus a `fk = {k: v …}` dict rebuild, ~0.26 µs). The
redundant half is **2.0 µs = 34% of `add_entity`, 20% of a full spawn, 16% of a churn pair** — available
for free, and w5 churn is microecs' worst workload at every N. The pop-swap the plan used to blame is
2.2 µs, **17%** of the pair.

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

## What changed since the previous run (2026-07-15 / 07-25) — and the lesson

The library changes in between (`#42` eager `e.field = v` + eager `set_data`, `#43` duplicate-component
rejection + idempotent remove, `_locate` as the shared guard, better error messages) are **not visible
in this benchmark**:

- **Absolute ms are not comparable across days on this machine.** Median shift of the **six
  non-microecs** libraries, previous run → this run: **+7.1% (N=200), +6.5% (1k), +11.7% (5k), −0.3%
  (20k), +3.3% (100k)**. Those libraries did not change a line. Any per-cell delta below ~15% is
  machine state (thermals/governor), not code.
- **Normalized to the field** (microecs ÷ median of the other six, inside each run), microecs moved
  by mixed-sign single digits on six of seven workloads: w1 +2.4%, w2 +4.8%, w3 +1.0%, w4 −2.6%,
  w5 +4.8%, w6 +3.3%, w7 +3.1% (mean over N). Only w7's five cells share a sign, and at +3% mean.
- **The one thing `#43` demonstrably costs is 50 ns** (`entity_id not in live_entities` +
  `removed_this_tick.add`) = **1.9% of a full despawn**, 0.4% of a churn pair. That is the honest price
  of same-tick idempotent removal, and it is below the harness' resolution.
- **Method note:** a first re-run was discarded because unrelated work (network fetches, file edits)
  ran concurrently and inflated *every* library by 10–30%. The published run had an otherwise idle
  machine.

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
   100k and ~2.1× at 1M.
3. **microecs wins broadly at N≥5k** (columnar from 5k, ai from 5k, random access and the mixed frame
   by 20k). At N=200 it still loses on fixed per-op cost, by ~1.4× on columnar.
4. Random access: batch it (`col[rows]`), never `get_entity` in a hot loop (up to 503× trap — and a
   cheaper accessor does not fix it; the access pattern is the cost).
5. **Churn is a validation problem, not a layout problem** (P4): 32% of a churn pair is validation,
   half of it redundant. Fixing that is microecs' single largest available win.
6. Capability gaps still decide churn/migration: xecs can't despawn OR migrate; ecs-pattern can't
   migrate. Only microecs/entt/flecs/esper/snecs do all seven.
7. Binding a native ECS is not the shortcut it looks like: entt/flecs are ~2.3–3.4× *slower* than plain
   esper on arithmetic, and 39–490× slower than microecs at 100k. In Python the axis that matters is
   **vectorized vs per-entity**, not native vs interpreted.
