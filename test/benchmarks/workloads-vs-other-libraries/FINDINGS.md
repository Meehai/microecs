# Findings — multi-workload ECS benchmark (raw data + the "why")

Environment: numpy 2.5.1, Python 3.12.12, Linux x86_64. Times = min-over-reps of mean-over-30-frames,
GC off. Every cell verified against a float64 reference. Full data in `results.json` (regenerate with
`./run_benchmark.sh`).

**Run: 2026-07-27c**, microecs at the merge of **#44 + #49** (branch `optimize-49` merged with master).
**Prev runs, both single-change and both superseded by this one:** 07-27a `3ba0e76` (#44 alone) and 07-27b
`3c16d06` (#49 alone). They were developed on branches off the same base, so each was measured against the
same 07-26 baseline `975097c` — an honest one-change-at-a-time pair, but neither described shipped code.
**This run does.**

What is in it: **#44** made `World.add_entity` the sole validator of a spawn (`CommandBuffer.append` used to
repeat the same pass). **#49** made the pool's per-field dtype/shape check a single cheap `raise` instead of
three asserts including an `np.issubdtype`, gave `Pool.add_entity` a dict instead of `**kwargs`, dropped a row
copy the despawn path threw away, and moved empty-pool reclamation to one sweep at the end of `update()`.

**Field: seven libraries**, i.e. all three ways to build an ECS you can drive from Python — numpy-vectorized
(microecs), native core + native columnar store (xecs), native core handing back Python component
objects (entt = C++ EnTT, flecs = C flecs), and pure Python (esper, snecs, ecs-pattern).

**Headline: the two structural workloads halved, and microecs now wins every workload at N=100k.** Normalized
to the field: **w5 churn −50.2%**, **w7 migrate −30.4%**, everything else mixed-sign single digits (drift).
The two changes **compound almost exactly** — #44 alone was −28% on w5 and #49 alone −29%, and
0.72 × 0.71 = 0.51 against the −50% measured here — because they cut *different halves* of the path: #44 the
buffered `add_entity`, #49 the commit. Churn at 100k went from 0.82× EnTT to **1.65×**; migration, which
neither change targeted, went from 0.77× to **1.01×** and is now microecs' from N=1k up.

**Fairness note (SoA vs SoA):** w4/w6 give microecs the SAME columnar scatter idiom xecs gets
(`col[rows] -= DMG`) — both are columnar SoA libs, so both batch. The naive `get_entity(id)` loop is
quantified separately as "the trap" (probe P3), never used in the hot path. The AoS libs
(esper/snecs/ecs-pattern/entt/flecs) legitimately loop with O(1) id lookups / direct object refs.

## Matrix — step ms/frame (lower is better)

### w1 physics (columnar integrate)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0129 | 0.0144 | **0.0201** | **0.0329** | **0.1591** |
| xecs | **0.0117** | **0.0129** | 0.0301 | 0.1034 | 0.4938 |
| entt | 0.0432 | 0.2615 | 1.3205 | 5.2238 | 29.0960 |
| flecs | 0.1255 | 0.7117 | 3.3429 | 14.8654 | 86.4460 |
| esper | 0.0159 | 0.0708 | 0.4855 | 1.9524 | 11.3919 |
| snecs | 0.0463 | 0.2535 | 1.3586 | 5.6937 | 29.5094 |
| ecs-pattern | 0.0229 | 0.1075 | 0.4910 | 2.0790 | 11.6986 |

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
| microecs | 0.0230 | 0.0368 | **0.0447** | **0.0908** | **0.3764** |
| xecs | **0.0175** | **0.0255** | 0.0574 | 0.1654 | 0.9907 |
| entt | 0.0701 | 0.4307 | 2.1330 | 8.5357 | 45.0544 |
| flecs | 0.1719 | 0.9513 | 5.9022 | 24.4903 | 137.0127 |
| esper | 0.0288 | 0.1308 | 0.7882 | 3.3169 | 19.6163 |
| snecs | 0.0840 | 0.3870 | 2.1689 | 9.2710 | 49.5531 |
| ecs-pattern | 0.0415 | 0.1601 | 0.8171 | 3.3918 | 19.3317 |

### w3 ai (per-entity health state machine)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0141 | **0.0188** | **0.0354** | **0.0918** | **0.4220** |
| xecs | 0.0149 | 0.0197 | 0.0430 | 0.1452 | 0.7280 |
| entt | 0.0304 | 0.1219 | 0.7977 | 3.1872 | 16.2600 |
| flecs | 0.0736 | 0.3835 | 2.0646 | 8.9095 | 49.1880 |
| esper | 0.0090 | 0.0386 | 0.2547 | 1.1100 | 7.3280 |
| snecs | 0.0230 | 0.1303 | 0.8456 | 3.2103 | 17.8096 |
| ecs-pattern | **0.0089** | 0.0505 | 0.2707 | 1.0834 | 6.0788 |

### w4 random (K=max(16,n//50) distinct hits/frame; SoA libs scatter, AoS libs loop)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0015 | 0.0017 | **0.0021** | **0.0034** | **0.0100** |
| xecs | 0.0028 | 0.0036 | 0.0068 | 0.0202 | 0.0991 |
| entt | 0.0030 | 0.0034 | 0.0174 | 0.0821 | 1.0334 |
| flecs | 0.0076 | 0.0104 | 0.0564 | 0.2804 | 2.9838 |
| esper | 0.0019 | 0.0026 | 0.0140 | 0.1085 | 1.5593 |
| snecs | 0.0022 | 0.0029 | 0.0155 | 0.1257 | 1.4645 |
| ecs-pattern | **0.0012** | **0.0015** | 0.0062 | 0.0344 | 0.4493 |

### w5 churn (spawn B + FIFO-despawn B/frame + integrate; B=max(16,n//100))
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.1447 | 0.1638 | 0.5164 | 2.1633 | **11.9196** |
| xecs | N/A | N/A | N/A | N/A | N/A  (no despawn) |
| entt | 0.0456 | 0.1693 | 0.8587 | 3.4349 | 19.6309 |
| flecs | 0.1051 | 0.4483 | 2.3067 | 9.9909 | 58.0529 |
| esper | 0.0705 | 0.3363 | 1.7842 | 7.3726 | 45.0817 |
| snecs | 0.0482 | 0.1685 | 0.9013 | 3.7909 | 20.9311 |
| ecs-pattern | **0.0251** | **0.0748** | **0.3986** | **2.0777** | 20.8042 |

### w6 mixed (physics + ai + K targeted damage, fixed set — a realistic frame)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0863 | 0.0964 | 0.1121 | **0.1858** | **0.7296** |
| xecs | 0.0295 | **0.0381** | **0.0841** | 0.2790 | 1.3489 |
| entt | 0.0895 | 0.4155 | 2.1793 | 9.0406 | 51.1328 |
| flecs | 0.2323 | 1.1019 | 5.8218 | 26.7532 | 162.6107 |
| esper | **0.0292** | 0.1343 | 0.7838 | 3.4410 | 27.9684 |
| snecs | 0.0809 | 0.3936 | 2.2290 | 9.5694 | 56.5126 |
| ecs-pattern | 0.0439 | 0.1618 | 0.8053 | 3.7057 | 19.8152 |

### w7 migrate (component add/remove → archetype migration; 2·max(4,n//200) migrations/frame)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.1058 | **0.1246** | **0.4493** | **2.0889** | **19.2336** |
| xecs | N/A | N/A | N/A | N/A | N/A  (no component add/remove) |
| entt | 0.0359 | 0.1597 | 0.8240 | 3.5945 | 19.3569 |
| flecs | 0.0858 | 0.4296 | 2.1494 | 9.6909 | 58.3754 |
| esper | 0.0681 | 0.3155 | 1.8331 | 7.9079 | 49.9080 |
| snecs | **0.0341** | 0.1528 | 0.8721 | 3.7985 | 19.4219 |
| ecs-pattern | N/A | N/A | N/A | N/A | N/A  (fixed inheritance-class entities) |

## Winner map (fastest library)
```
workload      N=200        1k          5k          20k        100k
w1 physics    xecs         xecs        microecs    microecs   microecs
w2 bounce     xecs         xecs        microecs    microecs   microecs
w3 ai         ecs-pattern  microecs    microecs    microecs   microecs
w4 random     ecs-pattern  ecs-pattern microecs    microecs   microecs
w5 churn      ecs-pattern  ecs-pattern ecs-pattern ecs-pattern microecs   <- #44+#49
w6 mixed      esper        xecs        xecs        microecs   microecs
w7 migrate    snecs        microecs    microecs    microecs   microecs   <- #49
```
**microecs now takes the entire 100k column — all seven workloads — for the first time.** The two structural
cells are the new ones: w5 churn @100k (entt → microecs) and **every w7 migrate cell from 1k up**, including
100k where entt had held on. w3 ai @200 flips esper → ecs-pattern, which is a rival-vs-rival tie inside drift
and says nothing about microecs.


## microecs / fastest-competitor, and who that competitor is
`>1` = microecs faster by that factor; `<1` = slower by `1/x`.
```
workload     N=200               1k                  5k                  20k            100k
w1 physics   0.90 xecs           0.90 xecs           1.50 xecs           3.15 xecs      3.10 xecs
w2 bounce    0.76 xecs           0.69 xecs           1.28 xecs           1.82 xecs      2.63 xecs
w3 ai        0.63 ecs-pattern    1.05 xecs           1.21 xecs           1.58 xecs      1.73 xecs
w4 random    0.82 ecs-pattern    0.90 ecs-pattern    2.91 ecs-pattern    5.95 xecs      9.90 xecs
w5 churn     0.17 ecs-pattern    0.46 ecs-pattern    0.77 ecs-pattern    0.96 ecs-pat   1.65 entt
w6 mixed     0.34 esper          0.40 xecs           0.75 xecs           1.50 xecs      1.85 xecs
w7 migrate   0.32 snecs          1.23 snecs          1.83 entt           1.72 entt      1.01 entt
```
**The two structural rows are the story.** w5: 0.08→0.11, 0.22→0.30, 0.36→0.49, 0.46→0.66, **0.82→1.25**.
w7: 0.24→0.31, 0.83→**1.27**, 1.02→**1.69**, 1.07→**1.57**, 0.77→0.96. Every other row is flat within drift.
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
| w1 physics | **1.6** | 4.9 | 291.0 | 864.5 | 113.9 | 295.1 | 117.0 |
| w2 bounce | **3.8** | 9.9 | 450.5 | 1370.1 | 196.2 | 495.5 | 193.3 |
| w3 ai | **4.2** | 7.3 | 162.6 | 491.9 | 73.3 | 178.1 | 60.8 |
| w4 random | **0.1** | 1.0 | 10.3 | 29.8 | 15.6 | 14.6 | 4.5 |
| w5 churn | **119.2** | N/A | 196.3 | 580.5 | 450.8 | 209.3 | 208.0 |
| w6 mixed | **7.3** | 13.5 | 511.3 | 1626.1 | 279.7 | 565.1 | 198.2 |
| w7 migrate | **192.3** | N/A | 193.6 | 583.8 | 499.1 | 194.2 | N/A |

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
| w6 mixed | 3.11 | 2.73 | 2.80 | 2.42 | 2.54 |
| w7 migrate | 1.12 | **0.94** | **0.92** | **0.97** | **0.90** |

A native core costs **~2.3–3.4× on field arithmetic** and only pays off where the work *is* the
data-structure operation (churn at 100k, migration). Binding a world-class C++ ECS lands you *behind*
plain esper on every arithmetic workload — you pay Python's loop **plus** a boundary crossing per
component access.

**w6 settles which side a real frame lands on: 2.42–3.11×, inside the arithmetic band.** A realistic
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

## What changed since the 07-26 baseline — and how it was attributed

Two library changes, developed on separate branches and merged for this run: **#44** (single-pass spawn
validation) and **#49 items 1 + 2** (cheap pool check, no discarded despawn copy, deferred pool reclamation).
Attribution, in order:

- **The field is the control.** Median shift of the **six non-microecs** libraries, 07-26 → now:
  **+3.2% (N=200), +0.8% (1k), +2.3% (5k), +3.3% (20k), +5.8% (100k)**. They did not change a line, so that
  is the noise floor for this pair of runs. Any single cell under ~10% means nothing.
- **Normalized to the field** (microecs ÷ median of the other six, inside each run; negative = improved),
  mean over N: w1 −3.7%, w2 −3.9%, w3 +3.0%, w4 −5.2%, **w5 −50.2%**, w6 −2.6%, **w7 −30.4%**. Only the two
  structural workloads clear the band, and they clear it by an order of magnitude.
- **The two changes compound, and that was checked rather than assumed.** Measured separately against this
  same baseline: #44 alone −28.2% on w5, #49 alone −29.3%. Multiplying the survivors, 0.72 × 0.71 = 0.51,
  against the **−50.2%** measured on the merge. They cut different halves of the path — #44 the buffered
  `add_entity`, #49 the commit — so the near-perfect composition is the expected result, not a coincidence.
- **w7 migrate's −30% is entirely #49's**, and it was not the goal. Component migration is
  `_pop_from_pool` → `_add_to_pool`: the *same* two functions churn uses. It was paying the `np.issubdtype`
  check on every field of the rebuilt row and the `**kwargs` unpack/repack at each call boundary — twice per
  migration, once out and once in. It also stopped tearing down archetypes that empty and refill in one tick.
  #44 cannot touch w7 (it never calls `add_entity`), and indeed #44's own run left w7 flat.
- **Isolated component costs**, in-process A/B: `#44` 1.97× on `add_entity`, 1.42× on a churn pair; `#49`
  1.13× on a spawn, 1.70× on despawn bookkeeping, 1.20× on a pair; together **1.92× on a pair** (P4).
- **Confirmed outside the benchmark.** robosim's own physics tick (`test/e2e/perf-physics-tick`, a real
  simulator, not a workload we wrote) is **2–18% faster on the merge, ~11% on average and biggest where the
  ECS work dominates** (−16% at 50 robots, −17% at 100); its render tick — draw-bound, no ECS mutation —
  stayed flat at 0–3%. The flat control next to a gain that concentrates at high N is what separates a real
  win from machine drift. Note this compares against a stored baseline from another day, so read the shape,
  not the individual cells; the mid-N cells (−2% at N=3) are inside that noise.
- **Method note:** a run is discarded if unrelated work runs concurrently — that inflates *every* library by
  10–30%. All four published runs had an otherwise idle machine.

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
   by 20k) — and as of this run it wins **all seven workloads at 100k**, structural ones included. At
   N=200 it still loses on fixed per-op cost, by ~1.1–1.3× on columnar and ~6× on churn.
4. Random access: batch it (`col[rows]`), never `get_entity` in a hot loop (up to 503× trap — and a
   cheaper accessor does not fix it; the access pattern is the cost).
5. **The structural workloads were paying per-entity Python, not archetype layout — and both halves are
   fixed.** P4 found 32% of a churn pair was validation with half of it redundant; `#44` removed the
   duplicate and `#49` made the surviving pool check cheap, dropped a copy `update()` threw away, and
   stopped rebuilding pools that empty and refill in one tick. Combined: **w5 −50% and w7 −30% against the
   field**, a churn pair **1.92× cheaper**, and **microecs now wins all seven workloads at 100k**. The
   lesson generalizes past churn — both structural workloads share `_pop_from_pool`/`_add_to_pool`, so
   anything charged per entity there is charged twice on a migration, which is why w7 moved without
   anyone aiming at it.
6. Capability gaps still decide churn/migration: xecs can't despawn OR migrate; ecs-pattern can't
   migrate. Only microecs/entt/flecs/esper/snecs do all seven — and microecs is no longer paying a
   speed penalty for the privilege at large N.
7. Binding a native ECS is not the shortcut it looks like: entt/flecs are ~2.3–3.4× *slower* than plain
   esper on arithmetic — including the realistic mixed frame (w6, 2.4–3.1×) — and 39–490× slower than
   microecs at 100k. In Python the axis that matters is **vectorized vs per-entity**, not native vs
   interpreted.
