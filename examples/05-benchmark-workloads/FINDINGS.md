# Findings — multi-workload ECS benchmark (raw data + the "why")

Environment: numpy 2.5.1, Python 3.12, Linux. Times = min-over-reps of mean-over-30-frames,
GC off. Every cell verified against a float64 reference. Full data in `results.json` (regenerate
with `./run_benchmark.sh`).

**This run is post-`microecs #26`** (the low-N `QueryResult`/`Field` optimization: a single-pool
query now returns a thin `np.ndarray` subclass `_QRArray` instead of a `QRField` — native C numpy
with no per-op dispatch — plus a per-field cache and a lazy `cumsum`). microecs got **faster in every
cell**; the biggest gains are at low N on single-archetype workloads. The columnar crossover vs xecs
moved **down from ~N=10k to ~N=1.5–3k**, and microecs now wins w1/w2 from N=5k and w3 from N=1k
(previous run: only from N≈20k). See "What changed" below.

**Fairness note (SoA vs SoA):** w4/w6 give microecs the SAME columnar scatter idiom xecs gets
(`col[rows] -= DMG`) — both are columnar SoA libs, so both batch. The naive `get_entity(id)` loop
is quantified separately as "the trap" (probe P3), never used in the hot path. The AoS libs
(esper/snecs/ecs-pattern) legitimately loop with O(1) id lookups / direct object refs.

## Matrix — step ms/frame (lower is better)

### w1 physics (columnar integrate)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0113 | 0.0138 | 0.0167 | 0.0301 | 0.1514 |
| xecs | 0.0089 | 0.0124 | 0.0297 | 0.1027 | 0.5017 |
| esper | 0.0145 | 0.0792 | 0.4012 | 1.6008 | 10.037 |
| snecs | 0.0388 | 0.2263 | 1.1595 | 5.0569 | 27.628 |
| ecs-pattern | 0.0219 | 0.0965 | 0.4683 | 1.9126 | 10.180 |

### w1 physics — columnar TAIL, ns/entity (microecs vs xecs)
| N | 100k | 200k | 500k | 1M |
|---|--:|--:|--:|--:|
| microecs | 1.51 | 1.44 | 1.65 | 2.76 |
| xecs | 5.02 | 4.54 | 5.33 | 5.55 |

xecs stays flat ~4.5–5.5 ns/e at every N; microecs falls to ~1.5 ns/e. **Crossover ≈ N=1.5k.**
(w2 bounce tail, same shape: microecs 3.5→6.9 ns/e vs xecs 8.8→11.8 ns/e over 100k→1M.)

### w2 bounce (integrate + np.where wall reflection)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0250 | 0.0328 | 0.0418 | 0.0770 | 0.3512 |
| xecs | 0.0174 | 0.0239 | 0.0579 | 0.1883 | 0.8815 |
| esper | 0.0258 | 0.1326 | 0.6582 | 2.8987 | 16.496 |
| snecs | 0.0724 | 0.3695 | 1.9079 | 8.0214 | 44.341 |
| ecs-pattern | 0.0393 | 0.1589 | 0.7508 | 3.3558 | 16.519 |

### w3 ai (per-entity health state machine)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0131 | 0.0179 | 0.0301 | 0.0823 | 0.3820 |
| xecs | 0.0151 | 0.0201 | 0.0430 | 0.1353 | 0.6371 |
| esper | 0.0084 | 0.0459 | 0.2358 | 0.9160 | 4.9959 |
| snecs | 0.0263 | 0.1362 | 0.7360 | 3.0431 | 15.647 |
| ecs-pattern | 0.0102 | 0.0506 | 0.2554 | 1.0556 | 5.3994 |

### w4 random (K=max(16,n//50) distinct hits/frame; SoA libs scatter, AoS libs loop)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0016 | 0.0017 | 0.0019 | 0.0030 | 0.0096 |
| xecs | 0.0029 | 0.0034 | 0.0070 | 0.0194 | 0.0955 |
| esper | 0.0019 | 0.0024 | 0.0134 | 0.0816 | 0.9970 |
| snecs | 0.0020 | 0.0030 | 0.0152 | 0.0949 | 1.1252 |
| ecs-pattern | 0.0012 | 0.0015 | 0.0062 | 0.0281 | 0.2813 |

### w5 churn (spawn B + FIFO-despawn B/frame + integrate; B=max(16,n//100))
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.2842 | 0.3136 | 0.9235 | 3.7958 | 18.943 |
| xecs | N/A | N/A | N/A | N/A | N/A  (no despawn) |
| esper | 0.0714 | 0.3164 | 1.5973 | 6.3924 | 34.834 |
| snecs | 0.0469 | 0.1673 | 0.8613 | 3.5895 | 17.817 |
| ecs-pattern | 0.0231 | 0.0710 | 0.3669 | 1.8946 | 18.832 |

### w6 mixed (physics + ai + K targeted damage, fixed set — a realistic frame)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.0773 | 0.0833 | 0.0972 | 0.1719 | 0.6454 |
| xecs | 0.0282 | 0.0336 | 0.0823 | 0.2618 | 1.3140 |
| esper | 0.0267 | 0.1219 | 0.6546 | 2.7326 | 21.052 |
| snecs | 0.0735 | 0.3494 | 1.9973 | 8.5442 | 53.440 |
| ecs-pattern | 0.0405 | 0.1485 | 0.7830 | 3.0208 | 17.251 |

### w7 migrate (component add/remove → archetype migration; 2·max(4,n//200) migrations/frame)
| lib | N=200 | 1k | 5k | 20k | 100k |
|---|--:|--:|--:|--:|--:|
| microecs | 0.1104 | 0.1575 | 0.6399 | 2.8967 | 21.563 |
| xecs | N/A | N/A | N/A | N/A | N/A  (no component add/remove) |
| esper | 0.0575 | 0.3028 | 1.6451 | 6.6149 | 36.002 |
| snecs | 0.0327 | 0.1456 | 0.7605 | 3.0572 | 17.071 |
| ecs-pattern | N/A | N/A | N/A | N/A | N/A  (fixed inheritance-class entities) |

## Winner map (fastest library)
```
workload      N=200        1k          5k          20k        100k
w1 physics    xecs         xecs        microecs    microecs   microecs
w2 bounce     xecs         xecs        microecs    microecs   microecs
w3 ai         esper        microecs    microecs    microecs   microecs
w4 random     ecs-pattern  ecs-pattern microecs    microecs   microecs
w5 churn      ecs-pattern  ecs-pattern ecs-pattern ecs-pattern snecs
w6 mixed      esper        xecs        xecs        microecs   microecs
w7 migrate    snecs        snecs       microecs    microecs    snecs
```

Four cells flipped to microecs vs the pre-`#26` run: **w1@5k, w2@5k, w3@1k, w3@5k** (all were xecs).

## microecs / fastest-competitor (>1 = microecs SLOWER; <1 = microecs faster by 1/x)
```
w1 physics  1.27  1.11  0.56  0.29  0.30
w2 bounce   1.44  1.37  0.72  0.41  0.40
w3 ai       1.57  0.89  0.70  0.61  0.60
w4 random   1.36  1.17  0.31  0.15  0.10
w5 churn   12.32  4.42  2.52  2.00  1.06
w6 mixed    2.89  2.48  1.18  0.66  0.49
w7 migrate  3.38  1.08  0.84  0.95  1.26
```

## Podium — winner ▸ runner-up (margin = runner-up / winner)
The 2nd-place library and how decisive the win is. `A▸B x` = A won, B was 2nd, A is x· faster than B.
```
workload    N=200                  1k                   5k                    20k                  100k
w1 physics  xecs▸microecs 1.27     xecs▸microecs 1.11   microecs▸xecs 1.78    microecs▸xecs 3.41   microecs▸xecs 3.31
w2 bounce   xecs▸microecs 1.44     xecs▸microecs 1.37   microecs▸xecs 1.38    microecs▸xecs 2.45   microecs▸xecs 2.51
w3 ai       esper▸ecs-pat 1.22     microecs▸xecs 1.12   microecs▸xecs 1.43    microecs▸xecs 1.64   microecs▸xecs 1.67
w4 random   ecs-pat▸microecs 1.36  ecs-pat▸microecs 1.17 microecs▸ecs-pat 3.18 microecs▸xecs 6.55  microecs▸xecs 9.91
w5 churn    ecs-pat▸snecs 2.03     ecs-pat▸snecs 2.36   ecs-pat▸snecs 2.35    ecs-pat▸snecs 1.89   snecs▸ecs-pat 1.06
w6 mixed    esper▸xecs 1.05        xecs▸microecs 2.48   xecs▸microecs 1.18    microecs▸xecs 1.52   microecs▸xecs 2.04
w7 migrate  snecs▸esper 1.76       snecs▸microecs 1.08  microecs▸snecs 1.19   microecs▸snecs 1.06  snecs▸microecs 1.26
```
Where microecs is *not* 1st: it's #2 (the runner-up) in most N≥1k cells — the exceptions are w3@200
(#3, behind esper/ecs-pattern), w5 churn (#3–#4, the structural-churn weak spot), and w6@200 (#5,
the fixed per-op cost still shows at the smallest N). xecs is microecs' nearest rival in every
columnar/composite cell; snecs is the nearest rival on migration.

## Mechanism probes (verified, adversarially checked — see `probes/`)

**P1 — columnar crossover is NOT a dtype artifact → CONFIRMED (`probes/xecs_dtype.py`).**
`Float32 * python_float` stays float32 in numpy 2.x (NEP 50); a float32 dt gives 0.88–1.00×
(no speedup) for xecs at every N. So the crossover is not upcast memory traffic.

**P2 — xecs' large-N slowdown is the Rust↔numpy copy boundary → CONFIRMED (`probes/boundary.py`).**
`view.x * dt` returns a plain numpy array — xecs does its arithmetic in numpy, not Rust; the Rust
core is storage. `.numpy()` is a COPY (docstring: "Copy the elements into a NumPy array"; no
zero-copy accessor). Per position-integrate step:
| N | xecs (ms) | raw-numpy fused (ms) | `.numpy()` one copy (ms) | xecs / numpy |
|---|--:|--:|--:|--:|
| 100k | 0.341 | 0.063 | 0.032 | 5.42× |
| 500k | 1.742 | 0.415 | 0.179 | 4.20× |
| 1M | 3.695 | 0.929 | 0.352 | 3.98× |
xecs ≈ **4–8× raw numpy** (the ~6 buffer copies/step). microecs' QueryResult/Field operate
**in-place on the pool arrays** (zero-copy) → the pure-python+numpy lib beats the Rust lib at scale
*because it has no FFI boundary*. Post-`#26`, the single-pool path skips even the `QRField` object,
so a columnar step is native-C numpy directly on the pool array.

**P3 — microecs' random-access `get_entity` is the trap; batched scatter is the fix → CONFIRMED
(`probes/microecs_random.py`).** (Unchanged by `#26`, which touched `QueryResult`, not `get_entity`.)
| N | get_entity ns/hit | batched-scatter ns/hit | speedup |
|---|--:|--:|--:|
| 1k | 2930 | 95 | 31× |
| 20k | 2551 | 11 | 233× |
| 100k | 2938 | 6 | 459× |
`get_entity(id).f` in a hot loop = Entity view + `__getattr__` dict-lookup ~2900 ns/hit. Batched
`col[rows] -= DMG` = 6 ns/hit — the fair SoA number now in the w4/w6 matrix. Caveat: batching
needs a static set / row-map rebuilt after `update()` (pop-swap reorders rows).

## What changed since the last run (post-`microecs #26`)
- microecs is **faster in every cell.** Low-N single-archetype gains are the largest: w1 physics
  N=200 0.0370→0.0113 ms (3.3× faster), w3 ai N=200 0.0695→0.0131 (5.3×), w6 mixed N=200
  0.1378→0.0773 (1.8×). The mechanism: a one-pool query no longer builds a `QRField` (per-op
  `__array_ufunc__` dispatch + fresh object per access) — it returns `_QRArray`, a numpy view that
  runs `+`/`*` as native C, so the fixed per-op cost that dominated at low N is gone.
- **Columnar crossover vs xecs: ~10k → ~1.5k (physics) / ~2.5–3k (bounce).** Below it xecs' leaner
  path still wins; above it microecs wins (~3.3× at 100k physics, unchanged at the top end).
- **The large-N regime is unchanged** — microecs already had no per-op overhead to amortize there.
- **Churn (w5) and small-N still lose.** Structural churn is archetype-SoA's inherent tax (pop-swap
  + realloc), untouched by `#26`; at N=200 the residual fixed cost still puts microecs behind the
  leanest lib on most workloads.

## Honest limitations (from the hostile fairness review)
- GC-off + min-over-reps flatters the object-per-entity libs (a real game cares p99, not min);
  min-of-2 reps at 100k is thin. A GC-on p50/p99 long-run variant would be the fairer "real session".
- Object libs use float64 python floats; SoA libs float32 (half the bytes) — a real edge for SoA at
  bandwidth-bound large N, independent of python overhead. microecs vs xecs is fair (both f32).
- min-16 touch floor means small-N w4/w5 touch ~8%/frame, not the advertised ~2%/1%.
- Verification is an order-independent multiset fingerprint (catches skipped work, not a which-entity
  permutation or a compensating cross-field error). Adequate for "didn't cheat", not a per-entity proof.
- Absolute ms are this-machine, this-run (numpy 2.5.1, Py 3.12); the portable results are the
  **ratios, winner map, and crossover**, which are stable run-to-run.

## One-line takeaways
1. No global winner — the fastest library flips by workload AND by N.
2. Columnar crossover ≈ N=1.5–3k (was ~10k pre-`#26`): xecs wins below, microecs (in-place numpy) above (~3×).
3. **microecs wins broadly at N≥5k** (columnar from 5k, ai from 1k, random-access/mixed by 20k) — the large-N regime is its house.
4. At N=200 microecs still loses (fixed per-op cost), but by only ~1.3–1.6× on columnar now, not ~4–6×.
5. Random access: batch it (`col[rows]`), never `get_entity` in a hot loop (up to 459× trap).
6. Capability gaps decide churn/migration: xecs can't despawn OR migrate; ecs-pattern can't migrate.
   Only microecs/esper/snecs migrate; snecs (sparse-set) is the migration champ, microecs mid (archetype copy tax).
