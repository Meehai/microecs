# Three access paths, two targets — how close is each to its own floor?

**Created**: 2026-07-26
**Type**: Perf plan
**Measured on**: `3c229ae` (v0.8.2), numpy 2.2.6, Python 3.12, Intel Ultra 7 165H.
Part 2 re-measured post-#45 on 2026-07-27.
**Evidence**: `test/manual/get-entity-perf/{ladder,per_touch,recursion_fix,columnar_gap,task45_review,`
`per_operation,ratio_stability}.py`. `test/manual/` is gitignored, so every number is inlined here — the plan
is the artifact. The Part 2 ratios are also pinned in CI by `test/integration/test_i_entity_level_regression.py`.

## The objective, stated

Two targets, one per surface — set deliberately, because they are not the same target:

1. **Columnar (`qr.field`, `pool.field`) — state of the art: limited by raw numpy and nothing else.**
   This is where a simulation should live and where microecs' whole thesis sits.
2. **Row / `Entity` (`world.get_entity(eid).field`) — as good as an OOP-shaped handle can be.**
   This is the *good API*: it is what most user-level code reaches for when it addresses one entity.
   It will never be columnar speed; it should be as close to its own floor as possible.

And one explicit **non-goal**: the `zip`-rows pattern. It is a weird middle — not a batch op, not an
entity handle — and it exists because *drawing* has no batch form: you must hand raylib one transform at
a time. Nothing to optimize there; see Part 3.

## TL;DR (grug verdict)

**Target 1 is already met. Target 2's 1.4× win has been taken (#45); the hard limit at ~19× remains.**

- **Columnar is AT the numpy floor** — 1.0–1.1× for N≥10k, verified by an interleaved A/B (the
  non-interleaved runs showed sub-1.0 ratios; that was thermal drift, not a lead over numpy).
- **The idioms we document are 1.3–1.5× off it**, because they allocate temporaries. Docs fix that, for free.
  ~~`qr.f += x` pays a full column self-copy~~ — **withdrawn, see #46**: numpy already short-circuits
  `a[:] = a` (0.4 µs vs 19.6 µs for a real 800 KB copy), so that copy never existed. The one-line identity
  check shipped anyway, but it is worth 1.0–1.1× (a fixed ~1 µs/call), not 2.3×.
- **Two pools at low N is the one real columnar gap**: at N=1000, `QRField` runs 2.3–6× the floor. #26
  fixed this for one pool (`_QRArray`) and left multi-pool.
- **Row access went ~40× → 29×** — [#45](../todos/done/45-entity-accessor-cost-and-recursion/TASK.md)
  **shipped 2026-07-27**, on forecast. **~19× is the floor** — not 16×. `zip` is faster than *any* indexed
  access before microecs does anything: numpy's array iterator is 718 ns/row, `col[i]` is 885 ns/row. The
  remaining 29× → 19× is three dunder dispatches and three id→row lookups per tick; the cache that would
  remove two of them was prototyped and rejected on measurement (see #45).

---

## Part 1 — Columnar: at the floor, with a 2× idiom tax on top

### Why

"Limited by raw numpy" needs a definition, or the number means nothing. The floor is **the same step, on
plain numpy arrays, in place, with the same layout the pools have**: a 2-archetype world is floored by two
N/2 arrays, not one N array. One contiguous buffer is not a fair floor — a 2-archetype world structurally
cannot have it (it is measured anyway, as the unreachable ideal; it comes out the same, so archetype
splitting is not what costs).

### What we measured

`pos += vel*dt`, ns/entity, min of 9 (`columnar_gap.py`). Five spellings:

| N | floor | `pool.f[:]=` | `pos=pool.f; pos+=` | `qr.f=` | `qr.f+=` | `p=qr.f; p+=` |
|---|--:|--:|--:|--:|--:|--:|
| **1 pool** (`_QRArray`) | | | | | | |
| 1 000 | 1.80 | 4.22 | 3.26 | 2.81 | 2.50 | **2.06** |
| 10 000 | 0.43 | 0.94 | **0.54** | 0.88 | 0.65 | 0.55 |
| 100 000 | 0.63 | 1.13 | **0.48** | 1.15 | 0.97 | 0.83 |
| 1 000 000 | 1.05 | 1.41 | 1.04 | 1.66 | 1.11 | **0.89** |
| **2 pools** (`QRField`) | | | | | | |
| 1 000 | 1.83 | 5.14 | **4.16** | 11.07 | 8.45 | 8.43 |
| 10 000 | 0.72 | 1.29 | **0.68** | 1.77 | 1.31 | 1.13 |
| 100 000 | 0.71 | 0.82 | **0.43** | 1.00 | 1.07 | 0.47 |
| 1 000 000 | 0.89 | 1.54 | **0.72** | 1.67 | 1.12 | 0.90 |

Absolutes wander ±20% run to run (benchmarks.md says so, and it is why several cells above read *below*
the floor). So the headline was re-measured **interleaved**, alternating the two on the same thermals:

| N | floor (min/med) | `pos=pool.f; pos+=` (min/med) | ratio |
|---|--:|--:|--:|
| 10 000 | 0.53 / 0.54 | 0.57 / 0.59 | **1.08× / 1.10×** |
| 100 000 | 0.72 / 1.00 | 0.71 / 0.81 | **0.99× / 0.81×** |
| 1 000 000 | 1.20 / 1.26 | 1.21 / 1.29 | **1.01× / 1.02×** |

**Target 1 is met for N≥10k: 1.0–1.1× raw numpy, with the SoA layout, zero copies.** Nothing to win here;
say so in the docs and stop looking.

### The two things that are NOT at the floor

**1. ~~`qr.f += x` copies the whole column onto itself.~~ WRONG — withdrawn 2026-07-27 (#46).**
The value assigned back really *is* the column, but **numpy already short-circuits self-assignment**:
800 KB, `a[:] = a` 0.4 µs vs `a[:] = b` 19.6 µs. There was no copy to delete. The 2.3× (1.07 vs 0.47) was
**non-interleaved noise** from the same run whose own caveat sits three paragraphs above. Interleaved on one
world in one process: 0.56 vs 0.55 = **1.03×**.

The identity check shipped regardless (`query_result.py:61-65`) — it skips the *python* dispatch into
`__setitem__`, a fixed ~1 µs per call: 1.08–1.11× at N≤1000, 1.00–1.02× at N≥100k. A low-N nibble, same
territory as #48. Original reasoning kept below, because the soundness argument is still the reason it is safe:

Fix — one line, and it is **sound**, which is the interesting part:

```python
if value is getattr(self, name):     # `qr.f += x` -> the in-place ufunc returned the cached field
    return                            #    object itself; `col[:] = col` is a no-op
```

Identity is on the **cached field wrapper** (`self._cache[name]`), not on memory, which is why there is no
aliasing hazard. Verified on both surfaces, 1 pool and 2 pools:

| assignment | `value is qr.f`? | correct? |
|---|---|---|
| `qr.f += x` | **True** | skip — the ufunc already wrote in place |
| `qr.f = qr.f` | **True** | skip — genuinely a no-op |
| `qr.f = qr.other_f` | False | copies |
| `qr.f = qr.f + 1` / `.numpy()` / `[:, 0:1]` / a scalar | False | copies |

(Contrast with `Entity`, where the same trick is **not** available: there the value is a *row view*, and
two rows of one column share a base, so identity cannot distinguish "write yourself" from "write your
neighbour". That is why #45 does not attempt it.)

**2. The documented idioms allocate.** `pool.f[:] = pool.f + pool.v*dt` builds `v*dt`, builds the sum, then
copies it in: 2 temporaries and a full column write, ~**2.1–2.4×** the hoisted-local form. Hoisting to a
local (`pos = pool.position; pos += …`) is a true in-place ufunc and *is* the floor.

Why the docs teach the slow one: **`pool.f += x` raises** — `Pool.__setattr__` (`pool.py:76-79`) rejects
every attribute set with *"Use `pool.component[:] = ...`"*. The guard is right (`pool.f = arr` must not
rebind a column) but it also blocks the one spelling that costs nothing, and the message points at the
2-temporary form. `QueryResult` has the same shape but can be fixed by item 1; `Pool` cannot use identity
(its `__getattr__` builds a fresh view per call, so nothing is stable to compare) — so for `Pool` this is a
**docs** fix plus possibly a better message.

**3. The remaining real gap: `QRField` at low N.** N=1000, 2 pools: floor 1.83, best spelling 4.16
(**2.3×**), and the `qr.f` spellings 8.4–11.1 (**4.6–6×**). This is fixed per-op Python cost, not numpy
work — exactly what **#26** attacked and solved *for one pool* by returning an ndarray subclass
(`_QRArray`); the ≥2-pool path still builds a `QRField` and dispatches every operator through
`__array_ufunc__`. Note the interaction with **#37**: plan 2 Part 4 wants the two branches to have *one
contract*, and its chosen direction (narrow `_QRArray` down rather than speed `QRField` up) does not
change this — the multi-pool arithmetic cost stays. Whether it is worth attacking depends on whether apps
run many archetypes at low N. The shooter does (4→8 live archetypes, ~100–900 entities), so the answer is
probably yes, but it is the largest item here and the least certain.

---

## Part 2 — Row access: the 1.4× has been taken (#45), and the wall at ~19× remains

### Why the target is not 16×

`benchmarks.md` prices the three per-entity loops at 16× (`zip`), 19× (`pool-loop`) and 40×
(`get_entity`), which makes 16× look like the goal. It is not reachable, and the reason is below microecs
entirely:

```
numpy array iterator (for p in arr):  718 ns/row
integer index         (arr[i]):       885 ns/row     <- +23%, no ECS in the picture
```

`get_entity` resolves an id → `(pool, index)`, so it **must** index. `pool-loop` indexes too — which is
why it sits at 19×, and why **19× is the floor for any id-addressed path**, measured at 925 ns/entity
(1.21× `zip`) with `(column, index)` already in hand and no ECS at all.

### What is available

Per field touch, and end-to-end for `ent.position += ent.velocity*dt` at N=100k
(full ladder in [#45](../todos/done/45-entity-accessor-cost-and-recursion/TASK.md)):

| | read | write | entity tick | ×zip | docs scale |
|---|--:|--:|--:|--:|--:|
| before #45 | 305 | 392 | 1929 | 2.53 | ~40× |
| inline the accessors (#45) — **SHIPPED 2026-07-27** | **159** | **267** | **1560** | **1.86** | **29×** |
| … and the caller avoids `+=` | 159 | — | 1156 | 1.52 | 24× |
| floor (indexed row access) | ~95 | — | 925 | 1.21 | ~19× |

Roughly **half of every field touch was bookkeeping**: a python call into `_locate`, a one-element list, and
a `set.issuperset` checking a field the next line looks up in `pool.data` anyway. #45 deleted it, and landed
within noise of the forecast (predicted 1.85× / −27%, measured 1.86× / −26%).

The residual 29× → 19× is three `__getattr__`/`__setattr__` dispatches and three id→row dict lookups per
tick. A version-stamped `(pool.data, index)` cache was prototyped to remove two of the three lookups and
**rejected on measurement** — 1416 ns warm (no gain) and 2008 ns cold (worse than today). Recorded in #45
so nobody re-derives it.

### The idiom, again

`ent.position += x` is **three** touches, not two: read, read, and a write-back through `__setattr__` of
the row onto itself (Python's `+=` on an attribute always assigns back). Worth 250 ns of the 1409.
`p = ent.position; p += x` avoids it. Same shape as Part 1's item 1, one level down — and here it can only
be fixed in the *docs*, not the library.

Also still true from #36: **`set_data(f=v)` is slower than `e.f = v`** for the identical effect — 2129 vs
1929 ns/entity — because it pays for the kwargs dict.

---

## Part 3 — `zip`-rows: a non-goal, and why

`for p, v in zip(qr.pos, qr.vel)` is the middle path: no id lookup, no batch op, just numpy's iterator over
each pool's rows. It is the **fastest per-entity loop available** (16×) and it is what **drawing** needs —
raylib takes one transform per call, so there is no batch form to escape to.

Two reasons to leave it alone:

- **It is already within 1.2× of the indexed floor**, and the gap is numpy's iterator being faster than
  numpy's `[i]` — not something microecs can influence.
- **There is nothing in the path to remove.** `QRField.__iter__` is `yield from part` per pool
  (`qr_field.py:109-111`); the cost is the two generator resumptions and the numpy row-view allocations,
  all of which are the work itself.

Worth stating in the docs as a *positive* recommendation rather than the current framing ("if you must
loop, loop right"): **when a per-entity pass is unavoidable — rendering, serialization, anything crossing
into a non-batch API — `zip`-rows is the right tool, and it is at its floor.** Today's docs read as if all
three loops are equally regrettable.

---

## Part 4 — Action list

Ranked by value ÷ effort. Sizes: XS = one line, S = under an hour, M = a session.

All six are filed. Items 1, 2 and 5 are one finding at three sites, so they are one task (#46).

| order | work | task | size | axis | payoff |
|---|---|---|---|---|---|
| 1 | ~~Identity short-circuit in `QueryResult.__setattr__`~~ **DONE 2026-07-27, reduced** | [#46](../todos/done/46-in-place-is-the-floor-idiom/TASK.md) | XS | columnar | premise withdrawn — numpy already short-circuits `a[:] = a`. Shipped for the fixed ~1 µs/call it does save: **1.0–1.1×**, not 2.3× |
| 2 | ~~Docs: the in-place idiom is the floor idiom~~ **DROPPED 2026-07-27, dev's call** | [#46](../todos/done/46-in-place-is-the-floor-idiom/TASK.md) | S | both | still real at **1.3–1.5×** (not ~2×) and still free — refile if wanted |
| 3 | ~~Inline the Entity accessors~~ **DONE 2026-07-27** | [#45](../todos/done/45-entity-accessor-cost-and-recursion/TASK.md) | S | row | 40× → **29×**, as forecast; also fixed the copy/pickle recursion |
| 4 | Lazy `entity_ids` (plan 2, Part 3.1) | [#47](../todos/open/47-lazy-entity-ids/TASK.md) | S | columnar | not the step, the *query*: 98% of a cold query at N=10k, which is 12× the system consuming it |
| 5 | ~~Better `Pool.__setattr__` message~~ **DROPPED 2026-07-27, dev's call** | [#46](../todos/done/46-in-place-is-the-floor-idiom/TASK.md) | XS | columnar | `pool.py:78` keeps recommending the 2-temporary spelling |
| 6 | `QRField` low-N fixed cost (Part 1.3) | [#48](../todos/open/48-qrfield-low-n-fixed-cost/TASK.md) | M | columnar | the last real gap: 2.3–6× the floor at N=1000 with ≥2 pools. Biggest and least certain |

#45 and #46 are done (#46 reduced to item 1; items 2 and 5 dropped). #47 is next. #48 is P3 and explicitly **not ready** — it waits
on the others being re-measured, and on [#37](../todos/open/37-qrarray-qrfield-one-contract/TASK.md)
choosing a shape, which decides whether it is optional or mandatory.

**Not filed, dev's call**: `__slots__` on `Entity` (#45's non-goal). Worth 13 ns, so not a perf item — but
it would let `ENTITY_RESERVED_NAMES` be derived from `vars()` instead of the hand-maintained
`_ENTITY_INTERNAL_ATTRS` set with its *"if Entity gets new fields, add them here!"* comment (verified: slot
descriptors do appear in `vars()`). That is a maintainability argument and belongs to whoever finds that
comment annoying enough.

## Validation

- **Every perf claim gets an interleaved A/B before it becomes a task.** #45 and #46 both shipped against
  forecasts built on non-interleaved runs: #45 landed on its number (2.48 → 1.86 vs a predicted 1.85), #46's
  headline evaporated entirely (a "2.3× self-copy" that numpy had already optimised away). Same measurement
  method, opposite outcomes — so non-interleaved absolutes are not evidence, in either direction.
- **Every item re-measures with the probe that found it**, and the ratio is the number that matters
  (absolutes drift ±20%): `columnar_gap.py` interleaved for items 1/2/6, `task45_review.py` + `per_operation.py` for
  item 3 (done), plan 2's finding-13 breakdown for item 4.
- **The row-access ratios are now pinned by a test**, not just re-measured by hand:
  `test/integration/test_i_entity_level_regression.py` asserts `get_entity ≤ 34× oop-scalar` and `≤ 2.15× zip`,
  with `zip`/`pool-loop` as controls that make it *skip* rather than fail on a noisy machine. Items 1/2 should
  get the columnar equivalent when they land — #45's two review defects were both invisible to the correctness
  suite, and that is the general case for this plan's work, not a one-off.
- **Item 1 needs a correctness test, not a perf test**: the four "must still copy" assignments from Part
  1.1's table, run against both a 1-pool and a 2-pool world, plus `qr.f += x` actually landing. That test
  is the soundness argument, so it must exist before the line ships.
- **Item 6 must not regress 1 pool.** #26's win is the thing most at risk from touching this area.
- Update `docs/source/benchmarks.md` (the `get-entity` row, the per-operation table, and Part 3's framing
  of `zip`) and `primitives.md` (the idioms) as part of items 2 and 3, not afterwards.

## Relates

- [#46](../todos/done/46-in-place-is-the-floor-idiom/TASK.md) — items 1/2/5, **closed 2026-07-27**: item 1
  shipped reduced, items 2 and 5 dropped. Its verdict corrects this plan's Part 1.1.
- [#45](../todos/done/45-entity-accessor-cost-and-recursion/TASK.md) — item 3, **done 2026-07-27**. Its
  postmortem also records what did NOT improve: `get_entity` random access at N=100k is unchanged, because
  there the cost is cache misses, not the accessor.
- [#36](../todos/done/36-optimize-entity-read-write-path/TASK.md) — closed "cost accepted"; its live item 1
  is superseded by #45, its live item 2 (`set_data` slower than `e.f = v`) is still open and still true.
- [#26](../todos/done/26-low-n-field-overhead/TASK.md) — solved item 6 for one pool; item 6 is its
  unfinished half.
- [#37](../todos/open/37-qrarray-qrfield-one-contract/TASK.md) + plan 2 Part 4 — the *contract* side of the
  same `_QRArray`/`QRField` split; the chosen direction there does not change item 6's cost.
- Plan 2 Part 7 — item 4 is its top entry; this plan adds the step-cost axis it did not cover.
- `docs/source/benchmarks.md` — the public version of all of this.
