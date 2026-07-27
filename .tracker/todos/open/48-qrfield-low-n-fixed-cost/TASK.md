# `QRField` fixed per-op cost: 2+ pools at low N runs 2.3–6× the numpy floor

**Created**: 2026-07-27
**Priority**: 3

**Not ready to start.** Do plan 3 items 1–5 ([#45](../45-entity-accessor-cost-and-recursion/TASK.md),
[#46](../46-in-place-is-the-floor-idiom/TASK.md), [#47](../47-lazy-entity-ids/TASK.md)) first and
re-measure — #46 alone removes a full column copy from this path. Filed now so it has an id, not because
it is next.

## Why

The last real gap on the columnar axis. `pos += vel*dt`, ns/entity (plan 3 Part 1):

| N | floor | best spelling | `qr.f` spellings |
|---|--:|--:|--:|
| 1 000, 2 pools | 1.83 | 4.16 (**2.3×**) | 8.43 – 11.07 (**4.6–6×**) |
| 10 000, 2 pools | 0.72 | 0.68 (**1.0×**) | 1.13 – 1.77 |
| 100 000+, 2 pools | 0.71–0.89 | at the floor | — |

Fixed Python cost, not numpy work — so it vanishes by N=10k and owns the frame below ~2k. **#26** solved
exactly this for the single-pool case by returning an ndarray subclass (`_QRArray`) so operators run as
native C numpy; the ≥2-pool path still builds a `QRField` and dispatches every operator through
`__array_ufunc__`, chunking each operand per pool.

## Is it worth doing? (the open question)

Only if apps really run several archetypes at low N. Evidence says probably yes: the shooter holds **4→8
live archetypes in normal play and 3→10 under churn** at ~100–900 entities — dead centre of this band. But
robosim (~100 entities, few archetypes) is not obviously hurt. **Decide with an app measurement, not from
this table.**

## What makes this urgent or not: #37

[#37](../37-qrarray-qrfield-one-contract/TASK.md)'s two candidate shapes point opposite ways here:

- **"Narrow the subclass"** (keep `_QRArray`, close its entity-axis surface) — this task stays P3, since
  the fast path stays for 1 pool.
- **"Stop returning an ndarray subclass"** (one `QRField` type always) — then **every** query pays this
  cost and #26's win is gone. This task becomes a **prerequisite**, not a follow-up.

So sequence it after #37's shape is chosen, and re-price it then.

## Validation

- **Must not regress 1 pool.** #26's numbers are the thing most at risk from touching this area; re-run its
  archived `run.py` (`todos/done/26-low-n-field-overhead/`) as well as
  `test/manual/get-entity-perf/columnar_gap.py`, interleaved.
- Sweep N = 200 → 100k at 2, 4 and 8 pools — the pool count is the variable this task is about, and no
  current benchmark varies it.

## Relates

- [#26](../../done/26-low-n-field-overhead/TASK.md) — its unfinished half.
- [#37](../37-qrarray-qrfield-one-contract/TASK.md) — decides whether this is optional or mandatory.
- Plan 3 item 6.
