# The in-place idiom is the floor idiom — stop the self-copy, stop steering people away from it

**Created**: 2026-07-27
**Closed**: 2026-07-27
**Priority**: 2

## Verdict (2026-07-27): item 1 SHIPPED. Items 2 and 3 dropped — dev's call.

**The task's headline premise was wrong.** It claimed `qr.f += x` pays *"a full column self-copy"* worth **2.3×**.
It does not: **numpy already short-circuits self-assignment.**

```
100k x 2 float32 (800 KB):   a[:] = a  ->   0.4 us
                             a[:] = b  ->  19.6 us
```

So `col[:] = col` was already nearly free, on both surfaces (`QRField.__setitem__` hands numpy `part[key] =
same_part`). The 2.3× came from the plan's **non-interleaved** run — the same measurement round whose own caveat
says absolutes wander ±20% and several cells read *below* the floor. Interleaved on the same world, same process:
`qr.f += x` 0.56 vs `p = qr.f; p += x` 0.55 → **1.03×**. There was no gap to close.

### Item 1 — shipped anyway, for the smaller real reason (`query_result.py:61-65`)

```python
if (col := getattr(self, name)) is value:
    return
col[:] = value
```

It does not skip a copy. It skips the **python-level dispatch** into `__setitem__` plus the loop over parts — a
**fixed ~1 µs per call**, so it shows at low N and vanishes at scale. A/B against the old body, interleaved, same
process:

| N | 1 pool | 2 pools |
|---|--:|--:|
| 200 | 1.08× | 1.11× |
| 1 000 | 1.08× | 1.08× |
| 5 000 | 1.03× | 1.08× |
| 10 000 | 1.02× | 1.05× |
| 100 000 | 1.02× | 1.00× |
| 1 000 000 | 0.99× | 1.01× |

Small, consistent, sound, two lines. Kept. **The in-source comment still describes it as avoiding `col[:] = col`;
left as-is per the dev.** If the fixed cost is ever chased properly it belongs with
[#48](../../open/48-qrfield-low-n-fixed-cost/TASK.md), which is the same ~µs-per-call territory.

### Items 2 and 3 — dropped, not done

The finding behind them **does** hold, just smaller than claimed: the documented `pool.f[:] = pool.f + v*dt`
against the hoisted local `pos = pool.f; pos += v*dt`, interleaved, 2 pools —

| N | documented | hoisted | ratio |
|---|--:|--:|--:|
| 1 000 | 5.20 | 3.95 | 1.31× |
| 100 000 | 0.75 | 0.53 | 1.41× |
| 1 000 000 | 1.31 | 0.86 | 1.52× |

1.3–1.5×, not the claimed 2.1–2.4×, and free (no library change). Dropped regardless:

- **Item 2** — `pool.py:78`'s message still recommends `pool.component[:] = ...`. Dev's call, stays.
- **Item 3** — `docs/source/systems.md:24` still teaches `pool.position[:] = pool.position + pool.velocity * DT`.
  Not updated. **Available if wanted**: it is a docs-only 1.3–1.5× and needs no code.

## Validation

- 470 pass, 8 xfailed. Two new tests in `test/unit/test_queryresult.py`, each over 1 pool (`_QRArray`) and 2 pools
  (`QRField`) — different objects, different `__setitem__`:
  - `test_setattr_inplace_add_still_lands_in_the_pools` — the case the short-circuit fires on.
  - `test_setattr_from_another_field_still_copies` — the case it must **not** fire on.
- They pin both sides of the skip, which is the whole risk: a wrong identity check drops a write **silently** —
  no exception, just stale data in the hot path. Mutation-checked (`if True: return` fails both).
- Identity is on the cached wrapper (`self._cache[name]`), one stable object per field per QueryResult, verified
  on both surfaces: only `qr.f += x` and `qr.f = qr.f` compare `True`, and both are genuine no-ops. Another field,
  arithmetic, a scalar, a slice of itself, and the same field off a different QueryResult all compare `False`.

## Lesson

Two perf tasks in a row (#45, #46) where the *shipped* number and the *filed* number disagreed, both because the
original measurement was not interleaved. #45 landed on forecast once re-measured interleaved; #46's headline
evaporated. **Non-interleaved absolutes are not evidence.** Every future perf claim in plan 3 gets an interleaved
A/B before it becomes a task, not after.

## Relates

- Plan 3 items 1, 2, 5 — item 1 done (reduced), items 2 and 5 dropped. Plan corrected with these numbers.
- [#45](../45-entity-accessor-cost-and-recursion/TASK.md) — the entity-side twin; that one held up.
- [#48](../../open/48-qrfield-low-n-fixed-cost/TASK.md) — the fixed-per-call cost item 1 nibbles at.
- [#37](../../open/37-qrarray-qrfield-one-contract/TASK.md) — item 1 touches `__setattr__`, on the
  `_QRArray`/`QRField` boundary.
