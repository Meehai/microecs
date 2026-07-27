# `entity_ids` is built eagerly on every cold query — make it a property

**Created**: 2026-07-27
**Priority**: 2

Filed from [plan 2](../../../plans/2-app-level-audit-and-mutation-timing.md) Part 3.1 (its top action item,
never filed) — the finding is measured there, this is the task.

## Why

`world.py:136` builds the id column for **every** cold query:

```python
entity_ids = np.array(sum((self._pool_ids[p] for p in res), []), dtype="int64")
```

A python list concat over every matching entity, then a list→ndarray conversion. That makes a cold query
**O(entities)** instead of O(pools), and it dominates:

| N | mask scan | field dicts | **`entity_ids`** | QueryResult | total | vs the system consuming it |
|---|---|---|---|---|---|---|
| 100 | 0.3 µs | 1.4 µs | **4.2 µs** (39%) | 4.9 µs | 10.8 µs | 0.9× |
| 1 000 | 0.2 µs | 0.9 µs | **26.0 µs** (85%) | 3.6 µs | 30.8 µs | 2.6× |
| 10 000 | 0.2 µs | 1.0 µs | **257.8 µs** (98%) | 3.7 µs | 262.7 µs | **12.1×** |

A cached query is flat at 0.36 µs at every N — but the cache is dropped whenever a tick had any structural
change, so the distinct queries are rebuilt every tick, forever.

**Honest scoping**: the win only lands for systems that never read the column. The shooter's motion, wrap,
cooldown and decay systems don't. **robosim mostly does** — 10 of its query sites read `entity_ids` — so
robosim itself gains little from this; it is a large-N, many-systems win.

## What / How

`QueryResult.entity_ids` → a `@property` with a `_entity_ids` cache; `world.py:136` stops running eagerly
and passes the pools instead. Three things that will silently defeat it if missed:

1. **`__len__` (`query_result.py:68`) is `len(self.entity_ids)`** — it would trigger the build on every
   `len(qr)`. Change it to `sum(len(p) for p in self.pool_list)`.
2. **`__repr__` (`query_result.py:71`)** reads it too; use the same sum.
3. **The `assert` in `__init__` (`query_result.py:26`)** validates the ids against the pool lengths — move
   it into the property, where the ids exist.

Also: `entity_ids` is in `_QR_INTERNAL_ATTRS` (`query_result.py:9`), whose `__setattr__` branch does
`super().__setattr__` — that must not collide with a read-only property.

Incremental extra, only if it still shows: keep `_pool_ids` as a numpy array per pool with a size counter
(append and pop-swap stay O(1)), so the property becomes `np.concatenate([p.ids[:len(p)] …])` in C rather
than a python list concat.

## Validation

- A query whose `entity_ids` is never touched must not build it (assert `_entity_ids is None` after using
  `qr.field` and `len(qr)`); touching it twice must build once.
- `len(qr)` and `repr(qr)` stay correct for 0, 1 and 2+ pools — including the empty-query case.
- The moved `assert` still fires on a mismatch.
- Re-measure with plan 2's finding-13 breakdown.

## Relates

- Plan 2 Part 3.1 (finding 13) — the measurement; Part 3.2 (refresh-not-clear) is the *other* half of the
  same "a query has no lifetime" problem and is deliberately **not** in this task.
- [#27](../27-stale-queryresult-guard/TASK.md) — same area, different symptom.
- Plan 3 item 4 — this is the query-build axis, not the step axis.
