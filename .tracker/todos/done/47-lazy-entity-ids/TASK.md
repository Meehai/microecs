# `entity_ids` is built eagerly on every cold query — make it a property

**Created**: 2026-07-27
**Closed**: 2026-07-27
**Priority**: 2

## Verdict (2026-07-27): DONE — the premise held, and it is the biggest win of plan 3

A cold query is now **O(pools) instead of O(entities)**. Measured, min of 300, cold each time (the query cache
is cleared before every call):

| N | cold query, before | cold query, after | `len(qr)` after | `.entity_ids` after |
|---|--:|--:|--:|--:|
| 1 000 | 36–44 µs | **3.7–5.6 µs** | 4.7 µs | 28.5 µs |
| 10 000 | 211–218 µs | **3.9–4.1 µs** | 4.8 µs | 219.6 µs |
| 100 000 | 1806–2154 µs | **3.6–4.2 µs** | 4.4 µs | 2133 µs |

**~9× at N=1k, ~53× at N=10k, ~490× at N=100k**, and flat in N instead of linear. When the ids *are* read the
total is unchanged (2133 vs 1806–2154 at N=100k), so nothing is paid twice — the cost simply moved to whoever
actually wants the column. Plan 2's "98% of a cold query at N=10k" was right.

Unlike [#46](../46-in-place-is-the-floor-idiom/TASK.md), whose headline evaporated on re-measurement, this one
survived an interleaved A/B intact.

### What shipped

- `QueryResult.entity_ids` is a `@property` over `_entity_ids`, built from a `{pool: [ids]}` map passed in at
  construction (`pool_ids=self._pool_ids`, `world.py:140`). `world.py` no longer builds the id array eagerly.
- The task's trap 1 — **`__len__` was `len(self.entity_ids)`** — handled: it is `sum(len(p) for p in pool_list)`,
  memoised in `_len`. This mattered: with the naive `__len__` still in place the whole win is invisible to any
  caller that asks for a count (measured 1846 µs at N=100k, against 4.4 after).
- `_QR_INTERNAL_ATTRS` gained `_entity_ids`, `_len`, `_pool_ids`; `entity_ids` moved out of it into the class dict
  as a property, so `QUERY_RESULT_RESERVED_NAMES` still covers it (it unions `vars(QueryResult)`).
- Bonus, unrelated but welcome: `World._cache` → `_qr_cache`. It used to collide by name with `QueryResult._cache`,
  which is a different cache.

Safety of holding World's live `_pool_ids` dict: it only changes on structural ops, and `_qr_cache.clear()`
(`world.py:160`) fires on exactly those, so a lazily-read id column cannot outlive the pools it describes.

### One defect caught in review

`self._pool_ids` was first read inside the property while living only on `World` — so **every** `qr.entity_ids`
raised and 68 tests failed. Worth recording for the shape, not the typo: **an `AttributeError` raised inside a
property falls through to `__getattr__`**, so the error read `'entity_ids' not part of ['position']` instead of
naming the missing `_pool_ids`. Any future failure inside that property will lie the same way.

### Two items from this task NOT done

- **`__repr__` still reads `len(self.entity_ids)`** (trap 2 in the What/How above), so `repr(qr)` materialises the
  column: **1907 µs at N=100k** against 4.2 for `len(qr)`. One line — `len(self)` — and it is gone. Only bites
  code that logs a query result, so a debug-path cost rather than a hot-path one, but it is a live hole.
- **The `assert len(entity_ids) == sum(len(p) …)` from `__init__` was dropped, not moved** (trap 3). It is the
  only check that the id map and the pools still agree, and it matters more now that the two are materialised at
  different times. It only became expressible once `__len__` stopped depending on `entity_ids`, so it can live in
  the property now: `assert len(self._entity_ids) == len(self)`.

### Validation

- 483 pass, 8 xfailed. Three new tests in `test/unit/test_queryresult.py`, over 1 and 2 pools:
  - `test_entity_ids_is_not_built_until_it_is_read` — field reads and `len()` leave `_entity_ids is None`.
    **This is the guard on the win**: anything that quietly touches the column puts the O(N) cost straight back,
    which is exactly how the `__len__` regression was caught.
  - `test_entity_ids_is_built_once_and_cached` — two reads return the same object.
  - `test_len_and_repr_and_ids_agree_including_the_empty_query` — 0/1/2 pools; `len()` and `entity_ids` now come
    from different sources, so their agreement needs pinning, and the no-pool query is the edge case.
- Test-side fallout of the signature change: `_query` in `test_queryresult.py` builds the `{pool: [ids]}` map from
  a flat list so ~70 call sites are untouched; `test_world.py`'s `_QUERYRESULT_RESERVED` now derives from
  `QUERY_RESULT_RESERVED_NAMES` instead of `vars(<a throwaway instance>)` — an instance dict cannot see a property,
  so `entity_ids` had silently dropped out of that guard.
- Docs updated: `primitives.md` (the dropped `entity_ids`-count assert, the reserved-names list, and a note that
  the column is lazy).

## Relates

- Plan 2 Part 3.1 (finding 13) — the measurement this came from. Part 3.2 (refresh-not-clear) is the other half of
  "a query has no lifetime" and is still open.
- Plan 3 item 4 — done.
- [#27](../../open/27-stale-queryresult-guard/TASK.md) — same area, different symptom.
- The `_pool_ids`-as-numpy-array idea (making the property a C-level `np.concatenate`) was **not** needed: the
  remaining `.entity_ids` cost is only paid by callers that actually want the column. Revisit only if it profiles.
