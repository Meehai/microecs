# QueryResult `entity_ids` (+ id-map refactor) + robosim migration

**Created**: 2026-06-04
**Updated**: 2026-06-04 (merged the `_pool_ids` sidecar refactor in as step 1; was briefly split to task 7)
**Priority**: 2
**Status**: Open

## Why

Split out of task 5 (QueryResult return type, ✅ done). The return type is complete; the remaining work is an
entity-id story so robosim can move off the old iterate-pools API. Three steps, in order: refactor the reverse
id map → expose `.entity_ids` → migrate robosim.

## Step 1 — `_pool_ix_to_eid` → `_pool_ids` sidecar (the data structure)

The reverse id map is a tuple-keyed dict today: `self._pool_ix_to_eid: dict[tuple[Pool, int], EntityId]`
(`world.py:28`). Replace it with a **sidecar list that mirrors each pool's own dynamic array**:

```python
self._pool_ids: dict[Pool, list[EntityId]]   # _pool_ids[pool][ix] == id of entity at pool index ix
```

Real simplification, not just `.entity_ids` plumbing:
- **Less memory / no hashing** — a flat `list[int]` per pool beats N `(Pool, int)` tuple keys.
- **Clearer popswap** — mirror the pool's swap instead of the current `_pool_ix_to_eid.pop((old_pool,
  len(old_pool)))` that relies on `len` being read *after* `pop_entity` shrank the pool.

Keep `_eid_to_pool_ix` (id → location): `get_entity` / `remove` still need O(1) id lookup. Only the **reverse**
map (location → id) changes. Three touch sites (`world.py:28,100,107-110`):

- [ ] **Declare** `self._pool_ids: dict[Pool, list[EntityId]] = {}` (replaces `_pool_ix_to_eid`, `world.py:28`).
- [ ] **Init on pool creation** in `_get_entity_pool` (`world.py:144-151`), next to `pool_to_components`:
      `self._pool_ids[self.pools[key]] = []`.
- [ ] **`_add_to_pool`** (`world.py:95-100`) → append (pool index is always the tail):
      `self._pool_ids[pool].append(entity_id)`.
- [ ] **`_pop_from_pool`** (`world.py:102-114`) → mirror the pool popswap, and on empty-pool cleanup also
      `del self._pool_ids[old_pool]`:
      ```python
      ids = self._pool_ids[old_pool]
      moved_id = ids[-1]                 # id popswap moved into pool_ix (== entity_id if it was last)
      ids[pool_ix] = moved_id
      ids.pop()
      if moved_id != entity_id:
          self._eid_to_pool_ix[moved_id] = (old_pool, pool_ix)
      ```
- [ ] **No behaviour change** to `get_entity` / `add_component` / `remove_component` / eager-lazy buffer — they
      go through `_eid_to_pool_ix` / `_pop_from_pool` / `_add_to_pool`, untouched in contract.

## Step 2 — `.entity_ids` accessor on `QueryResult`

- [ ] **`entity_ids` accessor** — reserved field name. `World` threads ids at query time from `_pool_ids`,
      aligned to pool+index order (pools have no id concept), so it lines up with the `qr.field` parts:
      `np.array(sum((self._pool_ids[p] for p in res), []), dtype="int64")` (`[]` if no pools match). Same
      pool-by-pool order as `qr.field`, so `zip(qr.entity_ids, qr.position)` lines up.
- [ ] **Flat `(N,)` array, NOT a `_Field`** — ids aren't in pools (no zero-copy view), are read-only, and need
      `[i]` / `np.isin` / fancy indexing for id-based ops (e.g. "remove these") — exactly what `_Field` forbids.
- [ ] **Reserved-name guard** — add `entity_ids` (and QueryResult's own internals: `pool_list`,
      `field_shapes`, `field_dtypes`, `len`) to a reserved set so a component field can't silently shadow the
      accessor (same shadowing class as `Pool.RESERVED_NAMES`).

## Step 3 — migrate robosim

- [ ] **Migrate robosim `server_ecs.py`** to the new API — heavier than examples 01/02: relies on
      iterate-pools, `[0]`, and `len` == pool-count. Needs `entity_ids` (step 2) first.

## Done when

- [ ] `_pool_ix_to_eid` is gone; `_pool_ids` is the single reverse structure; `_pool_ids[pool]` stays
      index-aligned with the pool's arrays through add / remove / swap / component add+remove; empty-pool
      removal drops the list entry (no leak). All existing tests green.
- [ ] `qr.entity_ids` is a flat `(N,)` array aligned to `qr.field`; non-queried/shadowing names are guarded.
- [ ] robosim runs on the new API with no iterate-pools / `[0]` / `len`-as-pool-count.

## Tests (tester owns these)

`test/unit/test_world.py`:
- [ ] **ids stay aligned through churn** — add several, remove from the middle (forces popswap), assert every
      live id round-trips: `_pool_ids[pool][ix]` agrees with `_eid_to_pool_ix`. The popswap is exactly where a
      mismatch would silently drift.
- [ ] **component add/remove migrates the id** — pop from one pool, re-add to another; id leaves the old list
      and lands at the tail of the new one.
- [ ] **empty pool cleanup** — remove a pool's last entity; pool AND its `_pool_ids` entry are gone.

`test/unit/test_queryresult.py` (+ an integration query across ≥2 archetypes):
- [ ] **`entity_ids` aligns with fields** — `zip(qr.entity_ids, qr.position)` pairs id↔row across pools.
- [ ] **flat-array ops work** — `qr.entity_ids[i]`, `np.isin(qr.entity_ids, [...])` (the ops `_Field` rejects).
- [ ] **shadowing guarded** — a component field named `entity_ids` (or a QueryResult internal) is rejected.

## Notes

- Order contract: `world.update()` freezes pool order/size within a tick (`world.py:37-41`), so the id↔row
  alignment (and thus `entity_ids`) is stable for the duration of a query, same lifetime as the `qr.field` views.

## Related

- `.tracker/todos/done/5-queryresult-flat-query/TASK.md` — the completed return-type work this builds on.
- `world.py:27-28` id maps; `world.py:95-114` `_add_to_pool` / `_pop_from_pool`; `pool.py:40-54` the popswap
  the sidecar must mirror; `world.py:78-89` `query_and`.
