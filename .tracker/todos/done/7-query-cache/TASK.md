# Cache `world.query_and` results between updates

**Created**: 2026-06-04
**Updated**: 2026-06-04 (done)
**Priority**: 2
**Status**: ✅ Done (2026-06-04) — `_cache: dict[PoolKey, QueryResult]` on `World`; `query_and` returns the
cached result on hit; `update()` clears the cache iff the command buffer was non-empty. 112 tests passing.

## Why

`query_and` rebuilds everything on every call (`world.py:78-90`): it re-scans the pools for matches, rebuilds
the `field_shapes`/`field_dtypes` dicts, re-concatenates `entity_ids` (`np.array(sum(...))`), and the
`QueryResult.__init__` rebuilds the per-pool view lists. In a hot loop several systems query the same archetype
set every tick, so that's per-tick allocation for a result that hasn't changed.

It's safe to cache because of an invariant the engine already guarantees: **all structural mutation is lazy and
funnels through `world.update()`** (`add/remove_entity`, `add/remove_component` only append to
`_command_buffer`). So between two `update()` calls the pools are frozen — no entity moves, no id changes, and
crucially **no array reallocates**. Systems only write field *values* in place through the views. A
`QueryResult` built this tick stays 100% valid until the next `update()`.

## Design

- Add `self._query_cache: dict[PoolKey, QueryResult] = {}` (keyed by `_make_key(component_types)`, the same int
  the pool lookup uses).
- `query_and`: if `key in self._query_cache`, return it; else build as today, store, return.
- `update`: clear the **whole** cache **iff** the command buffer is non-empty, i.e. iff this `update()` actually
  applies structural changes:

  ```python
  def update(self):
      if self._command_buffer:        # structural changes about to happen
          self._query_cache.clear()
      for fn in self._command_buffer:
          fn()
      self._command_buffer.clear()
  ```

### Two things to get right

1. **Invalidate on `len(self._command_buffer) > 0`, NOT `len(self) > 0`.** Live-entity count is the wrong signal
   — a world with 1000 entities and an empty buffer changed nothing this tick, so the cache should survive
   (that's the whole win). What matters is "did this `update()` apply any commands."
2. **Clear the whole cache, don't selectively invalidate.** One structural change (new archetype, deleted empty
   pool, entity migrating pools) can invalidate several cached queries at once; figuring out exactly which is
   more code and more bugs than dropping the dict. The rebuild is cheap. GRUG.

### The subtlety not to miss

A cached `QueryResult` holds *views* `p.data[f][0:len(p)]` (`query_result.py:95`). When a pool grows/shrinks,
`Pool._realloc` (`pool.py:56-61`) swaps in a **new** numpy array, so any cached view would point at the old,
freed buffer. But realloc only fires inside `add_entity`/`remove_entity` — i.e. during `update()` with a
non-empty buffer — so the rule above already covers it. No extra handling needed; the stale-view test below
guards it.

## Done when ✅

- [x] Repeated `query_and(same components)` with no intervening mutating `update()` returns the cached result.
- [x] A no-op `update()` (empty buffer) keeps the cache; a mutating `update()` drops it and the next query
      reflects the change.
- [x] No stale views: a query taken before an `update()` that grows/shrinks a pool is never reused.
- [x] `_cache` is the only added structure; no behaviour change to query results themselves.

## Tests (tester owns these — `test/unit/test_world.py`) ✅

- [x] `test_query_cache_returns_same_object_between_updates` — two `query_and` calls with the same components,
      no mutating update between, return the *same* `QueryResult` (identity).
- [x] `test_noop_update_keeps_cache` — `update()` with an empty buffer leaves the cached object in place.
- [x] `test_mutating_update_invalidates_cache` — after `add_entity` + `update()`, a re-query is a fresh object
      whose `entity_ids` / `len` reflect the change.
- [x] `test_cache_keyed_per_query` — different component sets get independent cache entries (no cross-talk).
- [x] `test_new_archetype_appears_after_invalidation` — first entity of a new component combo shows up in a
      re-query after the committing `update()` (the new pool isn't masked by a stale cache).
- [x] `test_no_stale_views_across_realloc` — query a pool, then add past `INITIAL_CAPACITY=100` to force
      `_realloc`, `update()`, re-query; writes through the fresh `QueryResult` land in the live pool and `len`
      is 101 (proves the pre-realloc views aren't reused).

## Implementation note

`update()` runs the buffer first, then `if len(self._command_buffer) > 0: clear buffer + clear cache` — so a
no-op tick keeps the cache, a mutating tick drops it. The `self._cache` type hint reads `dict[PoolKey, Pool]`
but it stores `QueryResult` (cosmetic; dev's call, non-test file).

## Notes

- Same lifetime contract as `entity_ids` (task 6): pool order/size is frozen within a tick by the lazy buffer,
  so a cached query is valid exactly as long as the views/ids it holds.
- The cache is correctness-transparent: it only ever returns a result equal to what a fresh build would produce
  this tick. If a test ever sees otherwise, the invalidation rule is wrong, not the test.

## Related

- `world.py:32` `_command_buffer`; `world.py:37-41` `update`; `world.py:78-90` `query_and`.
- `pool.py:56-61` `_realloc` (the stale-view source); `query_result.py:88-96` `QueryResult.__init__` (holds the
  views being cached).
- `.tracker/todos/done/6-queryresult-entity-ids/TASK.md` — prior work; the lazy-update invariant this relies on.
