# `World.get_entity(entity_id)` — read one entity's data by id

**Created**: 2026-06-03
**Priority**: 3

## Why

Today the id is a *write* handle: `add_component` / `remove_component` / `remove_entity`
all take an `eid`, but there is no way to **read** a single entity back by id. The only
read path is `query_and(...)` → SoA pools, where the caller must know the entity's row
index. After a swap-remove that index is stale, so "give me entity N's current fields"
is currently impossible without reaching into `world._eid_to_pool_ix` directly.

World already keeps the mapping (`_eid_to_pool_ix: eid -> (pool, index)`). `get_entity`
just makes that lookup a clean public method, completing the id-based API from
[task 2](../../done/2-id-based-world-and-trait-migration/TASK.md).

## API

```python
fields = world.get_entity(eid)   # -> dict[str, np.ndarray], one entry per field of the entity's archetype
```

- Returns the entity's field data keyed by field name (e.g. `{"position": [...], "velocity": [...]}`).
- Lookup is O(1) via `_eid_to_pool_ix`; no pool scan.

## Open design calls (developer decides)

1. **Copy vs view.** Return copies (safe, but caller can't mutate in place) or views into
   the SoA arrays (mutable like `pool.position[:] = ...`, but go stale on the next
   swap-remove)? Lean toward **copy** for a read API; document it either way.
2. **Uncommitted spawns.** An id minted this tick but not yet committed lives in
   `_live_ids` + the command buffer, not in `_eid_to_pool_ix`. Should `get_entity`
   resolve it (forcing a flush / reading the pending command) or reject it until
   `update()`? Simplest: only committed entities resolve; not-yet-materialized id → clear
   error. Mirror whatever `add_component`-on-pending-spawn already does.
3. **Unknown / removed id.** Must fail with a clear `AssertionError`, not a bare `KeyError`.

## Done when

- `get_entity(eid)` returns the live field values for a committed entity, keyed by field name.
- The returned data matches what the entity's pool holds at its current row, **after** a
  swap-remove has moved that row (the whole point — id stays valid, index does not).
- An unknown / already-removed id raises a clear error.
- Multi-field and multi-component archetypes return every field.

## Tests (tester writes, under `test/unit/test_world.py`)

- `test_get_entity_returns_all_fields` — multi-component entity, assert every field present and correct.
- `test_get_entity_after_sibling_swap_remove` — remove a sibling so the target row is swap-moved,
  assert `get_entity` still returns the target's data (not its old neighbour's).
- `test_get_entity_unknown_id_raises` — id never handed out → clear error.
- `test_get_entity_removed_id_raises` — removed (committed) id → clear error.
- (if copy semantics chosen) `test_get_entity_returns_independent_copy` — mutating the result
  does not touch pool storage.

## Out of scope

- Bulk/batched reads (`get_entities([id, ...])`). One id per call for now.
- Querying *which* components an entity has by id (return just field data; archetype
  introspection is a separate ask).
- A matching `set_entity(eid, **fields)` writer — reads first; add a writer only if a caller needs it.

## Related

- `microecs/world.py`: `_eid_to_pool_ix`, `_pop_from_pool` (swap-remove bookkeeping), `query_and`.
- `microecs/pool.py`: `pop_entity` already builds a per-field dict copy — same shape `get_entity` returns.
- Completes the id-based handle API from
  [task 2](../../done/2-id-based-world-and-trait-migration/TASK.md).
