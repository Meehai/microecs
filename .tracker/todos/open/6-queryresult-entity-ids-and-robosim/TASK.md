# QueryResult `entity_ids` + robosim migration

**Created**: 2026-06-04
**Priority**: 3
**Status**: Open

## Why

Split out of task 5 (QueryResult return type, ✅ done). The return type is complete; these two are the
remaining consumers that need an entity-id story before robosim can move off the old iterate-pools API.

## Scope

- [ ] **`entity_ids` accessor** on `QueryResult` — reserved field name. `World` threads ids at query time
      from `_pool_ix_to_eid` (`world.py:28`), aligned to pool+index order (pools have no id concept). Returns
      ids in the same pool-by-pool order as `qr.field` parts, so `zip(qr.entity_ids, qr.position)` lines up.
- [ ] **Migrate robosim `server_ecs.py`** to the new API — heavier than examples 01/02: relies on
      iterate-pools, `[0]`, and `len` == pool-count. Needs `entity_ids` first.

## Notes

- Order contract: `world.update()` freezes pool order/size within a tick (`world.py:37-41`), so the id↔row
  alignment is stable for the duration of a query.

## Related

- `.tracker/todos/done/5-queryresult-flat-query/TASK.md` — the completed return-type work this builds on.
- `world.py:28` `_pool_ix_to_eid`; `world.py:78-89` `query_and`.
