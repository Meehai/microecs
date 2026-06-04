# QueryResult `entity_ids` (+ id-map refactor)

**Created**: 2026-06-04
**Updated**: 2026-06-04 (done; robosim integration moved out — that's robosim's task, not this library's)
**Priority**: 2
**Status**: ✅ Done (2026-06-04) — `_pool_ids` refactor + `entity_ids` accessor + reserved-name guard, all tested
(104 passing). Branch `entity-ids-in-queryresult`.

## Why

Split out of task 5 (QueryResult return type, ✅ done). The return type was complete; this added the entity-id
story: a stable id per matched entity, exposed on the query so callers can map a row back to `world.get_entity`
/ `world.remove_entity`.

## Step 1 — `_pool_ix_to_eid` → `_pool_ids` sidecar ✅

Replaced the tuple-keyed reverse map with a per-pool list that mirrors each pool's own dynamic array:
`self._pool_ids: dict[Pool, list[EntityId]]` (`_pool_ids[pool][ix] == id at pool index ix`). `_eid_to_pool_ix`
(id → location) stays for O(1) `get_entity` / `remove`. Simplification: no `(Pool, int)` tuple keys/hashing, and
the popswap just mirrors the pool's swap.

- [x] declare `_pool_ids`; init on pool creation (`setdefault`); append on `_add_to_pool`.
- [x] `_pop_from_pool` mirrors the pool popswap (`ids[-1]` → `ids[pool_ix]`, then drop tail) and drops the list
      on empty-pool cleanup.
- [x] no behaviour change to `get_entity` / `add_component` / `remove_component` / eager-lazy buffer.

## Step 2 — `.entity_ids` accessor on `QueryResult` ✅

- [x] `query_and` threads ids from `_pool_ids`, pool-by-pool aligned with the `qr.field` parts:
      `np.array(sum((self._pool_ids[p] for p in res), []), dtype="int64")` (`[]` → `(0,)`).
- [x] flat `(N,)` array, NOT a `_Field` — ids aren't in pools, are read-only, and support `[i]` / slicing /
      `np.isin` (the entity-axis ops `_Field` rejects), so `zip(qr.entity_ids, qr.position)` lines up.
- [x] **reserved-name guard** — `World._check_components` rejects any component field whose name collides with a
      `QueryResult` attribute (derived by reflection: `data`, `entity_ids`, `field_dtypes`, `field_shapes`,
      `fields`, `len`, `pool_list`), so a field can't be silently shadowed by `qr.<name>`.

## Done when ✅

- [x] `_pool_ix_to_eid` gone; `_pool_ids` is the single reverse structure; stays index-aligned through add /
      remove / swap / component add+remove; empty-pool removal drops the list entry. All tests green.
- [x] `qr.entity_ids` is a flat `(N,)` array aligned to `qr.field`; shadowing names are rejected.

## Tests (all green; tester owns these)

`test/unit/test_world.py`:
- [x] `test_pool_ids_stay_aligned_through_random_churn` — 500 seeded add/remove/component ops; after every commit
      the reverse map mirrors the pools and each id's data round-trips through `get_entity`.
- [x] deterministic popswap cases (last-index, middle-swap, reclaim+no-dangling, id-resolve-after-swap, migration).
- [x] `entity_ids` through `World.query_and` — flat & aligned across pools, flat-array ops, empty query → `(0,)`,
      survives swap-remove.
- [x] `test_world_rejects_component_field_named_like_a_queryresult_attribute[*]` — parametrized over all reserved
      names (reflection-derived, so it stays complete).

`test/unit/test_queryresult.py`:
- [x] `entity_ids` length tracks `len(qr)`; flat pool-by-pool array; render loop `zip(qr.entity_ids, ...)`; repr.

## Out of scope

- **robosim migration** — robosim consumes this library; integrating the new `entity_ids` API into
  `server_ecs.py` is robosim's own task, tracked there, not here.

## Notes

- Order contract: `world.update()` freezes pool order/size within a tick (`world.py:37-41`), so the id↔row
  alignment (and thus `entity_ids`) is stable for the duration of a query, same lifetime as the `qr.field` views.

## Related

- `.tracker/todos/done/5-queryresult-flat-query/TASK.md` — the return-type work this builds on.
- `world.py:27-28` id maps; `world.py:95-116` `_add_to_pool` / `_pop_from_pool`; `world.py:163-175`
  reserved-name guard; `world.py:78-90` `query_and`; `query_result.py:88-96` `QueryResult.__init__`.
