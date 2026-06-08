# Sub-query — `QueryResult.query(...)` (query a query, not the world)

**Created**: 2026-06-08
**Updated**: 2026-06-08
**Priority**: 2
**Status**: 🟢 Open — design locked, ready to build. **Decided: Opt 1 (pass keys + bound resolver) + filter
semantics.** `qr.query(...)` is a sub-query: same fields, fewer pools.

## Goal

Let a `QueryResult` be re-queried to carve out a subset, reusing the parent's pools:

```python
aabb_entities = world.query(HasModel, HasPose, HasCollision, HasAABBCollider)
aabb_frozen   = aabb_entities.query(Frozen)          # subset: pools that ALSO have Frozen
aabb_movable  = aabb_entities.query(exclude=Frozen)  # subset: pools that do NOT have Frozen
```

The dev's hunch: *the underlying pools can be re-used to build a new QueryResult.* That is correct — a
sub-query is just "pick a subset of the parent's `pool_list` by one more bitmask predicate." The data
arrays are never copied; only the list of pools shrinks.

## The one design decision: filter, not re-project

Two possible meanings of `aabb_entities.query(Frozen)`:

- **(A) Filter / refine (recommended).** Keep the parent's fields (`HasModel, HasPose, …`), just drop the
  pools that lack `Frozen`. `Frozen` is a *predicate*, not a projection. This is the only meaning that
  makes `query(Frozen)` and `query(exclude=Frozen)` a clean partition (frozen vs movable) of the same
  field set. Matches how `exclude=` already behaves — it filters, it never adds fields.
- **(B) Re-project.** Result exposes only `Frozen`'s fields. Inconsistent with `exclude=`, and useless for
  the stated movement/collision use case. Reject.

Go with **(A)**: sub-query keeps the parent's field shapes/dtypes; it only narrows `pool_list`.

## Why QueryResult can't do this today

`QueryResult` (`query_result.py:97`) stores `pool_list`, a **flat** `entity_ids`, `_field_shapes`,
`_field_dtypes`, and per-pool `_data` parts. To filter itself it is missing two things:

1. **Per-pool archetype keys.** The filter is one line (`world.py:120`):
   `(key & include_key) == include_key and (key & exclude_key) == 0`. But a `Pool` does **not** store its
   own key — the world holds `pools: dict[PoolKey, Pool]`. QueryResult sees the pools, not their keys.
2. **A component → bit resolver.** Turning `Frozen` into a key needs `world._make_key` (`world.py:195`),
   which owns the component→bit registry. QueryResult has no link back to the world.

Also: `entity_ids` is flattened (`np.array(sum(...))`), so the subset's ids can't be sliced out per pool.
It needs per-pool id boundaries (the `_Field._bounds` trick already in the file) or per-pool id lists.

## Design — Opt 1 (locked)

`world.query` passes two new bits into the QueryResult constructor, **both optional / defaulting to None**:

- `pool_keys: list[PoolKey] | None` — the archetype key of each pool, parallel to `pool_list`.
- `make_key: Callable | None` — a reference to `world._make_key` (component list → bitmask).

`QueryResult.query(*include, exclude=None)` then:
1. If `pool_keys`/`make_key` are None → raise a clear error (this QueryResult wasn't built by a World, so it
   can't be sub-queried). See "constructor stays additive" below for why this case exists.
2. `inc = make_key(include); exc = make_key(exclude or [])`.
3. Keep pools where `(key & inc) == inc and (key & exc) == 0` — the **exact** `world.py:120` predicate, over
   the parent's `pool_list` instead of all pools.
4. Build a new QueryResult from the surviving pools, **carrying the parent's `_field_shapes`/`_field_dtypes`
   unchanged** (filter semantics), the surviving pools' keys, the same `make_key`, and the matching slice of
   `entity_ids`.

Near-copy of `World.query`'s loop over a shorter list. No public `world.query` signature change.

**Why not the others:** a `world` ref on QueryResult (Opt 2) couples it to World and invites calling
mutators through a result; pools learning their own key (Opt 3) still wouldn't give the component→bit map
the *sub-query's* include/exclude needs. Opt 1 is the least new state.

## Constructor stays additive (matters for tests)

`test/unit/test_queryresult.py` builds `QueryResult` **directly, with no World** (the `_query` helper,
`test_queryresult.py:82`, 4 positional args). So the two new params **must be optional with a None default**
— append them, don't reorder. Then:

- Every existing direct-construction test keeps passing untouched.
- A QueryResult built by hand (no keys/resolver) raises a clear error if `.query()` is called on it.
- Only World-built QueryResults can be sub-queried — which is exactly the real use case.

Consequence for the tester: the **behavioural** sub-query tests (partition, narrowing, exclude) live at the
**World level** (`test/unit/test_world.py`), because they need real keys + resolver. A couple of
**guard** tests (sub-querying a hand-built QueryResult raises) can sit in `test_queryresult.py`.

## Open questions

- **Caching.** `World.query` caches on `(include_key, exclude_key)`. Should sub-queries cache too? On what
  key — `(parent_pools, include, exclude)`? Simplest first pass: no cache, just rebuild (pools are reused,
  so it's cheap). Decide before shipping.
- **Cache invalidation.** Parent QueryResults already go stale after `world.update()` (pools mutate). A
  sub-query built off a stale parent is doubly stale. Same rule as today (don't hold results across
  `update()`), but call it out in the docstring.
- **`include` + `exclude` together** on a sub-query — should compose like `World.query` (A & ~B). Free if
  we reuse the same predicate.
- **Is it worth it?** vs. just calling `world.query(HasModel, HasPose, HasCollision, HasAABBCollider,
  Frozen)` directly. The win is (a) not re-scanning all archetypes, (b) expressing "split this set two
  ways" without repeating the long include list. If those don't matter, this is sugar — decide with the dev.

## Acceptance / test checklist (tester writes once design lands)

**World level** (`test/unit/test_world.py`) — real keys + resolver:

- [ ] `sub_query_keeps_parent_fields` — `aabb.query(Frozen)` exposes parent's fields, not Frozen's.
- [ ] `sub_query_include_narrows_pools` — only pools that also have the component survive.
- [ ] `sub_query_exclude_drops_pools` — `query(exclude=Frozen)` drops Frozen-bearing pools.
- [ ] `partition_is_disjoint_and_covers` — `query(Frozen)` + `query(exclude=Frozen)` partition the parent
      (no overlap, union == parent), entity-id-wise.
- [ ] `sub_query_entity_ids_subset` — `.entity_ids` of the subset ⊆ parent's, correct per pool.
- [ ] `sub_query_data_is_view_not_copy` — writes through the sub-query's fields hit the same pool arrays.
- [ ] `sub_query_empty_parent` — sub-querying an empty result is a no-op empty result (no crash).
- [ ] `sub_query_no_match` — include a component no parent pool has → empty result.
- [ ] `sub_query_chains` — `a.query(X).query(Y)` works (X then Y).
- [ ] `sub_query_unregistered_component_raises` — same guard as `World.query`.
- [ ] `sub_query_after_update_semantics` — document/verify stale-parent behavior.

**QueryResult level** (`test/unit/test_queryresult.py`) — hand-built, no World:

- [ ] `hand_built_query_result_has_no_subquery` — `.query()` on a QueryResult built without keys/resolver
      raises a clear error (not an AttributeError soup).
- [ ] existing direct-construction tests still green — proves the two new ctor params are additive/optional.

## Notes

Pure additive change to `query_result.py` + a small tweak to what `World.query` passes the constructor.
No change to the public `world.query` signature. Grug-cheap if Opt 1 — the hard part (bitmask filter) is
already written in `world.py:120`; this just runs it over a shorter list.
