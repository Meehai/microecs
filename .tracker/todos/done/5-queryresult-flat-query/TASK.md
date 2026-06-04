# QueryResult — archetype-transparent query (per-pool, zero-copy, numpy-like)

**Created**: 2026-06-03
**Updated**: 2026-06-04 (write-back is now positional `(N, *)` auto-split — see "Why" / "What landed")
**Priority**: 2
**Status**: ✅ Done (2026-06-04) — return type complete (88 tests green, lint 10/10). `entity_ids` + robosim
migration split out to task 6.

## Why

`query_and` returned `list[Pool]`; systems loop the pools by hand. We want a return
type that **feels like numpy across all matching entities** so a system writes
`qr.position[:] = qr.position + qr.velocity*dt` once instead of looping pools — and
without paying a copy on the elementwise hot path (Motion, WallBounce).

**The pivot.** The original motive was the cross-archetype collision N×N bug (a *global*
gather). We deliberately did **not** make the query implicitly gather. Pools are
separate allocations (`pool.py:24`), so a contiguous view across them is impossible,
and a hidden `np.concatenate` would tax every elementwise access for nothing. So:

- **default = per-pool, zero copy.** `qr.position` is a list of per-pool views, never merged.
- **cross-entity read = explicit.** Collision (the one op that truly needs every entity adjacent) reads
  via `qr.position.numpy()` (raw `np.concatenate` of the parts), computes all-pairs. The copy is opt-in
  and visible at the call site, never hidden in the query.
- **write = positional, like one `(N, *)` array.** `qr.color[:] = value` treats the field as one logical
  `(N, *e)` block: a `(N, *e)` value lands row i → entity i, split across pools **by order** (auto-split);
  anything broadcastable to `(N, *e)` (scalar, `(*e,)` row) fills every entity. Disambiguated purely by
  shape, exactly as numpy would against a real `(N, *e)` array.

So the cross-archetype collision now round-trips end to end: gather → all-pairs → `qr.color[:] =
np.where(mask, RED, BLACK)` writes back across **all** matched pools. The original bug is fully closed —
reads gather explicitly, writes auto-split positionally.

## What landed

`query_and(component_types)` → `QueryResult(pool_list, field_shapes, field_dtypes)` — two dicts
keyed by field name, the union of the queried components' fields/shapes/dtypes (from
`component_to_shapes`/`component_to_dtypes`, `world.py:21-25,85`). Shapes/dtypes (not just names) so an
empty query can synthesize a real `(0, *e)` block.

`QueryResult`:
- `__getattr__(field)` → `_Field([<pool.field view> for each pool])` — per-pool zero-copy views, **no concat**.
  When no pool matched, synthesizes a single `(0, *e)` empty array so the field is still a real block.
  Field-scoped: only queried fields are keys; `qr.mass` (not queried) raises `AttributeError`.
- `__len__` → total entities across pools.

`_Field(np.lib.mixins.NDArrayOperatorsMixin)` — holds `parts`, one view per pool:
- `__array_ufunc__` → `+ - * / > sqrt …` dispatch per pool, return a new `_Field`. Guards
  `method != "__call__"` → `NotImplemented` (so `reduce`/`accumulate` can't sneak a
  cross-pool reduction through).
- `__array_function__` → whitelist `{np.where, np.clip, …}`, dispatch per pool;
  anything not whitelisted → `NotImplemented` → numpy raises `TypeError`. The whitelist is
  what keeps reductions loud (see Boundaries).
- `__setitem__` → `qr.field[:] = value` (and `qr.field[:, k] = …`). A `_Field` value aligns per-pool; a
  plain ndarray is treated as one logical `(N, *e)` array — `(N, *e)` splits positionally across pools,
  broadcastable shapes fill all. A masked / entity-index *key* still raises (those reads use `numpy()`).

### Write idiom — now **matches** Pool (the old plan flipped it; reverted)

```python
qr.position[:] = qr.position + qr.velocity * dt                     # elementwise, zero gather
qr.velocity[:] = np.where(qr.velocity > T, -qr.velocity, qr.velocity)   # masked, via __array_function__
```

`qr.position = ...` is **not** the write path — the write goes through the field's `[:]=`,
exactly like `pool.x[:] = ...` (`pool.py:68-71`). Same muscle memory as Pool.

### Boundaries — deliberately loud, never silent

Whole-field `[:]` / `[:, k]` writes feel like an `(N, *)` array (positional or broadcast). But these
reads / partial ops need a gather (a copy), so they stay explicit and **raise** rather than guess:
- cross-pool reduction: `np.sum(qr.velocity)` → use `np.sum(qr.velocity.numpy())`
- entity-index read: `qr.velocity[0]` → use `qr.velocity.numpy()[0]`
- masked / partial write: `qr.velocity[mask] = 0` (only `[:]` / `[:, k]` keys allowed; future work)

Cross-entity reads gather explicitly: `qr.position.numpy()`. A global result writes straight back
positionally with `qr.field[:] = …` (auto-split across pools) — no by-hand split, no scatter.

## Done when

Done:
- [x] `query_and` returns `QueryResult`; `len(qr)` == total matching entities across archetypes.
- [x] elementwise math + whole-field write-through, per-pool **zero copy** (`__array_ufunc__`).
- [x] in-place ops (`qr.velocity *= 0.5`) write through — `__array_ufunc__` honors `out=(self,)`.
- [x] `np.where` & friends dispatch per-pool (`__array_function__` whitelist).
- [x] accessing a non-queried field raises.
- [x] reductions / indexed reads / masked writes raise (loud boundaries).
- [x] component-axis read+write `qr.x[:, k]` (view-preserving); entity-axis (`[0]`, `[0:1]`, mask) rejected.
- [x] per-entity iteration `for row in qr.field` (and `zip(qr.position, qr.radius)`) for the render loop.
- [x] `len(qr.field)` == total entities (== `len(qr)`) — systems size masks / `range()` off it.
- [x] **`numpy()`** — `_Field.numpy()` = raw `np.concatenate(self.parts)` (`query_result.py:12`). The
      explicit cross-entity read; feeds the all-pairs collision.
- [x] **Positional `(N, *)` write-back** — `qr.field[:] = value` (plain ndarray) treats the field as one
      logical `(N, *e)` block: `np.broadcast_to(value, (N, *e))` then split positionally across pools
      (`query_result.py` `__setitem__`). `(N, *e)` lands row→entity, broadcastable shapes fill all, anything
      else raises like numpy. Closes the cross-archetype collision end-to-end (detect via numpy(), write back positional).
- [x] **Migrated `test_world.py`** to the new API (`.pool_list[0]`, `len(qr)` == entities) — 45 tests green.
- [x] **Migrated examples 01 & 02** to the `qr.field` API (elementwise, column indexing, `numpy()` + positional collision).
- [x] **Empty results behave like `np.empty((0, *e))`** — both a zero-entity pool and a no-matched-pool query
      are a real `(0, *e)` block (shape/dtype threaded via the new constructor + `__getattr__` synthesis):
      `numpy()` → `(0, *e)`, broadcast write no-ops, wrong shape raises. Pinned by
      `test_empty_field_writes_behave_like_an_empty_numpy_block` + `test_empty_query_with_no_matched_pool_behaves_like_numpy`.

Done:
- [x] **Split a full-N raw operand per pool** (`_Field._chunk`). `_apply_fn_on_parts` AND the `__array_ufunc__`
      `out=` loop now split any operand whose `shape[0] == N` into per-pool chunks like a `_Field`; scalars /
      `(*e,)` rows still pass whole. So `np.where(mask_(N,*e), …)` and `qr.field += raw_(N,*e)` both work across
      any number of pools. Known corner (documented): when a `(*e,)` row has `e[0] == N` the `shape[0]==N` rule
      splits it instead of broadcasting — the same ambiguity `__setitem__` dodges via `np.broadcast_to`.

Follow-ups (moved to **task 6**, not part of this branch):
- `entity_ids` accessor — reserved name; `World` threads ids at query time from
  `_pool_ix_to_eid` (`world.py:28`), aligned to pool+index order (pools have no id concept).
- migrate robosim `server_ecs.py` (heavier: relies on iterate-pools / `[0]` / `len`=pool-count; needs `entity_ids`).

## Tests — `test/unit/test_queryresult.py`

Passing:
- `test_len_sums_entities_across_pools`
- `test_field_len_is_total_entities_across_pools` — `len(qr.position)` == `len(qr)` == iteration count.
- `test_field_write_scatters_per_pool_no_gather` — `qr.position[:] = qr.position + 1`, per pool, no gather.
- `test_two_field_motion_writes_through_per_pool` — `pos[:] = pos + vel*dt`, two pools, *different* velocities.
- `test_np_where_dispatches_per_pool` — `np.where` runs per pool and writes back (`__array_function__`).
- `test_unsupported_ops_are_rejected` — `np.sum` / `qr.x[0]` / `qr.x[mask]=` all raise.
- `test_inplace_op_writes_through_to_pools` — `qr.velocity *= 0.5` mutates the pools (`out` honored).
- `test_iterating_fields_yields_each_entity_for_rendering` — `zip(qr.position, qr.radius)` render loop.
- `test_component_axis_index_reads_a_per_pool_field` / `..._writes_through_one_column` — `qr.x[:, k]`.
- `test_entity_axis_index_stays_rejected` — `qr.x[0]` / `[0:1]` / `[mask]` raise.
- `test_wallbounce_per_axis_via_column_indexing` — column read + `np.where` + column write, per pool.
- `test_gather_concatenates_all_entities_for_cross_entity_ops` — gather order/shape.
- `test_collision_round_trips_via_gather_single_archetype` — gather → all-pairs → `qr.color[:] = np.where(...)`.
- `test_broadcast_row_or_scalar_writes_to_every_entity` — a `(*e,)` row / scalar fills every entity; pins
  broadcast is by shape against `(N, *e)` (N == row length on purpose, so a naive `shape[0]==N` split is caught).
- `test_contiguous_array_writes_back_positionally_across_pools` — `(N, 2)` block writes row i → entity i, split
  by order; reversing the block reverses every entity.
- `test_cross_archetype_collision_round_trips_via_positional_writeback` — numpy() detect + `(N, 4)` positional
  write-back across two archetypes; both overlapping balls end red.
- `test_empty_field_writes_behave_like_an_empty_numpy_block` — a `(0, 2)` field matches `np.empty((0, 2))`:
  no-op broadcast write, `ValueError` on wrong shape, `numpy()` → `(0, 2)`.
- `test_empty_query_with_no_matched_pool_behaves_like_numpy` — no pools matched is still a `(0, 2)` block
  (shape/dtype threaded): `numpy()` → `(0, 2)`, broadcast write no-ops, wrong shape raises.

Full-N raw operand split across pools:
- `test_raw_full_n_mask_mixed_with_field_splits_across_pools` — `qr.velocity[:] = np.where(raw_(N,2)_mask,
  -qr.velocity, qr.velocity)` over two pools; the raw mask splits per pool like a `_Field`.
- `test_inplace_op_with_full_n_raw_operand_splits_across_pools` — `qr.velocity += raw_(N,2)` over two pools;
  the in-place `out=` path splits the same way.
- `test/integration/test_i_multi_archetype_mask.py::test_full_n_mask_writes_back_across_two_archetypes` — the
  example-02 WallBounce idiom through a real World query matching TWO archetypes.

To write later: `entity_ids` alignment.

## Out of scope

- Spatial hashing / broad-phase. N×N stays O(N²).
- **Implicit** cross-archetype gather on READ — `qr.field` stays per-pool views; reductions / entity-index
  reads gather explicitly via `qr.field.numpy()`. (Writes auto-split positionally — that's a scatter, not a gather.)
- `query_or` / negation queries. This is about the return *type*.

## Related

- `world.py:78-86` `query_and`; `world.py:37-41` deferred `update()` (freezes pool order/size
  within a tick → per-pool view ordering is stable).
- `pool.py:68-71` `Pool.__setattr__`/`__getattr__` — QueryResult now **mirrors** the `x[:] =`
  write idiom (no longer the inverse).
- `examples/02` `CollisionDetectionSystem` — the global system; numpy() to detect, positional `qr.color[:] = …`
  to write back. Closes the cross-archetype miss once positional write-back lands.
