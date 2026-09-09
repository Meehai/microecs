# Single-entity read — `qr.field[i]` by integer index

**Created**: 2026-06-04
**Priority**: 3
**Status**: ✅ Done (2026-06-04) — `range(self.len)[key]` + `searchsorted`, numpy-exact, 120 tests green.

## What landed

`_Field.__getitem__` (`query_result.py:73-77`) handles a bare integer key:

```python
if isinstance(key, (int, np.integer)):
    key = range(self.len)[key]                                   # numpy-exact: negative wrap + IndexError on OOB
    pool_ix = int(np.searchsorted(self._bounds, key, side="right")) - 1
    return self.parts[pool_ix][key - self._bounds[pool_ix]]
```

- **numpy-faithful at every boundary** (checked against `np.concatenate(parts)[i]`): negative-in-range wraps,
  both-sign out-of-range raises `IndexError`, empty query raises `IndexError`. The index rule is delegated to
  Python's `range`, not re-derived by hand — which is what killed the earlier `% len` wrap / `% 0`
  ZeroDivision bugs.
- Returns a **view** into the entity's pool, so single-entity write-through (`qr.pose[i][...] = v`) works with
  no `__setitem__` change.
- O(log P) pool lookup via the existing `_bounds`; **no extra state** — the considered O(N) `ix_to_pool_ix`
  map was dropped.
- `qr.field[:, k]` still returns a `_Field`; entity slices/masks still raise `TypeError` — unchanged.

All "Done when" criteria met. 120 tests green:
- `test/unit/test_queryresult.py`: `test_single_entity_read_routes_index_to_right_pool`,
  `..._returns_a_writeable_view`, `..._positive_out_of_range_raises`,
  `..._negative_out_of_range_raises_like_numpy`, `..._on_empty_query_raises`; rejection narrowed to
  slices/masks in `test_entity_axis_index_stays_rejected` / `test_unsupported_ops_are_rejected`.
- `test/integration/test_i_fpv_camera_read_loop.py`: robosim's FPV loop via iteration + indexed read across
  two archetypes.

## Why

robosim drives object components one entity at a time and wants the natural spelling:

```python
qr = world.query_and([HasPose, HasFPV])
for i in range(len(qr)):
    cam, pose = qr.fpv_camera[i].item(), qr.pose[i]
    cam.set_position(position=pose[0:3, 3], up=pose[0:3, 1], target=pose[0:3, 3] + pose[0:3, 2])
```

Today `qr.pose[i]` / `qr.fpv_camera[i]` raise `TypeError` — `_Field.__getitem__` rejects **every**
entity-axis index (`query_result.py:74-77`). This is **not** a read-vs-write thing: a `_Field` is a
stitched view over many pools, so an entity-axis index has to resolve to a single pool — the guard
refuses that for reads exactly as it does for writes.

But a **single int** is the safe special case the dev's instinct flagged: one entity lives in exactly
one pool, so `i` maps cleanly to `(pool, local_index)` and `part[local]` is a plain numpy **view** — no
cross-pool stitching, no copy. The genuinely hard cases (entity *slices*, *masks*) stay rejected.

**Not blocking.** Field iteration already covers robosim today (verified —
`test/integration/test_i_fpv_camera_read_loop.py`):

```python
for cam, pose in zip(qr.fpv_camera, qr.pose):
    cam.item().set_position(...)        # works now
```

So this is ergonomic sugar (matches numpy/robosim intuition, gives random access), not a missing
capability. Hence P3.

## Design

`_Field` already carries everything needed: `self._lens` (per-pool lengths, `query_result.py:13`) and
`self._bounds = np.cumsum([0, *self._lens])` (`query_result.py:16`). Extend `__getitem__`
(`query_result.py:74-77`) with one branch for a bare integer key:

```python
def __getitem__(self, key):
    if isinstance(key, (int, np.integer)):        # single entity -> lands in exactly ONE pool
        i = key + self.len if key < 0 else key    # numpy-parity negative index
        if not 0 <= i < self.len:
            raise IndexError(f"entity index {key} out of range for {self.len} entities")
        p = int(np.searchsorted(self._bounds, i, side="right")) - 1
        return self.parts[p][i - self._bounds[p]]  # a VIEW into that pool's array (NOT a _Field)
    if not (isinstance(key, tuple) and key and key[0] == slice(None)):
        raise TypeError("entity-axis indexing crosses pools; use [:, k]. Use .numpy() to get a proper np.ndarray.")
    return _Field([part[key] for part in self.parts])
```

(Resolution verified in `test/manual/fpv-camera-readindex/` — searchsorted + bounds route 0..N-1 and
negative indices to the right pool/offset.)

### The one asymmetry to keep clear

- `qr.pose[i]` (bare int) → **plain `np.ndarray`** (one entity, collapsed to one pool).
- `qr.pose[:, k]` (component axis) → still a **`_Field`** (every entity, per-pool views).

That difference is correct: a single entity has nothing to stitch across pools, so it leaves the
`_Field` abstraction by design. The returned array is a **view**, so `qr.pose[i][0:3, 3] = ...` writes
through to the pool for free — single-entity *mutation* falls out of the read, with no `__setitem__`
change.

## Done when

- `qr.field[i]` (bare int, read) returns entity `i`'s value as a plain `np.ndarray` **view**, from
  whichever pool it lives in — across a query spanning >1 pool.
- Value routing is correct: with N entities split 2+3 across two pools, `qr.field[i]` for `i in 0..4`
  returns the i-th entity (pool-by-pool order), matching `qr.field.numpy()[i]`.
- Negative indices follow numpy (`qr.field[-1]` is the last entity).
- Out-of-range raises **`IndexError`** (numpy-parity), not `TypeError`; an empty query (`(0, …)`
  block) raises `IndexError` for any index, like `np.empty((0, …))[0]`.
- Object component: `qr.fpv_camera[i]` returns a 0-d object array; `.item()` is the stored object
  (matches robosim).
- The returned view writes through: mutating `qr.field[i]` mutates the pool.
- `qr.field[:, k]` still returns a `_Field` (unchanged); entity **slices** (`qr.field[2:5]`) and
  **masks** still raise `TypeError` (unchanged).

## Tests (tester writes)

Already in place:
- `test/integration/test_i_fpv_camera_read_loop.py::test_fpv_camera_indexed_read_matches_robosim_spelling`
  — `xfail(strict)` on robosim's literal spelling; **flips to xpass when this lands → remove the marker.**
- `test/unit/test_queryresult.py::test_single_entity_read_routes_index_to_right_pool` — `xfail(strict)`;
  2+3 entities across two pools, position encodes the global index, `qr.position[i] == [i, i]` for
  `i in 0..4` (+ negative index). **Same flip-and-unmark.**

To add when implementing:
- `test_single_entity_read_returns_a_writeable_view` — `qr.position[i][:] = v` writes through to the
  pool (proves it's a view, not a copy).
- `test_single_entity_read_out_of_bounds_raises_indexerror` — OOB index and empty query both raise
  `IndexError` (numpy-parity), NOT `TypeError`.
- `test_single_entity_read_object_component_item_roundtrips` — object dtype, `.item() is` the stored
  object.

To **update** when implementing (they pin today's blanket rejection of a bare int, which this changes):
- `test_queryresult.py::test_unsupported_ops_are_rejected` (`qr.velocity[0]` at lines 186-187) — drop
  the int case (it no longer raises); keep the reduction + masked-write cases.
- `test_queryresult.py::test_entity_axis_index_stays_rejected` (`qr.position[0]` at lines 292-293) —
  narrow to slices/masks only; the bare-int case moves to the new passing tests.

## Out of scope

- **Entity-axis slices** (`qr.pose[2:5]`) and **boolean masks** — these genuinely span pools and need a
  copy/concat (or a multi-pool view design). They stay rejected. Use `.numpy()` to gather first.
- **Direct entity-axis write** `qr.pose[i] = v` (the setitem path) — not needed: the read returns a
  view, so `qr.pose[i][...] = v` already mutates the pool. Add a symmetric `__setitem__` int branch
  only if a call site actually wants the one-liner; dev's call.
- **Tuple keys with a leading int** (`qr.pose[i, j]`) — index the returned view with numpy instead.

## Related

- `query_result.py:74-77` `_Field.__getitem__` (the guard to extend); `:13`/`:16` `_lens`/`_bounds`
  (the routing data, already computed); `:56-71` `__setitem__` (stays as-is); `:79-81` `__iter__` (the
  per-entity path that already works).
- `test/integration/test_i_fpv_camera_read_loop.py` — the motivating robosim scenario + the xfail to
  flip.
- `test/manual/fpv-camera-readindex/probe.py` — evidence: rejection today, working iteration, index
  resolution.
- Motivating call site: robosim `FPVCameraSystem.on_tick`.
