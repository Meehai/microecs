# Remove `qr.field[i]` (the bare-int entity-axis index) — Entity owns per-entity access now

**Created**: 2026-06-11
**Completed**: 2026-06-11
**Priority**: 3
**Breaking**: yes (pre-1.0, accepted)
**Status**: DONE — int branch removed from `_Field.__getitem__`; full suite green (224 passed). One leftover
nit for the dev: `query_result.py:65` `__setitem__` error still suggests the removed `[i][...]` spelling —
behavior is right, only the message text is stale.

## Why

`_Field.__getitem__` has one special case — a bare int — that returns a single entity's row by
*query position* (`query_result.py:80-83`, an `np.searchsorted` per access). It's the lone exception to
the rule "the entity axis crosses pools, so indexing it raises." It predates `Entity` and only existed for
ergonomics. `Entity` (task `14-entity-object`) now does the per-entity job strictly better, so this case is
dead weight that muddies the invariant.

Three reasons it should go:

1. **It's already asymmetric.** `qr.f[i]` reads, but `qr.f[i] += x` *raises* (`__setitem__` rejects int).
   Read-but-can't-write-in-place is a documented gotcha (README benchmark point 3).
2. **It's the slowest path in the benchmark** — `micro-ecs-index`, 64× slower than OOP, one
   `np.searchsorted` per access.
3. **`Entity` covers it and more** — keyed by id (survives swap/migration), all fields at once, and
   reads + writes + partial writes all work (`e.pose[0:3, 3] = ...` is a real pool view, writable). `qr.f[i]`
   only ever gave one field, one position, read-only-in-place.

The fast per-entity path is **untouched**: `zip(qr.pos, qr.vel)` goes through `_Field.__iter__` (yields from
`parts`), never `__getitem__`. We're deleting only the slow positional path; the idiomatic loop survives.

## Decision: remove outright, no deprecation

Pre-1.0; breaking changes for a cleaner API are expected. No warning shim. Migrate the one known consumer
(robosim) in the same pass.

## Fix (DEVELOPER — non-test code)

- `query_result.py`: delete the int branch in `_Field.__getitem__` (`:80-83`). Reword the `TypeError` at the
  bottom of `__getitem__` and `__setitem__` to drop the `qr[i]` / `qr[i][...]` escape-hatch references and
  point at the two replacements instead:
  - **iterate** → `for pos in qr.pos:` / `zip(qr.pos, qr.vel)` (faster, already idiomatic)
  - **random single access** → `world.get_entity(qr.entity_ids[i])` then `e.field`
  - `_bounds` / searchsorted machinery STAYS — still used by `_chunk` for ndarray operands. Only the
    `__getitem__` int case goes.

## Replacements (what users do instead)

| old | new |
|---|---|
| `qr.pose[i]` (read) | `world.get_entity(qr.entity_ids[i]).pose` |
| `qr.pose[i][0:3, 3] = v` (partial write) | `e = world.get_entity(qr.entity_ids[i]); e.pose[0:3, 3] = v` |
| `qr.fpv_camera[i].item()` | `world.get_entity(qr.entity_ids[i]).fpv_camera.item()` |
| iterate all | `for pose in qr.pose:` / `zip(qr.fpv_camera, qr.pose)` (unchanged, faster) |

## Done when

- `qr.field[i]` for a bare int raises `TypeError` with a message naming the replacements (no longer returns a
  row).
- `qr.field[:]`, `qr.field[:, k]`, `qr.field[...]`, full-field math/ufuncs, and `_Field.__iter__` (zip-rows)
  all still work unchanged.
- robosim's indexed loop is migrated to `get_entity` (or zip) and its integration test passes.
- Benchmark no longer offers `micro-ecs-index`; README table + benchmark prose updated.
- Full suite green.

## Tests / docs (tester — me)

**DONE — tests flipped red-first (2026-06-11), 6 reds until the code lands, all `DID NOT RAISE TypeError`:**

- [x] `test/unit/test_queryresult.py` — collapsed the five `qr.position[i]`-works tests into one
  `test_single_entity_int_index_is_forbidden` (in-range / negative / OOB / empty-query int all raise
  `TypeError`); added a bare-int case to `test_entity_axis_index_stays_rejected`.
- [x] `test/unit/test_field_numpy_parity.py` — added `0`, `4`, `-1` to `test_entity_axis_read_indexing_raises`
  and deleted `test_single_int_index_returns_entity_row` ("the one supported entity index").
- [x] `test/integration/test_i_fpv_camera_read_loop.py` — rewrote the robosim indexed-spelling test to
  `test_fpv_camera_per_entity_read_via_get_entity` (`world.get_entity(qr.entity_ids[i])`, asserts `qr.pose[0]`
  raises); zip-based `FPVCameraSystem` tests untouched. Verified `e.fpv_camera.item()` round-trips the camera.

**DONE — docs + benchmark written to the post-change state (2026-06-11):**

- [x] `examples/04-benchmark-ecs-vs-oop.py` — removed the `micro-ecs-index` mode, its `micro_ecs_index`
  function, and its dispatch branch. Compiles clean; no remaining `qr.f[i]` usage. (Dev OK'd editing it —
  Claude authored it originally in `test/manual/`.)
- [x] `README.md` — dropped the `micro-ecs-index` benchmark row; point 2 now "15–30×" (was "15–64×"); point 3
  rewritten to `get_entity` + "no `qr.f[i]` shortcut"; the "Per-entity systems" `qr.some_field[i]` mention and
  the edge-cases "Not a full ndarray" bullet both updated to `world.get_entity(qr.entity_ids[i])`.

**NOTE — ordering:** the README + benchmark now describe behavior the code does NOT have yet (`qr.f[i]` still
works until the `__getitem__` int branch is deleted). These edits + the red tests + the code removal must land
in the **same commit** so no commit is internally inconsistent.

## Out of scope

- Any change to `__setitem__`'s allowed forms (`[:]`, `[:, k]`) — those stay; only the int *read* in
  `__getitem__` is removed (its in-place write already raised).
- The `_bounds` / per-pool chunking machinery — still needed for operand chunking.
- `set_entity_data` / lazy-Entity work — that's task `15-lazy-entity-and-fields-set`.

## Related

- `microecs/query_result.py:79-89` `_Field.__getitem__` (delete int branch, reword error).
- `microecs/query_result.py:62-65` `_Field.__setitem__` (error wording references `[i][...]` — update).
- `test/unit/test_queryresult.py:335-370`, `test/unit/test_field_numpy_parity.py:198`,
  `test/integration/test_i_fpv_camera_read_loop.py:110`.
- `examples/04-benchmark-ecs-vs-oop.py` `micro_ecs_index`.
- Follows task `14-entity-object` (done) — `Entity` is the sanctioned per-entity access that makes this
  removable.
