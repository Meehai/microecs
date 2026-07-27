# w5 churn: three wins beyond #44 — a discarded copy, a third dtype check, and no batch spawn

**Created**: 2026-07-27
**Priority**: 2

## Why

[#44](../44-spawn-path-validates-twice/TASK.md) reclaims ~16% of a churn pair. Worth doing, but it does not flip a
cell where microecs is 12× behind ecs-pattern at N=200 and 1.2× behind EnTT at 100k. This task holds what is
left, measured on master (`f00b447`) with `test/manual/churn/churn_headroom.py`; the pair breakdown it builds on
is `test/manual/bench-compare/spawn_breakdown.py` (re-run 2026-07-27, unchanged: pair 12.8 µs = spawn 10.2 +
despawn 2.6).

Three items, independent, small → big. **Item 3 is the one that matters** — 1 and 2 are cheap while in there.

## 1. Despawn copies data it throws away — 807 ns/entity (31% of a despawn, 6% of a pair)

`Pool.pop_entity` (`pool.py:58-62`) copies every field into a dict, then `World.update()`'s `REMOVE_ENTITY`
branch (`world.py:150`) discards `_pop_from_pool`'s return value. Only `_do_add_component` /
`_do_remove_component` need the data — they carry it to the new pool.

| | ns/entity |
|---|--:|
| `Pool.pop_entity` (copy + pop-swap) | 1917 |
| `Pool.remove_entity` (pop-swap only) | 1110 |
| **discarded on the despawn path** | **807** |

Scales with field count (2 fields here), so a fat archetype pays more. Fix: give `_pop_from_pool` a no-copy
mode, or have `update()` call the pop-swap directly for `REMOVE_ENTITY`.

## 2. The SoA insert checks dtype/shape a third time — 1.64 µs of `Pool.add_entity`'s 2.88 µs

`pool.py:40-42` asserts `isinstance` / `shape ==` / `np.issubdtype` per field per spawn — after `World.add_entity`
and `CommandBuffer.append` already validated the same data twice (that pair is #44). Priced by re-running the
probe under `-O`: **2879 → 1241 ns**.

| check | ns/call |
|---|--:|
| `np.issubdtype(a.dtype, "float32")` | **653** |
| `a.dtype == "float32"` | 80 |
| `a.shape == shape` | 76 |
| `isinstance(a, np.ndarray)` | 65 |

[#34](../../done/34-asserts-to-raises-sweep/TASK.md) explicitly decided **keep as assert** — correct, and this task
does not reopen it. The point is narrower: `np.issubdtype` is **8× a plain `==`** and equivalent for the four
concrete dtypes the world allows (`float32/int32/bool/object`, `world.py:278`), so swapping that one call keeps
the assert and returns ~1.2 µs/spawn at 2 fields. `#34`'s "free under `-O`" is true and nobody runs `-O`,
including the benchmark.

## 3. There is no batch spawn — measured ceiling 99×

B entities of one archetype, spawned one at a time (B=1000, min-of-5):

| | ns/entity | per frame at B=1000 |
|---|--:|--:|
| today — `add_entity()` × B + `update()` | 10986 | 10.99 ms |
| floor — validate once + block write + id map | 111 | 0.11 ms |

The floor is ~all irreducible: two dict writes per id (`_eid_to_pool_ix`, `live_entities`). Everything else —
validation, the `Command`, the row-by-row numpy write — collapses into one pass: one shape/dtype check on the
`(B, ...)` arrays, one `pool.data[f][lo:lo+B] = arr` per field.

**This is the actual finding.** w5 is not losing to archetype layout — **spawn is the last per-entity Python loop
in the library**. Every workload microecs wins, it wins by batching; the one API with no batch form is the
structural one, while the docs tell users to batch everything including random access.

Scale at N=100k (w5 = 21.9 ms/frame, b=1000): spawn ≈ 10.2 ms, despawn ≈ 2.6 ms. Batch spawn alone takes ~10 ms
→ ~0.1 ms — enough to turn 0.82× EnTT into a win, on the only workload microecs loses at every N.

**Scope honestly:** it applies when B entities share an archetype and the data is already arrays. That *is* what
w5 models (bullet emitters, TD creep waves) and it is the shape of every real spawn burst. Scattered one-off
spawns keep today's path. Batch **despawn** is a separate, harder problem (pop-swap ordering) and is not in scope
here — item 1 is the cheap part of that end.

## How (dev writes the code)

Items 1 and 2 are local edits. Item 3 is an API decision, so state it before writing:

- **Signature.** `world.add_entities(components, **field_arrays) -> list[EntityId]` with leading axis B, vs. a
  batch-aware `add_entity`. Prefer the separate `add_entities` — no overload ambiguity between "a `(2,)` position"
  and "two scalars".
- **Contract stays deferred.** One `Command(ADD_ENTITIES, ...)` buffered, block write at `update()`. Validation
  stays eager at the call (#22). One `Command` per batch is ~270 ns total — it does not move the number.
- **Validate once, for the batch.** `arr.shape[1:] == field_shape` and `arr.dtype == field_dtype`, plus
  `len(arr) == B` agreement across fields. Defaults broadcast to `(B, *shape)`.
- **Ids are contiguous** (`_last_id + 1 … + B`), so the return can be a `range`. Do not build B `Entity` objects
  — `live_entities[eid] = None` as today.
- The prototype in the probe writes **eagerly** and bypasses the buffer; it measures the commit-side work only.
  It is a cost floor, not a design.

## Validation (tester)

- Items 1–2 are behaviour-preserving: whole `test_pool.py` + `test_world.py` green, unchanged. Item 1 needs a
  pin that a despawn still leaves the *other* entities intact after the pop-swap (the copy removal touches the
  path that reorders rows), and that add/remove_component still carry data across pools.
- Item 3 needs its own tests, all new: batch spawn lands B rows with the right values and contiguous ids; a
  mismatched B across fields raises **at the call**, not at `update()`; wrong dtype/shape raises the same
  exception type as the single-entity path; defaults fill for omitted fields; a batch spawn interleaved with
  single spawns and despawns in one tick commits in order. Extend the randomized churn invariant check
  (`test_pool_ids_stay_aligned_through_random_churn`, `test_world.py:1319`) to emit batch spawns too — that is
  what would catch an id/row-map desync.
- Re-run w5: `cd examples/05-benchmark-workloads && ./run_benchmark.sh 200 1000 5000`. **Compare ratios against
  the other libraries in the same run, never ms across runs** (absolutes drift ±10–25% — plan 1 Part 8).

## Relates

- [#44](../44-spawn-path-validates-twice/TASK.md) — do it first; same path, and item 3 makes it moot for batches
  (one validation for B entities) but not for single spawns.
- [#34](../../done/34-asserts-to-raises-sweep/TASK.md) — item 2 keeps its decision, only makes the check cheaper.
- Plan 1 Part 8 P4 — the churn breakdown that started this. Its headline ("validation, not pop-swap") is right
  but too narrow; the fuller version is **per-entity Python, not layout**.
