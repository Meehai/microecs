# Staged writes hold a reference to the caller's array, not a snapshot

**Created**: 2026-07-25
**Priority**: 2
**Closed**: 2026-10-18

## Why

`add_entity(**kwargs)` put the caller's array into the command; `_add_to_pool` read it at `update()` time —
whatever it held *then*. So the value that landed was not the value at the call. The idiomatic numpy
scratch-buffer loop silently broke:

```python
scratch = np.zeros(2, "float32")
for i in range(3):
    scratch[:] = i
    world.add_entity([HasPosition], position=scratch)
world.update()      # committed: [2,2], [2,2], [2,2]   expected: [0,0], [1,1], [2,2]
```

This is a consequence of #29 making the entity path buffered — the old eager path could not have it. "Buffered"
should mean *the value at call time is what lands*; right now it means *we will re-read your variable later*.
Validation had the same hole: dtype/shape were checked at the call, then the array could be swapped for a bad
one before `update()`.

Not currently hit by robosim, but reusing a scratch array in a spawn loop is normal numpy style — this is the
finding most likely to bite a user.

Verified: `test/manual/structural-audit/probe_invariants.py` probes 5 / 5b.

## What

Staged data is a value, not a promise.

## How (dev writes the code)

Copy on stage, in `CommandBuffer.append` (one place, after validation already passed, covers ADD_ENTITY /
ADD_COMPONENT alike) rather than at each call site.

Cost: one `np.ndarray.copy()` per staged field. Measure it against `#36` — that task is paying down a 7-14%
physics-tick regression on this exact path, so land the two together and benchmark once. If the copy is too
expensive for the hot path, the fallback is to **document it hard** and leave the semantics — but
silent-wrong-by-default is the worse default.

`dtype=object` fields: shallow copy only. The array is snapshotted, the referenced object is not, and that is
correct — see **#40**.

## What landed

`CommandBuffer.append` snapshots every `np.ndarray` value in `command.args` after the per-command validation
and normalization, before the command enters the buffer. Both `ADD_ENTITY` and `ADD_COMPONENT` reach that tail,
so `add_entity`'s default-filled args and `add_component`'s converted args are covered in one place. Object
arrays are shallow copies: the array is private, the referenced objects remain shared (#40).

## Validation (tester)

Landed in `test/unit/test_world.py` unless noted:

- `test_staged_spawn_snapshots_the_values_at_the_call` — mutate the source after `add_entity`.
- `test_staged_spawn_snapshots_the_callers_array` — resize the source after `add_entity`; no commit failure.
- `test_staged_add_component_snapshots_the_values_at_the_call` — mutate and resize after `add_component`.
- `test_spawn_loop_snapshots_each_scratch_buffer` — the scratch-buffer loop above commits `[0,0] [1,1] [2,2]`.
- `test_staged_object_field_snapshots_the_array_not_the_object` — outer array copied, referenced object shared.
- `test_buffer_snapshots_spawn_arrays_but_not_the_components_list` (test_command_buffer.py) — the buffer-level
  pin: args rebuilt, array copied, `components` list passed through.

Full suite at close: 552 passed, 10 xfailed.

## Relates

- **#29** (one entity write path) — introduced the buffered path this applies to.
- **#36** — same hot path; benchmark the copy together with those fixes.
