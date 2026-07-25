# Staged writes hold a reference to the caller's array, not a snapshot

**Created**: 2026-07-25
**Priority**: 2

## Why

`add_entity(**kwargs)` (`world.py:80-81`) and `set_data(**data)` (`entity.py:66`) put the caller's array into
the command; `_do_set_data` / `_add_to_pool` read it at `update()` time — whatever it holds *then*. So the
value that lands is not the value at the call. The idiomatic numpy scratch-buffer loop silently breaks:

```python
scratch = np.zeros(2, "float32")
for i in range(3):
    scratch[:] = i
    world.add_entity([HasPosition], position=scratch)
world.update()      # committed: [2,2], [2,2], [2,2]   expected: [0,0], [1,1], [2,2]
```

This is a consequence of #29 making the entity path buffered — the old eager path could not have it. "Buffered"
should mean *the value at call time is what lands*; right now it means *we will re-read your variable later*.
Validation has the same hole: dtype/shape are checked at `set_data`, then the array can be swapped for a bad
one before `update()`.

Not currently hit by robosim, but reusing a scratch array in a spawn loop is normal numpy style — this is the
finding most likely to bite a user.

Verified: `test/manual/structural-audit/probe_invariants.py` probes 5 / 5b.

## What

Staged data is a value, not a promise.

## How (dev writes the code)

Copy on stage, in `CommandBuffer.append` (one place, after validation already passed, covers ADD_ENTITY /
ADD_COMPONENT / SET_DATA alike) rather than at each call site.

Cost: one `np.array(v, copy=True)` per staged field. Measure it against `#36` — that task is paying down a
7-14% physics-tick regression on this exact path, so land the two together and benchmark once. If the copy is
too expensive for the hot path, the fallback is to **document it hard** (already done in
`docs/source/primitives.md`) and leave the semantics — but silent-wrong-by-default is the worse default.

`dtype=object` fields: shallow copy only. The array is snapshotted, the referenced object is not, and that is
correct — see **#40**.

## Validation (tester)

Mutate the source array between stage and `update()` for each of the three verbs; assert the committed value
is the one from call time. Plus the spawn-loop case above.

## Relates

- **#29** (one entity write path) — introduced the buffered path this applies to.
- **#36** — same hot path; benchmark the copy together with those fixes.
