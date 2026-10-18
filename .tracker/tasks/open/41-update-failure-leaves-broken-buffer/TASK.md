# A failed `update()` leaves a command buffer that cannot be re-run

**Created**: 2026-07-25
**Priority**: 3
**Closed**: n/a

## Why

`update()` is deliberately **not** atomic (**#22**: the buffer is a staging area, validated eagerly, applied
in order). Fine — but then a mid-loop failure must leave the world in a state you can reason about, and it
does not:

- `command.args.pop("components")` (`world.py:139`) **mutates the command in place**. The pop happens before
  the work, so a command that was already consumed is now malformed.
- the buffer is only cleared *after* the loop (`world.py:153-155`), so on an exception it survives intact —
  including the commands that already applied.

Retrying `update()` therefore raises `KeyError: 'components'`, and any command that did succeed would apply
twice. There is no path back to a consistent world; the only recovery is to throw the world away.

### Reachable? Not through the old alias route

Eager validation does not make `update()` unable to fail, but the public alias route is gone: **#39** landed and
`CommandBuffer.append` snapshots staged arrays, so changing the caller's array after `add_entity` /
`add_component` no longer reaches commit. The recovery tests inject the bad shape into the buffer's own staged
snapshot instead. The lesson stands for any internal producer that queues an invalid command.

```python
world.add_entity([P], p=np.zeros(2, "float32"))   # valid at the call
world._command_buffer.data[0].args["p"].resize(3, refcheck=False)
world.update()                                    # ValueError in Pool.add_entity, mid-loop
```

Three things the original write-up missed:

1. **The world is not just unrecoverable, it is bricked and misdiagnosed.** The malformed command never
   leaves the buffer, so *every* later `update()` raises the same `KeyError: 'components'` — forever. An app
   that logs-and-continues (any server loop) spins on an error naming a key, three frames after a resized
   array it will never connect to. That, not the first exception, is the damage.
2. **`-O` does not save it and does not change it.** The tripped guard is an `assert`, but numpy raises
   `ValueError: could not broadcast input array` on the same write, so the failure is identical under `-O`.
3. **The stale `_cache` survives too** — a query taken before the failed commit still answers, from pools
   that were partially mutated.

Repro: `test/manual/41-failed-update/probe.py`. Original: `test/manual/structural-audit/probe_invariants.py`
probe 8.

Priority stays **3**: **#39** closed the public route, so this now covers internal bad commands. A commit can
still fail, and when it does the buffer is bricked. #39 stopped the known trigger; this stops any trigger from
being permanent.

## What

Either `update()` is re-runnable after a failure, or it says plainly that it is not. Silent corruption is the
one option that is out.

## How (dev writes the code)

- Stop mutating `command.args`: read `args["components"]` instead of popping it, and pass the rest with a dict
  comprehension. **This one line is most of the value**: it is what turns a one-off failure into a permanent
  `KeyError`, so on its own it makes a retry either work or fail with the *original* error instead of a
  meaningless one.
- Clear the buffer in a `finally`, and drop the `_cache` alongside it — after a partial apply the cached
  QueryResults are stale regardless of whether the loop finished.
- Consider re-raising wrapped in an error that names how many commands committed before the failure. Cheap,
  and it is the one thing a caller needs to know.

## Validation (tester)

**Written, strict-xfail, in `test/unit/test_world.py`** — they turn green the moment this lands. The trigger
widens the buffer's own staged snapshot, so each test doubles as the repro for any internally malformed command:

- `test_failed_update_leaves_an_empty_buffer`
- `test_update_after_a_failed_update_is_a_clean_noop` — the headline: one bad frame costs one frame, not the world
- `test_failed_update_drops_the_query_cache`
- `test_failed_update_propagates_the_original_error` — **green today**, the control: no fix may swallow the failure

Checked that all three pass under exactly the fix prescribed above, with nothing else breaking:
`test/manual/41-failed-update/prototype_fix.py` (462 passed with `--runxfail`). Note they pass **only with both
halves** — read-not-pop alone leaves the buffer full, so a retry re-applies committed commands.

## Relates

- **#22** (buffer is a staging area, `update()` not atomic) — this defines what "not atomic" costs.
- **#39** — an aliased array mutated after staging is the one user-reachable way to trigger this, now
  verified rather than assumed. Do #39 first; it removes the trigger, and this task then only covers
  internal bugs.
- **#43 subtask 2** — `CommandBuffer.removed_this_tick` also survives a failed `update()`, but it does **not**
  lie: the tick never ended, so "removed this tick" stays true for both the applied and the unapplied id.
  Checked; nothing to do here.
