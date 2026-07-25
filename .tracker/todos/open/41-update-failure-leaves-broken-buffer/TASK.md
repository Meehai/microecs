# A failed `update()` leaves a command buffer that cannot be re-run

**Created**: 2026-07-25
**Priority**: 3

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

Verified: `test/manual/structural-audit/probe_invariants.py` probe 8.

Low priority because eager validation in `CommandBuffer.append` makes a mid-update failure unlikely — it takes
an internal bug or an aliased array mutated after staging (**#39**). But "unlikely" plus "unrecoverable" is
worth 10 lines.

## What

Either `update()` is re-runnable after a failure, or it says plainly that it is not. Silent corruption is the
one option that is out.

## How (dev writes the code)

- Stop mutating `command.args`: read `args["components"]` instead of popping it, and pass the rest with a dict
  comprehension (the SET_DATA branch at `world.py:149` already does exactly this).
- Clear the buffer in a `finally`, and drop the `_cache` alongside it — after a partial apply the cached
  QueryResults are stale regardless of whether the loop finished.
- Consider re-raising wrapped in an error that names how many commands committed before the failure. Cheap,
  and it is the one thing a caller needs to know.

## Validation (tester)

Force a failure mid-buffer (monkeypatch `_pop_from_pool`), then assert: the exception propagates, the buffer is
empty, the query cache is dropped, and a subsequent `update()` is a clean no-op rather than a `KeyError`.

## Relates

- **#22** (buffer is a staging area, `update()` not atomic) — this defines what "not atomic" costs.
- **#39** — an aliased array mutated after staging is the one user-reachable way to trigger this.
