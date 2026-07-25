# Finish the assert → raise sweep (14 asserts left, one decision each)

**Created**: 2026-07-25
**Closed**: 2026-07-25
**Priority**: 3

## Why
Python strips `assert` under `-O`, so an assert is only valid for *our own bugs*. Anything reachable from user
input must `raise` or it silently disappears in production. Tasks 20, 22 and 31 converted the paths they touched;
this closes the sweep and writes the rule down.

Measured example (`world.py:24`, `extra_metadata`): under `-O`, `World(components=[C], extra_metadata="serializable")`
passes the isinstance assert, the str is splatted into characters, and construction dies with
`expected_meta = {'i','e','s','z','l','r','b','a','shape','dtype','default'}` — the right rejection for an
unrecognizable reason.

## What — done
**Converted to `raise`:**
- `world.py:23` — `extra_metadata` not a list → `TypeError`.
- `world.py:180` — `_do_add_component` duplicate keys → `ValueError`. (Task 23 wants this rejected earlier, at
  construction; until then this is the last line of defence and it now survives `-O`.)
- `pool.py:49` — `remove_entity(index)` OOB → **`IndexError`** (not the `ValueError` first sketched: it is an
  index, and `IndexError` is what a python sequence raises). Measured with the check removed: the call succeeds,
  a never-live row is written, `size` goes to -1 and `len(pool)` itself then raises. The bound is `0 <= i < size`
  — the first cut only guarded the top, and a negative index then swap-removed the *tail*, so the caller asked
  to drop a nonexistent entity and a different one vanished. Same hole closed for `pop_entity`, which delegates.

**Kept as `assert` — our own bug, cheap, some hot (10 total, all allow-listed in the policy test):**
`command_buffer.py` `append` ×2, `world.py` `_add_to_pool`, `world.py` `_do_remove_component`,
`query_result.py` `__init__`, `qr_field.py` `_apply_fn_on_parts`, `qr_field.py` `__array_ufunc__`,
`pool.py` `add_entity` ×3.

`pool.py:40-42` (isinstance/shape/dtype, per field per spawn) was the one explicit decision: **keep as assert**.
World validates eagerly at the call (tasks 20, 22, 25), so on every World-driven path these are duplicates —
free under `-O`, and they still catch our own bug in the direct-`Pool` case.

**`Pool.__init__`'s two checks stay `raise`** (asked during the sweep): `Pool` is exported, so the ctor is a
front door; it runs once per archetype, not per spawn, so there is no cost argument; and with the checks gone a
len mismatch lets `zip()` truncate (later `KeyError` inside `add_entity`) and a reserved name shadows the method
`Pool` itself needs. Rule of thumb the sweep produced: **`raise` at the constructor, `assert` inside the loop.**

**Rule written down** in `docs/source/primitives.md` ("`raise` vs `assert`: who made the mistake?"). Same edit
fixed doc drift from the `*_RESERVED_ATTRS` → `*_RESERVED_NAMES` rename.

**Drive-by (from task 31), done:** `POOL_RESERVED_NAMES` is now one module-level name in `pool.py` after the
class body, used by both the check and its message, and imported by `world.py` — the duplicated union is gone.

## Validation (tester) — done
Each converted path has an ordinary test pinning the exception **type**, which is what catches a revert to
`assert`: `AssertionError` is not an `IndexError`/`TypeError`/`ValueError`, so the test goes red. `test_world.py`
gained the `extra_metadata` and field-clash-at-`update()` rejections; `test_pool.py`'s two OOB tests now expect
`IndexError` (they had been pinning `AssertionError` and went red the moment the conversion landed — the
mechanism working in the other direction).

`test_pool.py` also pins the negative half of the bound: `remove_entity(-1 / -3 / -4)` and `pop_entity(-1)`
raise and leave the pool *unchanged in contents*, not just in size — the bug moved data, so a `len` assertion
alone would have missed it.

**Dropped from the plan:** the "`-O` subprocess test per converted path" this task originally asked for, and the
AST allow-list of every remaining assert that went with it. The type-pinning tests above already catch
assert-vs-raise, so the subprocesses bought nothing, and an allow-list of assert sites is a process guard, not a
test of behavior — the rule belongs in the docs, where it now is. The one demonstration of the `-O` mechanism
(`test_world_reserved_name_guard_survives_python_dash_o`, task 31) stays.

Suite: 331 passed, 1 xfailed.

## Follow-up
- Task 23 — construction-time unique-field-name guard (would make `world.py:180` unreachable).
