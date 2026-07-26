# `set_data` goes eager again — `e.field = v` comes back, `set_data` wraps `setattr`

**Created**: 2026-07-26
**Closed**: 2026-07-26
**Priority**: 2

## Outcome

Shipped. `__setattr__` writes the pool row eagerly through `_locate(names=...)` — one gate that resolves the
row *and* checks the names, shared by read, write and `set_data`; `set_data` is the transaction over it
(single field → straight write, N>1 → convert + fit-check all, then write, no rollback because nothing can
fail after validation); `SET_DATA`, `_do_set_data`, the append branch, the rollback dance and the read-only
row view are all deleted (**net −40 lines**).

Two calls made along the way, both recorded above: a write goes where the row is **now** and never consults
the command buffer (so spawn-then-write and pending-component writes raise), and a write follows **numpy's
rules** — the same ones `qr.field = v` has always followed — which unifies the two write paths and leaves
strictness only on `add_entity` / `add_component`.

**Verified**: 432 unit+integration tests, pylint 10.00/10, `MUT=none` green with all 3 mutants caught
(11/5/16 failures), docs rebuilt. e2e: all 8 core correctness suites pass (`protocol-fuzz` fails on master
too — robosim's own `collision.py` `NotImplementedError` on a fuzzed `collider_kind`, out of scope here).
`perf-physics-tick` is **faster than baseline `2e3517e` at every N**, medians of 3 runs: −12/−16/−12% at N=1,
−9/−10/−7% at N=10, −5/−8/−3% at N=100 — the #29 read tax (#36 measured +6–7%) came back.

Follow-ups spun out, none blocking: finding 1 in plan 2 is only **half** fixed (`get_components`,
`get_fields`, `to_dict`, `has_component` still raise a bare `KeyError`, ~4 lines to route through `_locate`),
and `get_fields()` hands out the live `pool.fields_set`.

## Why

The eager/deferred line is on the wrong axis. Today it is **`Entity` = buffered, `QueryResult` = eager**
(#29). It should be **structural change = buffered, data write = eager**, on every surface. That rule is one
line, easy to explain and easy to check — which the current one is not.

Buffering *structure* is non-negotiable: moving a row between pools invalidates iteration in flight.
Buffering a *data write* buys nothing — `pool.data[f][ix] = v` moves no row and invalidates no query. It also
does not buy the thing deferral usually buys, snapshot semantics, because `update()` is deliberately not
atomic (#22) and staged values are references, not copies (#39).

#29 was a good experiment. The bill is not worth paying:

- **A permanent read tax.** The read-only row view runs `isinstance` + `setflags(write=False)` on *every*
  entity field read: **491 vs 227 ns/op** (2.2×) — what #36 measured as +6-7% on robosim's physics tick.
- **Machinery per write.** `Command` object + kwargs dict + buffer append + `_do_set_data` at commit, plus
  `_get_components_state`'s O(buffer) scan per append: **402 µs** for 100 `set_data` + `update()`, vs
  **1.1 µs** for the equivalent batch write.
- **Read-modify-write stopped composing, silently.** Eager reads + buffered writes ⇒ `damage(3); damage(4)`
  leaves 6, not 3 (plan 2 finding 5). Any accumulator written through `Entity` keeps only the last
  contribution — worst with physics substeps and with two systems touching one field.
- **`Entity` stopped being a view of a row.** It is a write-only mailbox with a read-through cache of the
  previous tick.

#36 accepted the tax because it bought the single-write-path guarantee. This task drops that guarantee by
decision, so the tax has nothing left to buy. It does **not** reopen #36's measurement — it deletes the thing
measured.

## What

1. **`__setattr__` writes eagerly.** For a field name: validate, then `pool.data[name][pool_ix] = value`.
   Only non-field, non-internal names still raise.
2. **`__getattr__` hands back a writable row again.** Drop `setflags(write=False)` and the `isinstance`
   branch (`entity.py:88-91`). `e.field[:] = v` and `e.field += v` are legal again — and `+=` now composes.
3. **`set_data(**data)` becomes a thin wrapper**: validate every field first, then a loop of eager writes.
   Multi-field / multi-component stays all-or-nothing (#24, #25's validate-first pattern) — that is the only
   reason it is not a bare `for k, v: setattr(self, k, v)`. Keep it: it is the one-call transaction and
   robosim's three call sites use it.
4. **Delete** `CommandType.SET_DATA`, its branch in `world.update()` (`world.py:148-149`) and in
   `CommandBuffer.append` (`command_buffer.py:100-107`), `World._do_set_data` (`world.py:201-204`), and the
   buffer-rollback dance in `Entity.set_data` (`entity.py:63-70`). Net negative LoC.
5. **Unchanged**: `add_entity` / `remove_entity` / `add_component` / `remove_component` stay buffered, still
   validated eagerly at `append`. Nothing about the buffer's structural half moves.

### The two wrinkles

> **Decided 2026-07-26 (dev): a write goes where the row is NOW, and consults the command buffer never.**
> One rule, no scan, no command patching:
> - **Spawn-then-write** raises the same `AttributeError` a read already raises — reads and writes agree. Spawn
>   data belongs in `add_entity`'s kwargs.
> - **Corollary the task did not cover** (two existing tests invert either way): a field whose component is only
>   *pending* has no column yet, so `e.b = v` / `set_data(b=v)` **raises** until `update()`; a field whose
>   component is pending *removal* still has its column, so the write **lands** and `update()` then drops it with
>   the component.

- **Entity needs a validator.** It holds `_eid_to_pool_ix`, `_world_command_buffer`, `_world_field_to_component`
  — no world ref, and validation lives on `World._validate_component(component, strict=False, check_extra=True,
  **data)`. `CommandBuffer.world` already exists, so a `_world` back-ref (replacing the two `_world_*` maps) is
  the clean shape. **Any new attribute must be added to `_ENTITY_INTERNAL_ATTRS`** — forgetting that was #29's
  defect 1 (95 test failures); the `set(vars(e)) == _ENTITY_INTERNAL_ATTRS` meta-test in `test_entity.py`
  catches it, and `ENTITY_RESERVED_NAMES` derives from the same set.
- **Spawn-then-write in one tick.** An entity spawned this tick has no row, so an eager write has nowhere to
  go. Today `add_entity` → `get_entity` → `set_data` → `update()` works (plan 2, Part 2 "came back clean").
  Either keep it by patching the staged `ADD_ENTITY` command's args in place (~10 lines; `CommandBuffer`
  already scans for exactly this entity in `_get_entity_components`), or make the write raise the same
  "not committed yet" `AttributeError` that a read already raises. **Dev's call** — reads and writes agreeing
  is the more honest contract, patching the command is the more convenient one. Robosim does not hit it (all
  three `set_data` sites run on committed entities), so pick on taste and document it either way.

## What we give up (accepted, must be documented not silent)

- **"No stray write can corrupt a pool" is gone.** `e.field[:] = v` is legal, so a row view stashed across an
  `update()` can be written after the entity migrated pools — writing into whatever row now lives at that
  index. This is exactly the pre-#29 status quo that robosim ran on for months; the rule is the same rule
  queries already have (#27): don't hold a view across `update()`.
- **`set_data` is no longer the *only* write path** — it is the transactional convenience over `setattr`.
  Two idioms, one timing, which is the trade being made.

## How — docs (same change, not a follow-up)

> **Done 2026-07-26** (after the suite went green — 432 pass, all mutants caught). `primitives.md`: the
> `Entity` bullet and the "few relevant concepts" bullet rewritten, §"Mutation timing" retitled to
> *structural changes are buffered, data writes are eager* (new anchor, both inbound links updated) and
> rewritten around one rule + numpy's write contract + `set_data`-as-transaction + the three
> "where the row is now" edges + the don't-hold-a-view-across-`update()` hole; the `add_entity`
> hold-by-reference trap kept but scoped to spawns; a short **History** subsection records why #29 was
> reversed. The `qr.f[0] = x` known-gap row lost its "unbuffered" contrast (the reason is addressing, not
> timing) and `ENTITY_RESERVED_NAMES` now mentions private methods (`_locate`). `systems.md` §3 rewritten.
> `pdoc` site rebuilds clean (9 modules, 6 prose pages). No `examples/*.py` change needed — they only use
> the batch path — but note `examples/04-benchmark-ecs-vs-oop.py` (`ent.position += ...`) and
> `examples/05-benchmark-workloads/probes/microecs_random.py` (`e.hp = ...`) were **broken by #29** and run
> again now; the get-entity benchmark measures 0.227 s at N=100k.
>
> **Left for the dev (source docstrings, rendered into the same site):** `entity.py:50` still says
> `set_data` "May leave entity broken if crashes midway" — false since the two-pass landed; and
> `world.py:90` `get_entity` still says "Lazy; call world.update()", which was the all-writes-buffered
> contract.

- `primitives.md` §"Mutation timing: the `Entity` API is buffered, the `QueryResult` API is eager" → retitle
  to the new rule; the `e.set_data(...)` row moves to **eager**; lines 34-35 (`e.field = x` is not a thing /
  read-only view), 44-48 (cannot read your own write) and 49-51 (validated eagerly, applied lazily) are
  rewritten or deleted; line 6 and 14's "all writes go through `set_data` and are buffered" as well.
- `systems.md:51-61` — drop "Every write through an `Entity` is buffered" and the `e.position = ...` raises
  comment. §3's `Entity`-is-buffered caveat disappears.
- `example-1-hello-world.md:74` still holds (that one is about `add_entity`, which stays buffered).

## Validation (tester)

This changes intra-tick visibility, so the real gate is **robosim e2e parity**, not unit tests: run
`bash test/e2e/run_all.sh` and `perf-physics-tick` before/after. Take **medians of ≥3 runs** — a single run
cannot see a 10% effect on that harness (#36).

> **Written 2026-07-26 (TDD, red).** The whole list below is in `test/unit/test_entity.py` (the view: eager
> writes, composability, what still raises, the accepted hole) and `test/unit/test_entity_set_data.py` (the
> transaction), plus `test_buffer_verbs_are_the_four_structural_ones` in `test_command_buffer.py` and the
> mutants in `test/manual/42-set-data-eager/mutation_check.py`. Against the in-flight implementation
> (`__setattr__` = `getattr(self, name)[:] = value`, `set_data` = bare `setattr` loop) **30 fail**, in four
> groups: (a) `__setattr__` does not validate — float64 truncates, a `(1,)` broadcasts into a `(2,)` row, a
> python list is accepted; (b) `[:] =` cannot write a 0-d field — use `pool.data[name][ix] = value`; (c) a
> non-field name (`e.to_dict = arr`) raises `TypeError` from numpy instead of `AttributeError` naming the valid
> fields — check `name in pool.fields_set` first; (d) `set_data` has no transaction — validate every field
> before writing any (that is the `# TODO: rollback?` in the current draft: no rollback needed, just ordering).

Unit side, on top of keeping the whole #29 suite green minus the eager-reject tests (which invert):

- RMW composes: `damage(3); damage(4)` ⇒ 3 (plan 2 finding 5, and #1's accumulator).
- Read-your-own-write, no `update()` in between; and a query taken *before* the write sees it (same memory).
- `set_data` still all-or-nothing on a bad field — same-component and cross-component — and writes **nothing**
  when it rejects (this is the one guarantee that has no buffer to fall back on now).
- `e.field = v` writes the pool; wrong dtype/shape raises and leaves the pool untouched; unknown field name
  raises with the valid-field list (plan 2 finding 15).
- `e.field[:] = v`, `e.field += v`, `e.f[0] = v`, object-dtype and 0-d fields all write through.
- Row resolution after a swap-remove and after a same-tick migration; stale handle still raises.
- The `_ENTITY_INTERNAL_ATTRS` meta-test above.
- Mutation-check the transaction like #29 did (`test/manual/29-entity-set-data/mutation_check.py` is the
  template): write-then-validate instead of validate-then-write must fail tests.

## Relates

- **Reverses the timing half of #29** and its read-only view; keeps `set_data`'s signature, so robosim's three
  call sites (`plugins_manager.py:78`, `protocol.py:188`, `simulator_object.py:152-155`) need no change.
- **Plan 2 Part 3** (`plans/2-app-level-audit-and-mutation-timing.md`) is the argument in full. It proposed
  keeping the read-only view and changing only the timing; this task goes further and drops the view too,
  because that view *is* the measured read tax and the guarantee it buys is not worth a 2.2× read.
- **#36** — closed as "cost accepted". Fixes 1 (read-only column alias) and 4 (single-field fast path) become
  moot; fix 3 (O(buffer) append scan) shrinks to structural commands only; fix 2 (don't clear the query cache
  for a data-only buffer) is moot for the same reason — a data-only buffer no longer exists.
- **#39** (staged writes are references) closes for this path — an eager write copies at call time. The
  `add_entity` half of #39 remains.
- **#41** (failed `update()` leaves a broken buffer) gets a smaller buffer to go wrong, not a fix.
- **#22** (buffer is a staging area, `update()` is a pure apply) — unchanged and now stated only about
  structure, which is what it was always really about.
