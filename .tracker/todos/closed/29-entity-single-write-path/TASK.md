# Entity: one write path — `set_data(**kwargs)`, forbid eager `e.field =` / `e.field[:] =`

**Created**: 2026-07-24
**Closed**: 2026-07-25
**Priority**: 3

## DONE (2026-07-25) — library side complete, **367 green**

`set_data(**fields)` is the single entity write path. All three eager idioms now raise:
- `e.field = v` → `AttributeError` naming `set_data` (`__setattr__` routes only `_ENTITY_INTERNAL_ATTRS`).
- `e.field[:] = v`, `e.field[0] = v`, `e.field += v` → `__getattr__` returns a **read-only row view**
  (`setflags(write=False)`, guarded by `isinstance(row, np.ndarray)` because a shape-`()` field comes back as a
  numpy scalar that has no `setflags` and is already immutable).

Supporting pieces: `world.field_to_component` (keyed by `f.name`) resolves a field name to its component, so
the `component` arg is gone; the duplicate-field-name guard (task **23**) now fires at World construction,
which is what makes that resolution well-defined. SET_DATA's command args are **flat** —
`{"component": c, <field>: value, ...}` — matching ADD_COMPONENT, so both verbs project field data identically.

**Atomicity — the part that needed the most care.** `set_data` groups kwargs by component (one command per
component, so `append`'s all-or-nothing validation covers every field of that component) and wraps the append
loop in try/except that slices the buffer back to its pre-call length. Grouping alone is not enough: K
components = K commands, so command 2's rejection would leave command 1 staged, the call would raise **and**
the write would land at the next `update()`. Concretely dangerous because robosim's handler
(`protocol.py:186-190`) catches the exception, returns `{"error": ...}` to the client and skips its own
`update()` — but the main loop calls `world.update()` every tick anyway, so the half-write lands one frame
later, from a command the server already reported as rejected.

The revert must be a **slice from `ix_before`**, not `buf.data.clear()` — a failing `set_data` is usually not
the first thing staged this tick, and other callers' commands must survive.

### Fixed during review (all measured, see `test/manual/29-entity-set-data/`)
1. `_world_field_to_component` missing from `_ENTITY_INTERNAL_ATTRS` → `Entity` unconstructable, `get_entity()`
   raised for every entity (was 95 failures).
2. `field_to_component` keyed by the `dataclasses.Field` object → name lookups KeyError'd and the duplicate
   check could never fire (`Field` hashes by identity).
3. SET_DATA args shape half-flattened — `entity.py` emitted flat while both consumers read `args["data"]`.
4. `__setattr__` raised `ValueError`; `AttributeError` is protocol-correct (`hasattr`/`copy` rely on it).
5. Non-atomic per-field append loop (above).

### Side effects
- `test_add_component_with_field_name_clash_raises_at_update` **removed** — unreachable now that `World()`
  rejects the clash. The `Duplicate keys` raise in `_do_add_component` stays as defence-in-depth.
- Task 23's xfail became a live test.
- **Scope note:** the read-only view blocks *replacing* an object reference, not mutating what it points at —
  `e.label[0]["v"] = 99` still writes through. Inherent to `dtype=object`, and robosim depends on it (it mutates
  stored `FPVData`/`Camera`). "All entity mutation goes through `set_data`" holds for numeric fields; for object
  fields it means "the *reference* is only replaceable via `set_data`".
- Spun off task **35** (reserve World's internal kwarg names): a field named `component` / `components` /
  `strict` / `check_extra` passes `World()` and dies at `add_entity` with a confusing internal `TypeError`.
  Pre-existing, sharpened by the flat args shape.

### Downstream — robosim, NOT this task (dev, immediately after)
All three sites still use the removed API and will crash:
- `src/robolib/plugins_manager.py:81` — `setattr(entity, key, value)` loop → one `entity.set_data(**payload)`.
- `src/robosim/protocol.py:187` — `set_component_data(comp, data=...)` → `set_data(**component_data)` (drop the
  `component` lookup at :183-185).
- `src/robosim/simulator_object.py:148-153` — four `robot.<field>[:] =` object writes → one `set_data(...)`.

## Why
An Entity has **three** ways to write a field and they disagree on semantics:
- `e.field = v` — eager pool write via `__setattr__` (`entity.py:86`).
- `e.field[:] = v` — eager in-place write on the view `__getattr__` returns (`entity.py:75`).
- `e.set_component_data(component, data)` — buffered `SET_DATA` command (`entity.py:49`).

**Entity-level ops are buffered, always** — like `add_component` / `remove_component`, applied at
`world.update()`. The two eager paths violate that invariant: they poke the pool immediately, so a write lands
at a different time depending on which idiom you reached for. Consolidate on **one** buffered path.

## What
1. **`set_data(self, **kwargs)`** — rename `set_component_data` → `set_data`, drop the `component` arg, take
   field values as kwargs (subsumes the old kwargs ask). Keys are field names; the entity's pool namespaces
   fields flat, so a name identifies the target without a component (relies on field-name uniqueness — task 23
   formalizes it). Buffered `SET_DATA` command applied at `world.update()` (the always-buffered rule);
   transactional semantics from task 24; validation stays eager at the call (cf. #22).
2. **Forbid eager writes.** `__setattr__` raises `AttributeError` for any field (route to `set_data`); only
   `ENTITY_INTERNAL_ATTRS` still route to `super()`. To also block `e.field[:] = v`, `__getattr__` returns a
   **read-only view** (`arr.setflags(write=False)` on the row slice) — cheap, reads unaffected, in-place writes
   raise. All entity mutation then goes through `set_data` / `add_component` / `remove_component`.

## Downstream (robosim relies on the eager path — dev updates in the same change)
- `src/robolib/plugins_manager.py:75-82` — the per-key `setattr(entity, key, value)` loop → one
  `entity.set_data(**event.payload)`. This is the existing `TODO(microecs)` at line 80.
- `src/robosim/simulator_object.py:148-153` — four eager `robot.<field>[:] = ...` object writes (channel,
  fpv_camera, fpv_data, fpv_texture) → `set_data(...)`.
- `src/robosim/protocol.py:187` — `set_component_data(comp, data=...)` → `set_data(**component_data)` (drop the
  `component` lookup at :183-185).

## Notes / to verify
- **Buffered timing (settled: entity ops are always buffered).** `set_data` writes land at the next
  `world.update()`, not immediately — same as `add_component`. The `setattr` it replaces in plugins_manager is
  eager, so confirm the event-apply loop is followed by an `update()` before systems read the values.
- **Read-only reads.** Confirm no robosim path writes *through* a read result other than the four sites above
  (the `robot.pose[0:3,3]` / `collider_bbox[0:2]` reads stay fine — read-only slices still read).

## Validation (tester) — all green (367), mutation-checked
- `test_entity.py` — every eager idiom raises AND leaves the pool untouched: `e.f = v`, `e.f[:] = v`,
  `e.f[0] = v`, `e.f += v`, object-dtype `e.f[0] =`, 0-d `e.f[...] =`. Reads (incl. sliced `e.pose[0:3, 3]`
  and liveness against a bulk `qr` write) still work. Plus the allowlist invariant: `set(vars(e)) ==
  _ENTITY_INTERNAL_ATTRS` — the meta-test that catches defect 1 and any future one like it.
  - `e.f += v` is the sharp one: numpy mutates the row in place and *then* `__setattr__` raises, so before the
    read-only view a **failed** write still dirtied the pool ([1,2] → [11,12], measured in the probe).
- `test_entity_set_data.py` — kwargs API: resolves field→component, single command per component, deferred,
  cross-component in one call, partial multi-field, copy-vs-reference, 0-d; eager rejects (unknown field /
  field of an absent component / shape / dtype / non-array) all stage nothing; all-or-nothing across the whole
  call (same-component and cross-component); row resolution after swap-remove and same-tick migration; stale
  handle. `set_component_data` asserted gone.
- `test_world.py` — `field_to_component` keyed by name, covers every field, resolves what `set_data` passes;
  duplicate field name rejected at construction (the task-23 xfail is now a live test).
- **Mutation-checked** (`test/manual/29-entity-set-data/mutation_check.py`): each wrong `set_data` is caught —
  no grouping → 9 fail, grouping without the revert → 6 fail, `data.clear()` instead of the slice → 2 fail
  (incl. `..._unstages_only_its_own_commands`, written specifically to kill that one).
- `test/manual/29-entity-set-data/` also holds `probe.py` (read-only view feasible for every shape/dtype, pool
  stays writable), `atomicity_probe.py` (which rejections leave a partial staged) and `run_tests_with_impl.py`
  (ran the whole suite against the intended implementation while it was being written — the satisfiability
  proof that no test demanded the impossible).

## Relates
- Supersedes the original item-12 kwargs-only change.
- Relies on field-name uniqueness → task 23.
- `add_component` / `remove_component` (`entity.py:26/31`) — the buffered mutators `set_data` sits beside.
- Source: robosim `183-feedback` item 12.
