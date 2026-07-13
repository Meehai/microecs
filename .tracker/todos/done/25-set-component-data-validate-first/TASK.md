# set_component_data v2: deferred SET_DATA command (join the command API)

**Created**: 2026-07-13
**Closed**: 2026-07-13
**Priority**: 2

## DONE (2026-07-13)

Landed. `entity.set_component_data(component, data)` is a thin builder that appends a `SET_DATA` command;
`CommandBuffer.append` validates it and `world.update()` applies it (`_do_set_data`). **No rollback** — nothing
invalid ever stages. Validation splits cleanly:
- **component present** — projection gate (`_get_components_state == -1` rejects absent / pending-removed).
- **fields** — `_validate_component(component, strict, check_extra, **data)`: `strict=False` (set_data) allows
  a **partial** subset; `check_extra=True` still rejects **unknown** fields; dtype/shape always checked.
  (`strict` = missing-required, `check_extra` = unknown-field — orthogonal. add_component passes both; the
  plural `_validate_components` multi-component loop passes neither per step, doing its own union extra-check.)
- **schema only, not values** — NaN/Inf into a matching float passes (finiteness stays robosim #167).

Tests (tester): **24 green** — 10 buffer (`test_command_buffer.py`) + 2 entity
(`test_entity_set_component_data.py`) + the existing add/remove. The extra-field guard also restored the
regressed `test_buffer_rejects_bad_field_name`. No xfails left.

Downstream (NOT this task): robosim **#173** wires the `entity_set_data` handler onto this (deferred → applied
next tick by the main loop's `world.update()`); **#177** converter + **#167** finiteness stay robosim-side.

## Why

v1 (#24, merged) writes eagerly + snapshot-rollback. The rollback is the wart (review): ~dead code (robosim
pre-casts via `ecs_data_as_np`, so the write can't fail), half a guarantee (wrong **shape** raises but wrong
**dtype** silently truncates), not a real transaction (double-fail leaves the entity "messed up"), and it
fights task 22's "validate at the gate, apply is infallible" model.

Root cause: `set_component_data` is **command-shaped** (an entity-level "describe a mutation" method, like
`add_component`) but sits on the **eager direct-write** surface. microecs has two write surfaces:

- **Command API** — `add/remove_entity`, `add/remove_component`: deferred, staged, validated at append,
  applied at `update()`.
- **Direct SoA access** — `qr.field[:]=`, `e.field[:]=`, `e.field = v`: eager in-place (the vectorized
  perf core).

`set_component_data` belongs on the **command** side. Move it there.

## What

Add a 5th verb `CommandType.SET_DATA` and make `set_component_data(component, data: dict)` a **deferred command**:

- append ONE `SET_DATA` command holding `{entity_id, component, data}`;
- **validate fully at append** (task-22 gate): `component` in the entity's *projected* component set;
  `data` keys ⊆ that component's fields; each value's dtype+shape matches the field metadata;
- apply at `world.update()`: write every field, resolving the entity's *current* pool/row.

Consequences, all wins:
- **No rollback** — nothing invalid is ever staged → apply is infallible. Delete the `before`/try/except block.
- **Atomic** multi-field set — one command validated as a unit; if any field is bad the whole append raises,
  nothing stages.
- **Validate-first for free** (the goal we discussed) via the staging model, not bespoke checks.
- **Uniform** with `add/remove_component` — same deferred, "call `update()` to see it" semantics.

**Scope — command API only.** Direct field access stays **eager, unchanged**: `e.field = v` (`__setattr__`),
`e.field[:] = v`, and all QueryResult column writes. That's the documented+tested contract
(`test_entity.py::test_entity_write_is_eager_visible_without_update`) and the SoA performance path. Do NOT
defer those. Two surfaces, on purpose: command API (deferred) vs direct access (eager).

## How / notes

- `CommandType.SET_DATA` in `utils.py`; `Command(SET_DATA, entity_id, args={"component": .., "data": {..}})`.
- Validation lives in `CommandBuffer.append` (the single gate):
  - **structural** — reuse task-22's projection (committed set + this tick's pending adds − removes) to check
    `component` is present, and `data` keys ⊆ `[f.name for f in fields(component)]`.
  - **field data** — per-field dtype+shape check. This is a PARTIAL set (subset of fields), so NOT
    `_validate_components` (that requires all-required-present). Extract/share a per-field
    `_validate_field(component, name, value)`, or check inline from `fields(component)[i].metadata`.
- `update()` apply branch: resolve eid→(pool,row) fresh (a same-tick migration may have moved it), then
  `pool.data[field][row] = value` per field. Buffer order is preserved, so a `SET_DATA` appended after an
  `add_component` applies after the migration that created the field.

## Perf note (deliberately deferred)

Append-time validation projects component state by scanning the buffer — **O(n) in buffer size**, same as
`add/remove_component` already do. Acceptable: n is small because `world.update()` runs frequently (each
tick / after each protocol mutation). Optimize the CommandBuffer structure (e.g. an incremental
projected-state index) **only if profiling shows a bottleneck** — not now.

## Tests (tester: Claude) — rework `test/unit/test_entity_set_component_data.py`

`set_component_data` is now DEFERRED:
- happy: append → **`world.update()`** → change visible (mirror the add_component tests); add a "staged, NOT
  visible before `update()`" assertion.
- validation now at the **call** (append), before update: bad component / bad field name / **dtype mismatch**
  / **shape mismatch** → raise at append, **nothing staged** (entity untouched after a following `update()`).
- **Delete** the rollback + double-fail tests (no rollback anymore).
- **Flip** the dtype-parity cases (float→int, int64→int32, overflow) from "silently truncates (numpy parity)"
  → **rejected at append**.
- **Keep** NaN/±Inf into a *matching* float field = accepted (dtype ok; finiteness is robosim #167, not the
  store's).
- atomicity: multi-field set with one bad field → whole call raises, nothing staged.

## Relates

- Supersedes #24's eager+rollback design (merged; this replaces it).
- Extends task **22** (staging model) with the `SET_DATA` verb; reuses its projection gate.
- robosim **#173** (`entity_set_data` handler — becomes append + `world.update()`, which it already calls),
  **#177** (`ecs_data_as_np(x, dtype)` converter), **#167** (finiteness guard, robosim-side).
