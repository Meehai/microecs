# Fully-eager staging for add/remove_component (buffer is always valid, update() stays a pure apply)

**Created**: 2026-07-08
**Closed**: 2026-07-09
**Priority**: 2

## Why

The old committed-only eager checks in `Entity.add_component`/`remove_component` were **inconsistent** —
they judged dup/absent against `get_components()` (committed truth), which can't see the pending buffer.
Two consequences, both confirmed by probe:

- **Valid same-tick sequences were rejected.** `add(V)` then `remove(V)` → the remove raised "does not
  exist"; `remove(V)` then `add(V)` → the re-add raised "already in components". Both failed at the call.
- **Same-tick self-conflict slipped through and poisoned the buffer.** `add(V)` twice → neither call saw
  the other (same committed snapshot), both queued, and `world.update()` raised mid-loop with the entity
  already popped → next frame `KeyError('component')`.

Fix: make the command buffer a **staging area** (git-index model). Fully validate every op at the call, so
only valid commands enter the buffer and `update()` is a pure, infallible apply.

## What

Bring `add_component`/`remove_component` up to the bar `add_entity` already met (it validated field data
eagerly). Two halves:

1. **Structural** — dup-add / absent-remove judged against the **projected** set = committed + this tick's
   queued adds − queued removes. Makes `add→remove→add→remove` legal; rejects `add+add` at the 2nd call.
2. **Field data** — dtype / shape / missing-required checked at the call, so no field-data error reaches
   commit.

## How

`CommandBuffer` (`command_buffer.py`) is the single gate. Validation moved into `CommandBuffer.append` —
every command, from `add_entity` / `add_component` / `remove_component`, is validated as it enters the
buffer. Entity/World mutators are thin Command-builders that call `append`; "everything in the buffer is
valid" then holds by construction.

- The buffer holds a back-reference to `World` for committed state (`_eid_to_pool_ix`, `pool_to_components`)
  and the field-metadata maps.
- Structural: `append` computes the PROJECTED component set for the command's entity by replaying its own
  contents over the committed base — `ADD_ENTITY` sets the base for uncommitted spawns, `ADD_COMPONENT` /
  `REMOVE_COMPONENT` apply. Reject dup-add / absent-remove against that projected set.
- Field data: validate the command's kwargs (dtype / shape / missing-required / bad-field-name) at `append`,
  reusing the same validator `add_entity` uses.
- `update()` stays a dumb apply — no atomicity/rollback.

## Explicit non-goal

`update()` atomicity / rollback. Commit does **not** re-validate or unwind — the guarantee is "nothing
invalid ever gets staged", not "commit recovers from invalid input".

## Resolution (DONE — 2026-07-09)

**Both halves landed. Full microecs suite green.**

`CommandBuffer.append` is the single eager gate. Every command is validated as it enters:

- **Liveness** — a non-live id raises `ValueError` (raise-over-assert; replaces the old `Entity` assert).
- **Structural** (dup-add / absent-remove) — judged against the projected set. `_get_components_state`
  replays the buffer over the base; the base is the committed pool for committed entities and the
  `ADD_ENTITY` command's `args["components"]` for same-tick spawns, so churn on an uncommitted entity works.
- **Field data** (unknown-component / bad-name / shape / dtype / missing-required) — one call to
  `world._validate_components([component], **fk)`; defaults filled by `world._defaults_for`. The old
  `_check_components_against_pool` was split into a pure `_validate_components` (raises, survives `-O`) and
  a pure `_defaults_for` (no mutation). Same validator `add_entity` uses — add/remove_component now meet the
  same bar.

`update()` stays a pure apply — commit-time re-validation removed. Duplicate-entity is a non-issue by
construction (`add_entity` mints a fresh monotonic id).

## Known exhaustiveness gap (→ task 23)

`add_component` validates the new component **in isolation**, so a component whose field name collides with
a field the entity already has slips past the gate and crashes inside `update()` on a `-O`-erasable assert.
Real fix is upstream of the gate: enforce globally-unique field names at `World` construction. Tracked in
task 23; guarded by an `xfail(strict=True)` test on this branch.
