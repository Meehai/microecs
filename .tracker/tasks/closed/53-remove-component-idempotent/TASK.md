# `remove_component` twice in one tick should be a no-op

**Created**: 2026-10-18
**Priority**: 2
**Closed**: 2026-10-18

## Why

`remove_entity` is idempotent within a tick: a repeat kill is a no-op request, because several systems can decide it. `remove_component` has the same request shape: no payload, target state is "component absent". Today the second call raises (`command_buffer.py:107`), so every app that expires a capability carries the same guard workaround. The asymmetry is not principled; it is the half `#43` did not cover.

Staged adds stay strict: the second add carries data, so a repeat is a conflict, not a request. That distinction is documented in the operations table in `docs/source/primitives.md`.

## What

- The first `remove_component(C)` on an entity with `C` in the projected component set stages the removal.
- A second `remove_component(C)` in the same tick, while that removal is pending, returns and stages nothing.
- `remove_component(C)` when `C` is absent and no removal is pending still raises: typo or stale state, not a race.
- After `update()`, removing an already-removed component raises. Same tick-boundary rule as `remove_entity`.
- Add/remove cycles keep working: `add -> remove -> add -> remove` commits to the final set.

Assumes `#54`: a live entity always keeps at least one component. No test here removes the last one; that
guard lands in `#54`.

## How

- Change `CommandBuffer._entity_has_buffered_component` into a newest-decision query `_get_component_in_buffer_state`: scan the buffer right to left and return `True` if the newest staged op for this entity and component added it, `False` if it removed it, and `None` if this tick staged neither. Drop the `existing_components` parameter and the committed-set fallback; the caller already holds the base set.
- In the `REMOVE_COMPONENT` branch of `CommandBuffer.append`, read the three states:
  - `False`: the removal is already pending, return before staging; a repeat request is a no-op.
  - `None`: reject when the component is absent from the base set (`_get_entity_components`: pool components, or spawn args for an entity spawned this tick); otherwise stage.
  - `True`: the component was re-added this tick, so stage the removal.
- Mirror the same three-way read in the `ADD_COMPONENT` branch: `True` rejects, `False` allows the re-add, `None` falls back to the base set.
- Add no buffer state: the pending removal is already the newest `REMOVE_COMPONENT` command.
- Tests: add the no-op case; replace `test_command_buffer.py::test_buffer_rejects_double_remove_at_second_append` and `test_world.py::test_remove_staged_component_twice_raises_eagerly`; remove the strict xfail marker from `test_world.py::test_remove_component_twice_is_a_noop`; add `add -> remove -> remove` where the entity never had the component committed. The empty-entity invariant tests belong to `#54`.
- Keep `test_buffer_rejects_remove_of_absent_component` unchanged: committed-absent removal must still raise.

## Acceptance

- `remove_component` twice in one tick stages exactly one command; `update()` commits the smaller archetype.
- Committed-absent removal still raises.
- `add -> remove -> add` and `remove -> add -> remove` still commit correctly.
- The strict xfail for the double removal XPASSes and its marker is removed.
- `pytest pkg/microecs` green, task lint green.
