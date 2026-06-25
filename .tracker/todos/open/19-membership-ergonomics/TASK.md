# Membership ergonomics: `eid in world` and `entity.has_component(C)`

**Created**: 2026-06-25
**Priority**: 3
**Status**: OPEN — design agreed, not yet implemented.

## Goal

Two small read-only membership checks, in the spirit of the `entity.add_component` move (object-level,
user shouldn't reach into internals). Both are pure lookups, no command buffer, no pool change.

### 1. `eid in world` — liveness check
Today checking if an entity is alive means touching a private: `eid in world.live_entities`.
Add `World.__contains__`:

```python
def __contains__(self, entity_id: EntityId) -> bool:
    return entity_id in self.live_entities
```

Natural guard before `get_entity` / `remove_entity` so callers avoid catching the assert:
```python
if eid in world:
    ...
```
"Live" follows the same definition the rest of World uses: committed OR pending-spawn this tick, minus
pending-despawn (exactly what `live_entities` tracks). So `eid in world` is True right after `add_entity`
(before `update()`) and False right after `remove_entity`.

### 2. `entity.has_component(C)` — component presence
Today: `HasVelocity in e.get_components()` (a list scan + method call). Add an explicit method on `Entity`:

```python
def has_component(self, component: ComponentType) -> bool:
    return component in self.get_components()
```

Pairs directly with the new add/remove:
```python
if not e.has_component(HasVelocity):
    e.add_component(HasVelocity, velocity=...)
```

**Explicit method, NOT `__contains__` on Entity** — `X in e` is ambiguous (component type vs field name
`"velocity"`). Grug: name it.

## Decisions / scope

- **Rejected: `entity.remove()` (despawn self).** Entity *lifecycle* is world-level (you often despawn by
  id straight from a query, never touching the object). `add_component` is the entity "acquiring"
  something — the user shouldn't care a pool migration happens. Removing the entity is a different concern.
  Keep `remove_entity` on `World`.
- **Do NOT add sugar for per-entity query iteration** (e.g. `qr.entities()`). README deliberately steers
  away from the `get_entity`-per-row path (~30× slower) toward vectorized `qr.field[:]`. Easier slow path
  fights the design — leave the friction, it's load-bearing.

## Tests (tester — to add once implemented)

- `test/unit/test_world.py`:
  - `eid in world` True after `add_entity` pre-commit, True after `update()`, False after `remove_entity`
    (eager — before the next `update()`), False for an id never minted.
- `test/unit/test_entity.py`:
  - `has_component` True/False for present/absent component; tracks an archetype migration
    (False before `add_component`+`update()`, True after; inverse for `remove_component`).

## References

- API moved in commit `f687d30` (add/remove_component on `Entity`), guard finalized eager in
  `Entity.add_component` / `remove_component`.
- Discussion: ergonomics follow-up to the `entity.add_component` change.
