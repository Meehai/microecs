# Membership ergonomics: `entity.has_component(C)`

**Created**: 2026-06-25
**Completed**: 2026-07-05
**Priority**: 3
**Status**: DONE — `has_component` shipped; `eid in world` rejected.

## Shipped: `entity.has_component(C)`

Explicit component-presence check on `Entity` (object-level, no reaching into internals):

```python
def has_component(self, component: ComponentType) -> bool:
    return component in self.get_components()
```

Enables the acquire-if-missing pattern (used by the house manual-move plugin):
```python
if not e.has_component(HasVelocity6DoF):
    e.add_component(HasVelocity6DoF, ...)
```

Explicit method, **NOT** `__contains__` on `Entity` — `X in e` is ambiguous (component type vs field name).
Grug: name it.

## Rejected: `eid in world` (`World.__contains__`)

Dropped. "world contains an entity id" doesn't really make sense as an API — liveness is a world-internal
concern, not a membership relation callers should lean on. No `World.__contains__`.

## Tests (tester — to add)

- `test/unit/test_entity.py`: `has_component` True/False for present/absent component; tracks an archetype
  migration (False before `add_component`+`update()`, True after; inverse for `remove_component`).
