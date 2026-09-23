# World-level invariants

**Created**: 2026-09-18
**Priority**: 3
**Closed**: n/a

**What**

We may want to preserve some global invariants, such as maximum n entities or all entities must have some specific value on some components etc. These are called on add_entity, remove_entity etc.

Syntax (TBD):

```python
world = World(..., invariants=lambda world, ent: entity.radius in (0, 1, 2))
```
