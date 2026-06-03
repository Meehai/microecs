# Code review — `World.get_entity(entity_id)`

**Reviewer**: EM  **Date**: 2026-06-03
**Verdict**: 🔴 Request changes — two bugs make the method unusable as written.
**Update (2026-06-03)**: ✅ Resolved. Both blockers + the minor `assert` are fixed; all 5
`get_entity` tests pass, full suite 66 green. Findings below kept for the record.

## What's under review

```python
def get_entity(self, entity_id: EntityId) -> tuple[EntityData, list[type[Component]]]:
    """Gets the entity (data) and its components (list of types) given an entity id. Used for 'object-like' ops"""
    pool, pool_ix = self._eid_to_pool_ix.pop(entity_id)
    entity = {k: pool.data[pool_ix] for k in pool.fields}
    components = self.pool_to_components[pool]
    return entity, components
```

Verified by running it (see `test/unit/test_world.py::test_get_entity_*`, currently red).

## 🔴 Blocker 1 — wrong indexing: `KeyError` on every call

`pool.data` is keyed by **field name** (`dict[str, np.ndarray]`), not by row index. So
`pool.data[pool_ix]` looks up the integer row as a dict key and raises `KeyError`.

```
>  entity = {k: pool.data[pool_ix] for k in pool.fields}
E  KeyError: 0
```

The field name `k` is built but never used to index. Fix:

```python
entity = {k: pool.data[k][pool_ix] for k in pool.fields}
```

(This is the same shape `Pool.pop_entity` builds: `{field: data[field][index]}`.)

## 🔴 Blocker 2 — a *read* mutates id bookkeeping (`.pop`)

`self._eid_to_pool_ix.pop(entity_id)` **removes** the entity from the lookup map. After a
single `get_entity`, the id no longer resolves — `add_component` / `remove_component` /
`remove_entity` on it will fail, and a second `get_entity` raises. A getter must not consume
its argument. This looks copy-pasted from `_pop_from_pool` (which legitimately pops).

```python
pool, pool_ix = self._eid_to_pool_ix[entity_id]   # index, do not pop
```

Note it only pops `_eid_to_pool_ix` (not `_pool_ix_to_eid` / `_live_ids`), so it doesn't even
leave consistent state — it half-deletes the entity.

## 🟡 Minor — unknown id raises a bare `KeyError`

Every other id-taking method asserts `entity_id in self._live_ids` and raises a clear
`AssertionError` with a message. `get_entity` has no such guard, so an unknown id surfaces a
bare `KeyError`. For consistency:

```python
assert entity_id in self._eid_to_pool_ix, f"Entity id {entity_id} not live/committed: {self._eid_to_pool_ix.keys()}"
```

(The test accepts either error today, but the assert is the house style.)

## 🟡 Minor — uncommitted spawns won't resolve

`get_entity` reads `_eid_to_pool_ix`, which is only populated at `update()`. An id minted this
tick but not yet committed is in `_live_ids` + the command buffer, not the map — so it raises.
Probably acceptable ("read committed state only"), but call it out in the docstring so callers
know they must `update()` first. This is the open design call from the task; fine to defer.

## 🟢 Good

- Signature `(EntityData, list[type[Component]])` is the right shape and reuses the new
  `EntityData` alias — consistent with `_pop_from_pool`.
- Returning components alongside the data is handy for the "object-like" pop→re-add flows.

## Suggested fix (both blockers)

```python
def get_entity(self, entity_id: EntityId) -> tuple[EntityData, list[type[Component]]]:
    assert entity_id in self._eid_to_pool_ix, f"Entity id {entity_id} not live/committed"
    pool, pool_ix = self._eid_to_pool_ix[entity_id]
    entity = {k: pool.data[k][pool_ix] for k in pool.fields}
    return entity, self.pool_to_components[pool]
```

Verified: with this body all 5 `get_entity` tests pass; with the current body 4 of 5 fail
(the 5th only "passes" because the `KeyError` happens to match the unknown-id expectation).

## Note on copy vs view

This returns **views** into the SoA arrays (no `.copy()`), unlike `pop_entity`. That makes the
result mutable-in-place but it goes stale after the next swap-remove. Either choice is fine —
just decide deliberately and document it. The tests compare by value, so they don't constrain it.
