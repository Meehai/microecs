# Formal state language for structural tests

**Created**: 2026-10-18
**Priority**: 2
**Closed**: 2026-10-18

## Why

Structural tests were written as World plumbing: create a `World`, queue mutations, call `update()`, then
inspect pools and id maps. The boolean being tested was buried under setup and commit noise, and every test
invented its own ordering for several entities.

The formal language makes the test read as a state formula:

```python
state = setup({0: [C1, C2], 1: [C1]})
state = act(state, [
    rmc(0, C2),
    addc(0, C3),
    rme(1),
    adde([C1, C3]),      # World allocates id 2
])
assert state == {0: [C1, C3], 2: [C1, C3]}
```

## What

`test/unit/formal.py` provides:

- `setup({eid: [components]}, components, data=None) -> State`: spawn, commit once, return the state.
  Eids must be contiguous `0..N-1`, matching the World's monotonic allocator; a hole asserts.
- `act(state, [actions]) -> State`: apply the ordered list to a copy, commit once, return a new state.
- `addc(eid, C, **data)` -> `entity.add_component`.
- `rmc(eid, C)` -> `entity.remove_component`.
- `adde([C...], **data)` -> `world.add_entity`. No eid: the World allocates it; the expected dict checks it.
- `rme(eid)` -> `world.remove_entity`.
- `Action` is a closed union of those four verbs. There is no base class to subclass, no generic callable, and
  `act` checks exact types, so a test cannot add a fifth verb or inject an arbitrary World mutation.

`State` is a `Mapping[int, frozenset[type[Component]]]`, not a `World` subclass. It exposes the snapshot through
normal mapping reads and compares directly with a plain dict:

```python
assert state == {0: [C1, C3], 2: [C1, C3]}
assert state[2] == frozenset({C1, C3})
assert list(state) == [0, 2]
assert dict(state) == {0: frozenset({C1, C3}), 2: frozenset({C1, C3})}
```

`State` also carries `next_id`, the id the next `adde` will allocate. Comparison with a plain dict is the
structural projection; comparison between two States is the full state, `entities` and `next_id`. The mapping
alone is not a state: after removals, equal mappings can allocate different ids.

```python
a = act(setup({0: [C1], 1: [C1], 2: [C1]}, _COMPONENTS), [rme(1)])
b = act(setup({0: [C1], 1: [C1], 2: [C1], 3: [C1]}, _COMPONENTS), [rme(1), rme(3)])
assert a != b                            # same mapping, different next_id
assert a == b.entities                   # the mapping is the structural projection
assert a.next_id == 3 and b.next_id == 4
```

`State.__eq__` normalizes the supplied mapping's values to `frozenset`; lists, tuples, and sets are all
accepted and component order is irrelevant. Dict comparison deliberately ignores `next_id`: it is a projection,
not a congruence for `adde`.

## How

- `act` deep-copies the World and returns that copy after one commit. The copy is the new state; the input
  State is never mutated. A rejected action raises `ActError`, whose `.state` is the recovered state: the
  accepted prefix is committed, and the input stays unchanged. Commit failures stay direct on `World`.
- `_copy_world` turns cached `Entity` views into `None` (they rebuild lazily against the copy) and drops the
  derived `QueryResult` cache. No World internals are otherwise rewritten.
- `act` asserts the returned World's pool/id invariants before returning it.
- `State.__init__` asserts the World's command buffer is empty. A State is always committed: pending
  commands are not a state. Pre-commit buffer inspection stays in `test_command_buffer.py`.
- Inspection beyond the component formula is explicit through `state.world`; `State` does not proxy World
  attributes.
- Field data is only used to satisfy required fields. Formal tests use dummy float32 data; object-dtype,
  dtype, shape, default, and eager-validation tests stay on the direct API.

Canonical partial-update recovery (`test_failed_act_exposes_unrelated_prefix_work`):

```python
state = setup({0: [C1], 1: [C1]}, _COMPONENTS)

with pytest.raises(ActError) as exc:
    act(state, [rme(1), rmc(0, C3)])

assert exc.value.state == {0: [C1]}
state = act(exc.value.state, [adde([C1])])
assert state == {0: [C1], 2: [C1]}
```

The accepted prefix is the state; the rejected action stages nothing; the recovered state accepts the next
transition.

## Coverage

Ported to the formal language:

- `test_world.py`: component add/remove transitions, same-tick cycles and repeats, same-tick spawns, tags,
  entity spawn/despawn, id allocation and swap-remove bookkeeping, pool layout/order/conservation, stale
  entity rejects, and the last-removal guard xfails.
- `test_command_buffer.py`: valid add/remove churn and repeats. Structural rejects and spawn pass-through
  still append raw commands because the buffer contents are the subject.
- The recovery tests cover success, failure, accepted prefixes, branches from one state, and pool-data
  isolation.

Kept direct on purpose:

- Query and query-cache behavior.
- Pool identity/reuse and realloc.
- Entity handle/cache behavior (`get_entity`, stale handles, lazy allocation).
- Command-buffer gate internals (`removed_this_tick`, partial staging).
- Data/eager validation (dtype, shape, defaults, object payloads, conversion).
- Multi-system same-tick integration and randomized id-map churn.

## Acceptance

Met at close:

- `state == {eid: [components]}` works for lists and sets, in both directions.
- `State` carries `next_id`; State-to-State equality includes it. The mapping alone is a projection, not a
  complete state, because removal history changes the next spawn id.
- Successful `act` returns a new independent state; the input state is unchanged.
- A rejected action raises `ActError`; `exc.value.state` is the recovered state and the input state is unchanged.
  A failed commit propagates raw instead of `ActError`; commit failures stay direct `World` tests.
- Staged writes are snapshotted on append; resizing the caller's array after staging cannot change the commit.
- `act` rejects generic callables and duck-typed `apply()` objects; only the four exact verb classes are accepted.
- `adde` takes no eid; expected state checks the allocated id.
- `pytest test`: 557 passed, 10 xfailed.
- `ruff check .` clean; task lint green.
