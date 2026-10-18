# `remove_component` must not leave an entity with zero components

**Created**: 2026-10-18
**Priority**: 2
**Closed**: 2026-10-18

## Why

`add_entity([])` is refused, but `remove_component` of the last component reaches the same state anyway:
`_do_remove_component` builds `new_components=[]`, `_add_to_pool` creates a pool with no fields, and after
`update()` the entity is live with `components=[]`. It is invisible to every typed query, visible to the
no-arg query, and serializes as `{"components": [], "data": {}}`. The library thus declares one state
illegal on the spawn path and creates it on the removal path. Plan finding 9b; also the open follow-up in
`#2`.

Decision for now: empty entities cannot exist. The removal that would empty an entity raises, and the
caller despawns with `remove_entity()`. Auto-despawn is not chosen: it hides a lifecycle change inside a
component verb, and it collides with `#53` because the first removal would take the eid out of
`live_entities`, making the second removal a stale-id error instead of a no-op.

## What

- `remove_component(C)` raises `ValueError` when `C` is the entity's only projected component.
- The check uses the projected set: committed components plus this tick's staged adds and removes.
  `add V -> remove P` on a `{P}` entity stays legal because `V` survives.
- A refused removal stages nothing and leaves the entity untouched.
- `add_entity([])` keeps raising. Together the invariant is: a live entity always has at least one
  component, so the empty pool and empty entity are unreachable through the public API.

## How

- In `CommandBuffer.append`'s `REMOVE_COMPONENT` branch, after the `False` repeat no-op and the
  absent-component reject, build the projected set: start from `_get_entity_components(eid)`, replay this
  entity's staged `ADD_COMPONENT` / `REMOVE_COMPONENT` commands in order, then discard `C`.
- Raise `ValueError(f"Cannot remove the last component; use world.remove_entity({eid})")` when that set is
  empty. Do not stage the command and keep the buffer untouched.
- Reuse the same base helper and buffer scan as the other component checks; no new buffer state.
- Component unit tests are state transitions:
  `state = setup({eid: [C1, C2]})`,
  `state = act(state, [addc(eid, C2), rmc(eid, C1)])`,
  `assert state == {eid: [C2]}`.
  `act` applies actions to a copy, commits once, and returns that copy. If an action raises, `ActError.state`
  is the recovered state with the accepted prefix committed; the input state stays unchanged.
  - `test_world.py`: migration, same-tick cycles, eager validation, the last-removal guard, same-tick
    spawns, tags, and stale-entity rejects.
  - `test_command_buffer.py`: the append gate for staging, rejects, repeat no-op, and data conversion.
  The guard rejections are strict xfails until the guard lands; remove the markers once they XPASS.
- Keep the staged-data-survivor transition green: a staged add is a survivor.
- Add the last-removal rule to the `remove_component` row in `docs/source/primitives.md`.

## Acceptance

- Removing the last projected component raises `ValueError`, stages nothing, and leaves the component on
  the entity.
- Two removals that together empty the entity: the first stages, the second raises.
- `add V -> remove P` on `{P}` commits to `{V}`.
- `add_entity([])` still raises.
- A rejected action mid-`act` raises `ActError` carrying the accepted prefix; the input state is unchanged.
- All guard xfails XPASS and their markers are removed.
- `pytest pkg/microecs` green, task lint green.
