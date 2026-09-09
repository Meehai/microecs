# Two structural-API gaps: duplicate components at spawn, and `remove_entity` that is not idempotent

**Created**: 2026-07-26
**Closed**: 2026-07-26
**Priority**: 2

Two one-liners on the same surface — `World`'s structural API judging a *request* wrongly. One accepts a
malformed archetype; the other refuses a harmless repeat. From plan
`2-app-level-audit-and-mutation-timing.md`, findings 9 and 3. Merged into one task because either alone is
too small to file.

## Subtask 1 — reject duplicate components at spawn (**done 2026-07-26**)

### Why

`add_entity([Pos, Pos, Vel])` was accepted. `_validate_components` de-duped into a `set` for its checks, then
the loop — and `_get_entity_pool` — iterated the **list**, so `pool.fields == ['position', 'position',
'velocity']`. Verified before the fix:

- `pool.data` is a dict, so the dup collapsed to one column; the second `np.empty` was allocated and dropped.
- `pool.fields` kept the dup, so `add_entity`, `remove_entity` and `_realloc` did that column **twice, for
  every entity of the archetype, forever**.
- `to_dict()["components"] == ['Pos', 'Pos', 'Vel']` — a save/load round-trip fed the malformed list back in.
- The archetype key is a bitmask, so **the first caller's argument list decided the pool's shape**: spawn the
  dup first and every later, well-formed `add_entity([Pos, Vel])` landed in the malformed pool.
- `add_component` could not *create* it (it rejects a component the entity has) but **propagated** it.

No values were corrupted — both writes hit the same column — so this was a validation hole, not a data bug.
But it was a malformed state the library declares illegal one line above (`add_entity([])` raises).

### Done

`_validate_components` (`world.py:232`), beside the empty/unknown checks:

```python
if len(cs) != len(components):
    raise ValueError(f"Duplicate components: {components}")
```

Both spawn gates inherit it — `world.add_entity`'s pre-pass and `CommandBuffer.append`'s ADD_ENTITY branch —
so it survives #23 subtask 1 deleting the pre-pass. Green tests: `test_add_entity_rejects_duplicate_components`,
`test_rejected_duplicate_component_spawn_leaves_nothing_behind`,
`test_buffer_alone_rejects_duplicate_components`, plus `test_query_with_a_repeated_component_is_harmless`
(the raise must not leak into the read side — `query(Pos, Pos)` stays legal, the key de-dupes it).

## Subtask 2 — `remove_entity()` of a dead id is a no-op (**done 2026-07-26**)

### Why

The second `remove_entity(eid)` raises `ValueError: Entity: 0 not in live entities` — from
`CommandBuffer.append` (`command_buffer.py:69`), reached before the `del` in `world.py:87`. Same message
whether the first remove was this tick or ten ticks ago — and only the first of those two is a race.

But a kill is decided by *several* systems in one tick — damage, TTL, out-of-bounds — and two bullets hitting
one asteroid is the most common event in a game. An already-dead id is a **no-op request**, not a programming
error. Every app therefore writes the same helper: a kill set drained once per tick (`Reaper`, in the audit's
shooter). That is bookkeeping the library can delete.

And the guard the exception tells you to write is awkward on purpose: `eid in world.live_entities` works only
because `live_entities` is a public dict. Same shape of complaint as plan 2's finding 2 (buffer-blind
`has_component`).

### What

`remove_entity` on an id that is not live but **was** issued (`entity_id <= world._last_id`) returns silently
and stages nothing. An id the world **never handed out** (`> _last_id`) still raises — that is a typo, not a
race.

"Stages nothing" is the load-bearing half: two `REMOVE_ENTITY` commands for one id would blow up at commit on
`self._eid_to_pool_ix.pop(entity_id)` (`world.py:169`).

### Done

Idempotent **within a tick**, and the boundary is the point. `CommandBuffer` keeps a
`removed_this_tick` set (added in `append` on a REMOVE_ENTITY, emptied in `clear()`), and
`World.remove_entity` asks it before the `append` (`world.py:87-92`):

```python
if entity_id not in self.live_entities:
    if entity_id not in self._command_buffer.removed_this_tick:
        raise ValueError(f"Entity: {entity_id} is not in the world (stale). Either wrong id or removed earlier")
    return
```

**Why the tick and not "any dead id".** Within a tick, system order is arbitrary, so two systems killing the
same entity is a *race* — a no-op request. Across an `update()` the order is explicit, so a dead id still
being held is a **stale reference**, and swallowing it would hide the bug. "Tick" here means *since the last
`update()`*, which is the only definition the library has: an app that commits mid-frame gets the strict
answer for the rest of that frame.

That also made the rule smaller. An earlier attempt gated on `0 <= entity_id <= self._last_id` — id
arithmetic standing in for "was this ever spawned" — which needed both halves (`_last_id` starts at −1, so a
one-sided check swallowed every negative id) and still could not see the tick. The buffer already records
what happened this tick, so never-spawned and long-dead collapse into one answer: not live, nothing staged,
raise. No `_last_id` in the rule at all.

**The guard cannot live in `CommandBuffer.append`** — the first attempt put it there and it was dead code.
`append` has a blanket liveness gate at its top (`command_buffer.py:69`) that fires before the per-command
dispatch, and more fundamentally `append` can only *raise* or *stage*: it has no vocabulary for "do nothing".
So the decision belongs at the call, and the buffer only answers the question. Recorded because it is the
reason this is 5 lines in `world.py` against the "append is the single gate" direction of #23 subtask 1.

A first attempt also added a `if entity_id in self._eid_to_pool_ix` skip in `update()`'s REMOVE_ENTITY
branch. Reverted: with the staging rule correct a doubled command cannot exist, so the check only fires on a
malformed buffer — and turns a loud `KeyError` into a silent skip, the failure mode **#41** exists to
prevent. The invariant is owned by the staging layer.

Green tests: `test_remove_entity_twice_is_a_noop`, `test_remove_entity_after_the_despawn_committed_raises`
(the boundary), `test_removed_this_tick_is_dropped_by_update` and
`test_removed_this_tick_agrees_with_the_commands_it_summarizes` (the mechanism, and that the derived set
still matches the commands it summarizes), `test_remove_entity_of_an_id_the_world_never_handed_out_raises`
(999 / −1 / −5 — all the same rule now),
`test_remove_entity_of_an_uncommitted_spawn_is_removed_once_and_then_a_noop` (the no-op must not eat a legal
despawn), `test_remove_entity_accepts_the_numpy_ids_a_query_hands_out`, plus
`test_remove_unknown_entity_id_fails` unchanged.
`test_remove_entity_twice_fails_on_second_call` pinned the old behaviour and was deleted. That the tests bite
is checked by `test/manual/43-idempotent-remove/mutation_check.py` — 4 mutants (no guard, the old
looser predicate, swallow-everything, and a `removed_this_tick` that outlives the tick), 4 caught.

## Relates

- **#23** (unique field names across components) — the other duplicate-name rule, at `World` construction
  rather than at spawn. Its subtask 1 (delete `add_entity`'s redundant pre-pass) is why subtask 1's rule is
  pinned at the buffer too.
- Plan 2's finding **9b** — `remove_component` of the last component reaches `components == []`, exactly what
  `add_entity` forbids, and the repair path is guarded by an `assert` (`command_buffer.py:83`) that vanishes
  under `python -O`. Left out on purpose: it needs a decision (forbid the removal, or accept the empty
  archetype), not a check. Not filed.
- Plan 2's finding **2** (buffer-blind `has_component`) — same "the guard you cannot write" theme. Not filed.
