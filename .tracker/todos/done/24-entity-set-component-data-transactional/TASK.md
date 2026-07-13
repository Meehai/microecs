# Entity.set_component_data: transactional multi-field write with rollback

**Created**: 2026-07-13
**Closed**: 2026-07-13
**Priority**: 2

## Status (2026-07-13): DONE — landed on `entity.py:49`, 15 tests green (full suite 268 pass)

Signature `set_component_data(component, data: dict)`. Three bugs surfaced by the tests, all **fixed**; the
dtype question resolved as **not a bug**:

1. ~~Happy path broken — `self[k]` not subscriptable (`Entity` has no `__getitem__`).~~ **Fixed** → uses the
   attribute API (`before[k] = getattr(self, k).copy()`, `setattr(self, k, v)`).
2. ~~Inconsistent error channel — unknown-field `return`ed a `{"error": ...}` dict (robosim's shape, a
   layering leak) while bad-component raised.~~ **Fixed** → unknown-field now `raise ValueError`. One
   contract: succeed (return `None`) or raise.
3. ~~Failure/recovery raised a bare `Exception` (lost the original type, no chaining).~~ **Fixed** → both
   paths now `raise ValueError(...) from e` / `from e2` — specific type, cause preserved.

**dtype: not a bug (numpy parity).** The write is `pool.data[field][row] = value` — a numpy in-place assign
with no added validation — so float→int truncates, int64→int32 downcasts, NaN/Inf is accepted, overflow
wraps: **identical to `arr[:] = value`** (verified, parametrized parity test). No dtype guard here by design;
robosim casts to the declared dtype at its wire boundary (`ecs_data_as_np(x, dtype)`), and rejects non-finite
there (robosim #167). microecs stays a permissive, numpy-faithful setter.

Tests (`test/unit/test_entity_set_component_data.py`, 15): happy ×2, bad-component/bad-field/wrong-component
raise, empty-noop, returns-None, rollback (1 field / N>1 fields / object dtype / double-fail-surfaces-both),
+ the numpy-parity suite.

## Why

There is no way to set **several** field values on an entity as one unit. Today the only write path is
`Entity.__setattr__` (`entity.py:102`) — one field, eager (direct pool write), no atomicity. A caller that
wants to update N fields loops `e.f = v`; if the 3rd write fails (bad shape/dtype, or a runtime fault), the
first two are already committed and the entity is left half-updated.

robosim's `entity_set_data` handler needs exactly this and currently hand-rolls it inline (`server/server.py:653`
— snapshot each field into `b4`, write, restore `b4` on exception). That logic is a generic entity mutation
and belongs in microecs next to `add_component` / `remove_component`, not copy-pasted per consumer.

## What

`Entity.set_component_data(component: ComponentType, **kwargs)` — set one or more fields of `component` on
this entity, atomically-or-nothing.

**Signature** mirrors `add_component(component, **kwargs)`. Field names are unique per entity (→ task 23),
so writing by field name is unambiguous; `component` scopes/validates which fields are legal.

**Two phases:**

1. **Structural pre-check (raise BEFORE any write — nothing to roll back):**
   - entity must have `component` → else raise (`ValueError`).
   - every key in `kwargs` must be a field of `component` → else raise (`ValueError`/`AttributeError`).

2. **Transactional eager write (rollback on partial failure):**
   - for each field: snapshot `b4[k] = self.__getattr__(k).copy()`, then `self.__setattr__(k, v)`.
   - on any write-time exception: restore every already-written field from `b4`, then re-raise the original.
   - **eager**, like `__setattr__` — a direct pool write, visible immediately, **no** command buffer, **no**
     `world.update()`. (Contrast `add/remove_component`, which are buffered.)

**Double-fail is acceptable.** The rollback writes may themselves raise (defensive — normally a restored
snapshot has the right shape/dtype and cannot fail). If rollback fails, **surface both errors** — recommend
`raise rollback_err from original_err` so the caller reaches both (`exc` + `exc.__cause__`). The robosim
handler maps this to `{"error": ..., "extra_error": ...}`.

## Explicit non-goal / design note

- **The rollback IS the atomicity mechanism** — do not also fully pre-validate values. Structural pre-check
  only (component + field names). Whether to *additionally* pre-validate shape/dtype up front (via
  `world._validate_components`, as `add_component` does) is the implementer's call, with the tradeoff: pre-
  validating values makes rollback fire only on genuine runtime faults; not pre-validating makes a bad
  shape/dtype the thing rollback recovers from. Either way the **invariant is "no partial mutation on
  failure"** — that is what the tests pin.
- Not buffered; no `update()` atomicity involved (that stays a pure apply — cf. task 22 non-goal).

## Tests (tester owns — `test/unit/test_entity.py` or a focused `test_entity_set_component_data.py`)

RED until implemented — ship them `@pytest.mark.xfail(strict=True, reason="microecs #24 ...")` so the suite
stays green; when the method lands they XPASS-strict-fail, signalling "remove the markers".

- happy: single field written (eager, nothing queued); multiple fields written.
- structural reject: entity lacks `component` → raises, entity unchanged.
- structural reject: unknown field name → raises **before** any write (a sibling valid field in the same
  call is NOT written).
- **rollback** (the point): a write-time fault on the 2nd field restores the 1st to its prior value and
  re-raises; **no field is left changed** (full-snapshot restore). Inject the fault (monkeypatch
  `Entity.__setattr__` to raise on the 2nd write) so the test is robust to the pre-validation choice above.
- **double-fail**: rollback write also raises → both the original and the rollback error are reachable from
  the raised exception (walk `__cause__`/`__context__`).

## Relates

- Sits beside `add_component` / `remove_component` (`entity.py`). Uses `__getattr__` (snapshot) + `__setattr__`
  (write), `ENTITY_INTERNAL_ATTRS` (`entity.py:12`).
- Prereq for robosim **#173** `entity_set_data`; consumed via robosim's shared `ecs_data_as_np(x, dtype)`
  converter (robosim #177 follow-up) with the finiteness guard (robosim #167) in front.
- Field-name uniqueness assumption → task **23**.
