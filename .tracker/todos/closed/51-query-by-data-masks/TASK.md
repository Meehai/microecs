# Query by data: `qr.field[mask]` boolean-mask reads & writes (numpy parity)

**Created**: 2026-09-09
**Closed**: 2026-09-09
**Priority**: 2
**Status (2026-09-09): implemented.** `_row_level_mask` + both branches landed in `qr_field.py`;
suite green (536 passed + 8 xfail). Consumers pending: robosim law dispatch (188, masked by
`world.query(HasMotion)` → per-level `physics_level` masks → gather/scatter) and the e2e suite.

## Outcome

**Shipped 2026-09-09.** `_row_level_mask` (bool `(n,)`/`(n, 1)`/QRField masks, per-pool
mask slices) + the `QRField.__getitem__`/`__setitem__` boolean branches landed in
`qr_field.py`; reads gather one copy, writes scatter via `np.broadcast_to` before any write
(byte-identical on rejection). Single-pool `_QRArray` needed nothing (numpy free).

- named `_row_level_mask` in code, not `_coerce_mask` (What updated)
- tests: `test_/unit/test_field_bool_mask.py` + boolean-mask carve-outs in the parity file
  and the 3 stale `test_queryresult.py` pins; suite 536 passed + 8 xfail at close
- docs: primitives.md gains the *Boolean masks* section and drops `qr.f[mask]` from the
  rejected set; `_selects_axis0` + `_row_level_mask` docstrings updated
- consumers unblocked: robosim law dispatch (task 188) and the mixed-archetype e2e

## Why

robosim needs data-driven subquerying: take the entities of a query that satisfy a *field condition* and
operate on just those — batched. The driving case is the vectorised collision narrow-phase (robosim task
147): per cell, batch-test only the spheres of that cell (`collider_kind == SPHERE`), because
sphere-vs-sphere is the hot path while mixed pairs stay scalar.

Physics level ALSO takes the data route now (2026-09-09, user decision): the tag+dependencies idea
(`HasPhysicsLevel1/2` + World-level `dependencies` constraints, microecs #52) was abandoned —
"which component combinations are valid is a systems problem, not a world problem" — and level became a
`physics_level` field on `HasMotion` with law dispatch by mask (robosim #188). So this task is now
load-bearing for BOTH consumers: per-pair kind dispatch in the narrow-phase (#147) and value-partitioned
law dispatch (#188). Both want exactly numpy's mask semantics: read = gather copy, write = scatter into
the pool rows.

Current state: `_QRArray` (single-pool queries) already is an ndarray view, so `qr.field[mask]` **works
today** on it (reads gather, writes scatter into the pool). `QRField` (multi-pool queries) raises on
entity-axis boolean masks — pinned in `test_field_numpy_parity.py`'s RAISES bucket. This task is: give
`QRField` the same numpy semantics.

## Decision (2026-09-09)

**Implement it through fields** — `qr.field[mask]` (candidate B below) is the task's scope; no
filtered-`QueryResult`/`where()` machinery. The where-like **sugar** remains the long-term goal but, when
built, it must be **composition over the field masks**, not a new result type.

## Candidate solutions

### A — `QueryResult.where(mask)` (considered, not chosen)

```python
qr = world.query(HasModel, HasCollision, HasPose)
qr_spheres = qr.where(qr.collider_kind[:, 0] == ColliderKinds.SPHERE)
centers = qr_spheres.candidate_pose.numpy()[:, 0:3, 3]
radii = qr_spheres.collider_radii.numpy()[:, 0]
qr_spheres.is_colliding = hit          # write through the filter
```

Cons / feasibility:

- A filtered `where()` **is implementable** (feasibility checked 2026-09-09): it needs `QueryResult` to gain
  parent + per-pool slot indices; reads gather, but writes must scatter through the slots — or they
  **silently vanish into gathered copies** (the worst footgun), so filtered fields cannot be `_QRArray`
  views (masking destroys the contiguous-slice property that makes writes write through). It needs its own
  `len`/`entity_ids`/cache story; every consumer of `QueryResult` must then handle two flavors.
- **Zero-copy filtered views are impossible** (numpy has no row views) — a `where()` subset is always a
  gather copy, so it buys ergonomics, never speed. For zero-copy selection of a *static* class, the tool is
  archetype tags (the #50 route).
- Masks are per-tick data → children can't participate in `_qr_cache`; a second identity for results.
- It re-invents per-field numpy indexing at the result level — heavier surface, more tests, more semantics
  to keep honest. **Rejected for this iteration**; revisit only if a second consumer appears.

### B — `qr.field[mask]` (chosen direction)

```python
qr = world.query(HasModel, HasCollision, HasPose)
mask = qr.collider_kind[:, 0] == ColliderKinds.SPHERE          # (N,) bool
centers = qr.candidate_pose[mask][:, 0:3, 3]                   # gather (copy, like numpy)
radii = qr.collider_radii[mask][:, 0]
hit = (centers[:, None] - centers[None]) ** 2 < (radii[:, None] + radii[None]) ** 2
qr.is_colliding[mask] = hit                                     # scatter into the pool rows
```

Pros:

- **Exactly numpy's semantics** — the library's doctrine for field access is "whatever numpy accepts"
  (same rule as `e.field = v` / `set_data`). `arr[mask]` reads and `arr[mask] = v` writes are numpy's own
  contract; no new object, no new identity, no cache story.
- `_QRArray` already gets it for free; only `QRField` needs the bool-mask branch in `__getitem__` /
  `__setitem__`. ~30 LoC + the mask coercion helper.
- Writes land through the per-pool views (scatter) — no silent-dataloss path if implemented per-part.
- The entity-axis RAISES pin shrinks by exactly one case: *boolean masks allowed; fancy/range/partial stays
  raising* (those are the ones that cannot be made consistent across parts).

Trade-off: no named "subset result" — you re-apply the mask per field (fine: the narrow-phase needs 2–3
fields). If a named subset is ever wanted, build it as a robosim helper on top, not in the lib.

## Final goal (beyond this iteration)

Long-term, anything that looks like a `WHERE` in SQL should be expressible with where-like, ergonomic,
**efficient** syntax — and per the decision above, the route is **through fields**, never a new result type:

```python
qr = world.query(HasModel, HasCollision, HasPose)
mask = qr.collider_kind[:, 0] == ColliderKinds.SPHERE
center = qr.candidate_pose[mask][:, 0:3, 3]      # where-like read: gather
qr.is_colliding[mask] = hit                       # where-like write: scatter
```

- **Disjoint partition queries stay safe at the field level** — `qr.field[mask_pos] = v1; qr.field[mask_neg] = v2` with disjoint masks writes disjoint rows, order-independent (overlapping masks: numpy's per-row last-write-wins). No named subset object needed for the guarantee.
- A future sugar (e.g. `qr.where(lambda q: q.x >= 0)`) would be a **thin helper evaluated to a mask** — predicate → `(n,)` bool → the same per-field gather/scatter. No `len`/`entity_ids`/"behaves like qr" promise; if a named subset is ever needed, build it as a robosim helper on top.
- **Efficiency is a hard constraint**: vectorized end-to-end, no per-entity Python loop, no copies beyond the inherent gather — must not silently re-introduce the scalar `.item()` per-pair path that motivated all of this.
- The filtered-`QueryResult` design (candidate A) is the reference for what the sugar should *feel* like — re-evaluated (with its feasibility notes) only if a second consumer justifies the machinery.

## What

`pkg/microecs/microecs/qr_field.py`:

- `_row_level_mask(key)` (named `_coerce_mask` in the original design): accepts a bool ndarray of shape
  `(n,)` or `(n, 1)` (squeeze the feature axis —
  the natural comparison `qr.field[:, 0] == value` gives `(n,)`, but `qr.field == value` gives `(n, 1)`;
  support both) and a QRField mask (via `.numpy()`); raise on non-bool dtype / wrong length.
- `QRField.__getitem__` bool-mask branch: per-part gather
  `np.concatenate([part[rows_i] for ...])` — one copy, like numpy's `arr[mask]`.
- `QRField.__setitem__` bool-mask branch: per-part scatter `part[rows_i] = v_chunk` (`np.split` the value
  by per-pool row counts; scalar values broadcast like numpy). No `where()` method, no QueryResult change.

## Not this

- No `QueryResult.where()` filtered-result type — **rejected**: feasible but needs parent+slots+scatter
  machinery for ergonomics only (feasibility notes under candidate A); revisit only if a second consumer
  appears and the machinery is justified.
- No where-like sugar in *this* task — the final-goal syntax is a thin composition over the field masks;
  this task ships only the primitive.
- No entity-axis fancy/range/partial indexing — those stay RAISES (document them as such in the parity
  file). Only whole boolean masks.
- No component-per-enum tag explosion for collider kinds etc.
- No caching of masks or filtered results.

## Validation

Unit tests live in `test/unit/test_field_bool_mask.py` (TDD green-to-red 2026-09-09: 14 red = the
multi-pool mask branches; single-pool `_QRArray` cases already green) + the boolean-mask carve-outs
in `test/unit/test_field_numpy_parity.py` (both RAISES parametrizes). Exception types pinned by the
tests: non-bool dtype -> `TypeError`, wrong length -> `ValueError`, wrong value shape -> `ValueError`,
rejected writes leave the pool byte-identical.

Parity is the whole test (the `test_field_numpy_parity.py` doctrine):

- Read: `np.array_equal(qr.field[mask], qr.field.numpy()[mask])` per dtype (float32/int32/bool/object),
  single-pool **and** multi-pool; mask shape `(n,)` and `(n, 1)`; mask taken from
  `qr.field[:, 0] == value` and from another query's field comparison.
- Write: `qr.field[mask] = v` → pool bytes identical to `qr.field.numpy()[mask] = v` on a live pool;
  scalar-value broadcast `qr.field[mask] = 1.0`; `(k, ...)` values.
- Raises: non-bool mask, wrong length, fancy index `[0, 1]`, range `[0:2]` (documented carve-out), wrong
  value shape for write; nothing written on rejection (byte-identical pool, like the set_data tests).
- Existing suite green: the RAISES pins updated to name the boolean-mask exception.

## Relates

- robosim task 147 (vectorised collision narrow-phase) — consumer 1; per-cell sphere-sphere batch.
- robosim task 188 — consumer 2; `physics_level` law dispatch by mask (tags + dependencies abandoned
  2026-09-09: combination validity is a systems/domain problem, not generic-engine knowledge).
- microecs #50 — the query surface / field builders used by both consumers.
- `test_field_numpy_parity.py` — the RAISES bucket being carved out by the boolean-mask case.
