# Custom field metadata — `World(extra_field_metadata=[...])`

**Created**: 2026-06-05
**Priority**: 3
**Status**: ✅ Done (2026-06-05) — `extra_field_metadata` kwarg, strict `==` semantics; 67 tests green.

## What landed

`World.__init__` takes `extra_field_metadata: list[str] | None = None` (`world.py:18-19`;
`self.extra_field_metadata = extra_field_metadata or []`). `_check_components` (`world.py:187`) now
requires each field's metadata to equal **exactly** `{"shape", "dtype", *extra_field_metadata}`.

- **Strict (`==`) chosen over superset.** A field must carry exactly shape+dtype+declared-extras, no
  undeclared keys: a plain world rejects a component that declares `serializable`; a serializing world
  rejects one that omits it. Keeps the anti-typo guarantee. (Superset `>=` was the alternative — would
  let a plain world tolerate extra declared keys; not taken.)
- **Back-compat.** Default `None`/`[]` → required set is `{"shape","dtype"}` → unchanged; full suite green.
- **Bug found in review, fixed by dev.** First cut wrote `or None`, so `*self.extra_field_metadata`
  (`world.py:187`) did `*None` → `TypeError` on *every* world construction. Now `or []`.
- **Exposure map not built.** The design's optional `component_to_field_metadata` mirror was deferred —
  a consumer can read `C.__dataclass_fields__[f].metadata[key]` directly until a serialization system
  actually needs it.

Test (tester): `test/unit/test_world.py::test_extra_field_metadata_required_strictly_on_every_field` —
2 components × 2 worlds = 4 cases, 2 of them `pytest.raises` (missing key / undeclared key).
README: the `Component` bullet now documents `World(extra_field_metadata=[...])`.

## Why

Today every component field must carry **exactly** `{"shape", "dtype"}` — no more, no less. That is
pinned by one strict equality (`world.py:181`):

```python
assert _field.metadata.keys() == {"shape", "dtype"}, _field.metadata
```

So a field that also wants to say "this is serializable" (or "unit: meters", "net-replicated", …)
is **rejected at world creation**. There is no way to attach custom, world-known metadata to a field.

A serialization system is the concrete motivator: it needs to know *which* fields to persist. The
flag lives naturally on the field (`metadata={"shape": ..., "dtype": ..., "serializable": True}`),
but the world won't accept it. This task lets a world **declare** the extra metadata keys it requires,
turning the strict check from "forbid extras" into "require exactly shape, dtype, and the declared
extras". Opt-in: declare nothing and behavior is byte-for-byte unchanged.

```python
world = World(components=[...], extra_field_metadata=["serializable"])
# => every field of every component must now define "serializable" alongside shape+dtype
```

## Design (dev's call — non-test file)

### 1. The kwarg (`world.py:13`)

```python
def __init__(self, components: list[ComponentType], extra_field_metadata: list[str] | None = None):
    self.extra_field_metadata = extra_field_metadata or []   # MUST be set BEFORE _check_components
    self._check_components(components)                        # (world.py:14) — it reads the line above
    ...
```

Order matters: `_check_components` runs first thing in `__init__` (`world.py:14`), so
`self.extra_field_metadata` has to be assigned before that call.

### 2. The check (`world.py:181`) — the heart of the task

```python
required = {"shape", "dtype", *self.extra_field_metadata}
...
assert _field.metadata.keys() == required, f"{component}/{field_name}: {_field.metadata.keys()} != {required}"
```

Still an **exact-set** equality, just a bigger required set. This keeps the existing anti-typo
guarantee (a stray `"shpae"` or an undeclared `"foo"` key still fails) — the strict contract is
*extended*, not loosened. `extra_field_metadata=[]` → `required == {"shape","dtype"}` → identical to
today.

### 3. Exposing the values (optional, recommended)

Validation alone only guarantees the key is *present*. A consumer still has to read the value. Two
honest options:

- **No new state (minimal).** The value already lives on the dataclass field — a serialization system
  reads `C.__dataclass_fields__[f].metadata["serializable"]` directly. The world's only job was to
  *guarantee it's there*. This is the grug-minimal version and may be all we need.
- **Mirror the existing maps (convenience).** Add one map symmetric to `component_to_shapes` /
  `component_to_dtypes` (`world.py:21-24`):

  ```python
  self.component_to_field_metadata: dict[type, dict[str, list]] = {
      t: {meta: [f.metadata[meta] for f in t.__dataclass_fields__.values()]
          for meta in self.extra_field_metadata}
      for t in components}
  # read: world.component_to_field_metadata[HasPose]["serializable"] -> [True, ...] in field order
  ```

Pick one when a real consumer exists; don't build the map speculatively if direct field access is
enough. Either way the **validation** half (steps 1–2) is the deliverable.

### The one decision to make explicit

**Are undeclared extra keys allowed?** Recommendation: **no** (exact equality, as above) — it preserves
the current "metadata is exactly what's declared, typos fail loud" property and makes
`extra_field_metadata` the single source of truth for what metadata a world understands. If the dev
instead wants ad-hoc per-field keys, switch `==` to `>=` (superset) — but that re-opens silent typos.
The tests below pin whichever is chosen; default is strict.

### Note on value types

The world checks **presence only**, not the value. `"shape"` is asserted `tuple`, `"dtype"` asserted
str-in-set (`world.py:182-183`) because the world *uses* them; a custom key like `"serializable"` can
hold anything (`True`, `"v2"`, a callable) — validating its value is the **consumer's** job, not the
world's. Keep it that way.

## Done when

- `World(components)` and `World(components, extra_field_metadata=None)` / `[]` behave **identically to
  today** — every existing component (shape+dtype only), example, and test still passes unchanged.
- `World(components, extra_field_metadata=["serializable"])` constructs iff **every field of every
  component** declares `"serializable"`; a single field missing it raises a clear error **at world
  construction** (not lazily).
- Multiple extras work: `extra_field_metadata=["serializable", "unit"]` requires both on every field.
- Undeclared extra keys behave per the chosen semantics (default: rejected, preserving the strict
  contract). The error names the offending component/field.
- The extra value is **not** type-validated by the world (any value is accepted as long as the key
  is present).
- Field-less **tag** components (task 9) stay valid under any `extra_field_metadata` — no fields means
  nothing to require (vacuously true).
- If the exposure map (design §3) is built, `world.component_to_field_metadata[C][key]` returns the
  per-field values in field order.

## Tests (tester writes, `test/unit/test_world.py`)

- `test_extra_field_metadata_default_unchanged` — `None` and `[]` both equal the no-kwarg world;
  shape+dtype-only components construct fine (back-compat guard).
- `test_extra_field_metadata_required_on_every_field` — component whose field has
  `{"shape","dtype","serializable"}` constructs under `extra_field_metadata=["serializable"]`.
- `test_extra_field_metadata_missing_raises_at_construction` — same world, a component whose field
  omits `"serializable"` → raises in `World(...)` (eager, names the field).
- `test_extra_field_metadata_partial_on_multifield_raises` — two-field component, one field has the
  key and one doesn't → raises (all-or-nothing).
- `test_extra_field_metadata_multiple_keys` — `["serializable","unit"]`; both required on every field.
- `test_extra_field_metadata_undeclared_key_rejected` — field carries an extra key the world did **not**
  declare → rejected (pins strict semantics; flip/drop if dev picks superset).
- `test_extra_field_metadata_value_not_validated` — value can be `True`, `False`, a str, an object;
  world accepts any (presence-only check).
- `test_extra_field_metadata_with_tag_component` — field-less tag + non-empty `extra_field_metadata`
  constructs (no fields → nothing required). Composes with task 9.
- `test_extra_field_metadata_exposed_for_reading` — **only if** design §3 map is built;
  `world.component_to_field_metadata[C]["serializable"]` is the per-field list, field order.

## Out of scope

- **A serialization system itself.** This task only makes the metadata *declarable and required* so a
  serializer can rely on it. Persisting/loading entities is a separate task.
- **Validating extra-metadata values.** World checks presence; the consumer validates meaning.
- **Per-component (vs per-world) required metadata.** One world-wide required set, mirroring how
  shape/dtype are universally required. Per-component opt-in is a bigger design, not this.
- **Default/inherited metadata** (e.g. "serializable defaults to True if absent"). Explicit-only here;
  the field must declare every required key.

## Related

- `world.py:13` `__init__` (add the kwarg); `world.py:14` the early `_check_components` call (ordering);
  `world.py:172-184` `_check_components`; **`world.py:181`** the strict `metadata.keys()` check (the
  one line that changes); `world.py:21-24` `component_to_shapes`/`component_to_dtypes` (the pattern to
  mirror if exposing values); `world.py:182-183` shape/dtype value validation (why custom keys get
  none).
- Only callers of `.metadata` are `world.py:22,24,181-183` — nothing else in `microecs/` reads it, so
  the blast radius is one method.
- README:13 ("metadata fields: shape and dtype") — update to mention world-declared extra metadata
  once this lands.
- `.tracker/plans/1-comparison-with-other-projects.md`:22 (metadata "leaks" as an ergonomic tax — note
  this stays opt-in, so the **default** ergonomics are unchanged) and :117 (snecs has serialization,
  which this enables).
- Composes with [task 9](../9-tag-components/TASK.md): tags have no fields, so they're vacuously valid.
