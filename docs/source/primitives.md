# Primitives

microecs is five small primitives: `Component`, `Entity`, `Pool`, `QueryResult`, `World`.

- `Component` is a simple python dataclass holding only data. All entries must be numpy arrays with metadata fields: shape and dtype. We support 4 dtypes only: `int32`, `float32`, `bool` and `object`. Python strings (and any other non-numeric data) go in `dtype=object` — numpy's fixed-width strings truncate in a pre-allocated pool, so they are not supported. A component with no fields is a valid **tag** for querying (e.g. `class Frozen(Component): pass`).
- `Entity` is an `OOP-like` view inside the arrays of components. The data is column-major, so this approach is the slowest (row-major), but is sometimes needed when iterating through all the objects of some type (e.g. rendering or serialization). Its per-entity API (`add_component`, `remove_component`, `to_dict`, single-field read/write) is covered in [Systems & Per-Entity Iteration](systems.md).
- `Pool` is a simple 'archetype' dynamic array, holding entities of the same type (same set of components). Uses `Components` metadata to construct contiguous arrays for all entities of the same type. All fields of all entities of the same archetype are stored in column-major numpy arrays.
- `QueryResult` is a list of pools that match some query on all the entities of the `World`. It acts as a contiguous numpy-like container that implements numpy's interface. For all intents and purposes it should feel like a `(N, ...)` view over all selected entities. To get a proper numpy array, use `qr.numpy()`. To iterate over each entity in a query result (e.g. rendering), use `for eid, position in zip(qr.entity_ids, qr.position): ...`.
- `World` is a manager of `Pools` and has an overview of all the entities in the scene. It also manages the migration of entities from one pool to the other. A `World` can also require extra metadata keys on every field via `World(extra_metadata=["serializable"])`, to enforce component-level behavior such as field serialization.

## Few relevant concepts

- `Pool` operates on array indices, while `World` operates on entity IDs (also integers). This allows seamless movement between pools while the high-level systems still working as intended.
- All mutable operations are lazy. Entity lifecycle lives on `World` (`add_entity`, `remove_entity`); component changes live on the entity itself (`world.get_entity(eid).add_component(...)`, `.remove_component(...)`). They are added to a command buffer which is only executed when calling `world.update()`.
- `Systems` are a convention, they are not part of this library. They can be defined at application level and act as hooks or callbacks. The `World` object doesn't need to know more than entities and components. See [Systems & Per-Entity Iteration](systems.md).

## Mutation timing: field writes are eager, structural changes are deferred

One frame holds two different timings. Know which is which:

- **Structural changes are lazy (command-buffered).** `add_entity`, `remove_entity` (on `World`) and
  `add_component`, `remove_component` (on the entity, via `world.get_entity(eid)`) only queue a command;
  they take effect at the next `world.update()`. This is what
  keeps queries stable within a tick — pools don't move under a running system.
- **Field writes are eager.** A write through an `Entity` (`e.position = ...`, `e.position += ...`,
  `e.position[:] = ...`) and the vectorized `qr.field[:] = ...` path both write straight into
  the pool buffer and are visible immediately — no `update()` needed.

So inside one tick: a freshly spawned entity is **not** visible until `update()`, but a field write on an
already-committed entity **is** visible at once. Rule of thumb: **structure is deferred, data is live.**
If a field write must be ordered against a spawn/despawn, do the structural change, call `update()`, then
write.

## How much are `Pool` and `QueryResult` numpy-like and corner cases

Given `qr=world.query(A, B)`, then `qr.position` returns a `Field`: a view over the matching pools that behaves like one contiguous-like `(N, *e)`
numpy array (e.g. `(N, 2)` for a `(2,)` field). It applies each op **per pool** and stitches result back.

That covers elementwise math and ufuncs (e.g. `np.where`, `np.linalg.norm(..., axis=1)` etc.), broadcasting (every operand shape numpy accepts, and it raises on the ones numpy rejects). See `test/unit/test_field_numpy_parity.py` for a whole set of operations comparing both.

Edge cases worth knowing:

- **Not a full ndarray — these raise, never lie.** Entity-axis indexing of any kind
  (`qr.f[i]`, `qr.f[:]`, `qr.f[2:4]`, `qr.f[mask]`, fancy), partial entity writes, and ndarray
  methods/attrs (`.sum()`, `.mean()`, `.dtype`, `.ndim`, `.T`). Need a single entity? Use
  `world.get_entity(qr.entity_ids[i])`. Need a real array? Materialize first with `qr.f.numpy()`.
- **Axis-0 ops are per-pool, not global (footgun).** `np.sort` / `np.cumsum` / `np.sum` over
  `axis=0` run within each pool and reset at pool boundaries — they do **not** see all entities
  at once, so they differ from numpy. They're allowed, but if you want a global result, do
  `qr.f.numpy()` first. A reduction that collapses the entity axis is rejected when its length no
  longer matches the pool's row count.
- **Operands must come from the same query.** Alignment is per-pool, not by flat index, so don't
  mix a `Field` from one `world.query(...)` into an op on another.
- **Reserved field names.** A field is read back as `qr.<field>` and `entity.<field>`, so it may not
  collide with an attribute or method of `QueryResult` or `Entity` — `World(...)` rejects such a
  component at construction instead of silently shadowing it. The reserved set (source of truth:
  `QUERY_RESULT_INTERNAL_ATTRS` in `query_result.py`, `ENTITY_INTERNAL_ATTRS` in `entity.py`):
  - from `QueryResult`: `pool_list`, `entity_ids`, `fields`, `_data`, `_len`, `_field_shapes`, `_field_dtypes`
  - from `Entity`: `entity_id`, `get_components`, `get_fields`, `to_dict`, `_eid_to_pool_ix`, `_pool_to_components`
