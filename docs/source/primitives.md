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
  `e.position[:] = ...`) and the vectorized `qr.field = ...` path both write straight into
  the pool buffer and are visible immediately — no `update()` needed.

So inside one tick: a freshly spawned entity is **not** visible until `update()`, but a field write on an
already-committed entity **is** visible at once. Rule of thumb: **structure is deferred, data is live.**
If a field write must be ordered against a spawn/despawn, do the structural change, call `update()`, then
write.

## `raise` vs `assert`: who made the mistake?

`python -O` deletes every `assert`. So the two are not interchangeable, and which one a check uses says who is
being blamed:

- **`raise` — bad input or bad state from outside the library.** A component definition, a ctor argument, a
  field value, an index. These are the library's contract with its caller, so they must reject under `-O` too:
  `World(extra_metadata=...)`, `Pool(fields=..., shapes=..., dtypes=...)`, `pool.remove_entity(i)`, every check
  in `_validate_component(s)` and in `CommandBuffer.append`.
- **`assert` — our own bug.** Internal bookkeeping that user input cannot reach because it was already
  validated at the call: `_pool_ids` length vs pool size, `entity_ids` count vs pool sizes, per-part shapes in
  `QRField`. Free under `-O`, which is why the hot ones (`Pool.add_entity`, per field per spawn) stay asserts.

An assert on a user-reachable path is a **bug**, not a style choice — in production the guard is simply absent
and the bad value flows on until something unrelated breaks. `test/unit/test_assert_raise_policy.py` pins both
halves: a `-O` subprocess per public rejection, and an allow-list of every remaining assert (so a new one has to
be argued for).

## How much are `Pool` and `QueryResult` numpy-like and corner cases

Given `qr=world.query(A, B)`, then `qr.position` returns a `Field`: a view over the matching pools that behaves like one contiguous-like `(N, *e)`
numpy array (e.g. `(N, 2)` for a `(2,)` field). It applies each op **per pool** and stitches result back.

That covers elementwise math and ufuncs (e.g. `np.where`, `np.linalg.norm(..., axis=1)` etc.), broadcasting (every operand shape numpy accepts, and it raises on the ones numpy rejects). See `test/unit/test_field_numpy_parity.py` for a whole set of operations comparing both.

Edge cases worth knowing:

- **Not a full ndarray — these raise, never lie.** Entity-axis *selection* of any kind
  (`qr.f[i]`, `qr.f[2:4]`, `qr.f[::2]`, `qr.f[mask]`, fancy), partial entity writes, and ndarray
  methods/attrs (`.sum()`, `.mean()`, `.dtype`, `.ndim`, `.T`). Keeping every entity is fine, so
  `qr.f[:]` and `qr.f[...]` read the whole field. Need a single entity? Use
  `world.get_entity(qr.entity_ids[i])`. Need a real array? Materialize first with `qr.f.numpy()`.
- **Axis-0 ops are per-pool, not global (footgun).** `np.sort` / `np.cumsum` / `np.sum` over
  `axis=0` run within each pool and reset at pool boundaries — they do **not** see all entities
  at once, so they differ from numpy. They're allowed, but if you want a global result, do
  `qr.f.numpy()` first. A reduction that collapses the entity axis is rejected when its length no
  longer matches the pool's row count.
- **Operands must come from the same query.** Alignment is per-pool, not by flat index, so don't
  mix a `Field` from one `world.query(...)` into an op on another.
- **Reserved field names.** A field is read back as `qr.<field>`, `entity.<field>` and `pool.<field>`,
  all three via `__getattr__` — which Python only calls *after* normal lookup fails. So a field named
  like any member of those classes is shadowed by that member and unreachable. `World(...)` rejects
  such a component at construction (a `raise`, so `python -O` keeps it) instead of silently shadowing
  it. Each class publishes its own reserved set — instance attrs plus its class dict, derived at
  import — and `World._check_components` unions the three:
  - `QUERY_RESULT_RESERVED_NAMES` (`query_result.py`): `pool_list`, `entity_ids`, `fields`, `_data`,
    `_cache`, `_field_shapes`, `_field_dtypes`
  - `ENTITY_RESERVED_NAMES` (`entity.py`): `entity_id`, `_eid_to_pool_ix`, `_pool_to_components`,
    `_world_command_buffer`, plus every public method (`get_components`, `to_dict`, …)
  - `POOL_RESERVED_NAMES` (`pool.py`): `size`, `capacity`, `fields`, `shapes`, `dtypes`, `data`,
    `fields_set`, plus every public method (`add_entity`, `pop_entity`, …) and `INITIAL_CAPACITY`

  Adding an instance attr to one of those classes means adding it to that file's private
  `_*_INTERNAL_ATTRS` — methods are picked up from the class dict automatically, attrs cannot be.
