# Primitives

microecs is five small primitives: `Component`, `Entity`, `Pool`, `QueryResult`, `World`.

- `Component` is a simple python dataclass holding only data. All entries must be numpy arrays with metadata fields: shape and dtype. We support 4 dtypes only: `int32`, `float32`, `bool` and `object`. Python strings (and any other non-numeric data) go in `dtype=object` — numpy's fixed-width strings truncate in a pre-allocated pool, so they are not supported. A component with no fields is a valid **tag** for querying (e.g. `class Frozen(Component): pass`).
- `Entity` is an `OOP-like` view inside the arrays of components. The data is column-major, so this approach is the slowest (row-major), but is sometimes needed when iterating through all the objects of some type (e.g. rendering or serialization). Its per-entity API (`add_component`, `remove_component`, `set_data`, `to_dict`, single-field read/write) is covered in [Systems & Per-Entity Iteration](systems.md). Reads and data writes both go straight to the pool row; only **structural** changes (add/remove a component) are buffered.
- `Pool` is a simple 'archetype' dynamic array, holding entities of the same type (same set of components). Uses `Components` metadata to construct contiguous arrays for all entities of the same type. All fields of all entities of the same archetype are stored in column-major numpy arrays.
- `QueryResult` is a list of pools that match some query on all the entities of the `World`. It acts as a contiguous numpy-like container that implements numpy's interface. For all intents and purposes it should feel like a `(N, ...)` view over all selected entities. To get a proper numpy array out of one field, use `qr.<field>.numpy()` (there is no `qr.numpy()`). To iterate over each entity in a query result (e.g. rendering), use `for eid, position in zip(qr.entity_ids, qr.position): ...`.
- `World` is a manager of `Pools` and has an overview of all the entities in the scene. It also manages the migration of entities from one pool to the other. A `World` can also require extra metadata keys on every field via `World(extra_metadata=["serializable"])`, to enforce component-level behavior such as field serialization.

## Few relevant concepts

- `Pool` operates on array indices, while `World` operates on entity IDs (also integers). This allows seamless movement between pools while the high-level systems still working as intended.
- Structural change is lazy, data is not. Entity lifecycle lives on `World` (`add_entity`, `remove_entity`) and component changes on the entity itself (`world.get_entity(eid).add_component(...)`, `.remove_component(...)`) — those go into a command buffer that is only executed when calling `world.update()`, because they move rows between pools. Writing a field (`e.field = ...`, `e.set_data(...)`, `qr.field = ...`) moves nothing, so it lands in the pool immediately. See [Mutation timing](#mutation-timing-structural-changes-are-buffered-data-writes-are-eager).
- `Systems` are a convention, they are not part of this library. They can be defined at application level and act as hooks or callbacks. The `World` object doesn't need to know more than entities and components. See [Systems & Per-Entity Iteration](systems.md).

## Mutation timing: structural changes are buffered, data writes are eager

One frame holds two different timings, and **what the write does decides which one you get** — not which
object you reached for:

| you write | timing | visible |
|---|---|---|
| `world.add_entity(...)` / `world.remove_entity(eid)` | **buffered** | after `world.update()` |
| `e.add_component(...)` / `e.remove_component(...)` | **buffered** | after `world.update()` |
| `e.field = x` / `e.field[:] = x` / `e.field += x` / `e.set_data(field=x)` | **eager** | immediately |
| `qr.field = ...` / `qr.field[:] = ...` | **eager** | immediately |

Rule of thumb: **moving a row between pools is staged; changing a number inside a row lands now.**

That line is where it is for one reason: moving a row invalidates iteration in flight, so a query cannot
stay stable within a tick unless structural change is deferred. `pool.data[f][ix] = v` moves nothing and
invalidates nothing, so there is nothing to defer — and deferring it used to cost real things (see
[what changed](#history-set_data-used-to-be-buffered)).

Consequences worth internalising:

- **One write rule for both APIs, and it is numpy's.** `e.field = x` accepts exactly what
  `qr.field = x` accepts: the value is converted to the field's dtype and broadcast into the field's
  shape. Lists, tuples and scalars are fine; a `float64` array is cast down; anything that does not fit
  raises *before* the row is touched.

  ```python
  e = world.get_entity(eid)
  e.position = np.float32([1, 0])   # exact
  e.position = [1, 0]               # list -> converted
  e.velocity = 0.0                  # scalar -> fills the row, like `qr.velocity = 0.0`
  e.position[0] = 5.0               # element
  e.position += np.float32([1, 1])  # read-modify-write, in place
  e.label = {"hp": 7}               # dtype=object field: the python object itself
  ```

  `add_entity` / `add_component` are **stricter**: they still demand an exact `np.ndarray` of the right
  dtype and shape, because that call *declares* the row rather than updating one.

- **You can read your own write, and read-modify-write composes.** Two independent contributions to one
  field in the same tick both count:

  ```python
  def damage(e, amount):
      e.health = e.health - np.float32(amount)   # or: e.health -= np.float32(amount)

  damage(e, 3); damage(e, 4)      # health drops by 7, not by 4
  ```

- **`set_data` is the transaction, not the only write path.** Reach for it when several fields (across any
  number of components) must land together: it validates every field first, then writes, so a rejected
  call writes **nothing**. A single-field `set_data(a=v)` is exactly `e.a = v`.

  ```python
  e.set_data(position=np.float32([1, 0]), velocity=np.float32([0, 0]))   # both, or neither
  ```

- **A write goes where the row is *now*.** It never consults the command buffer, which has three visible
  edges: a spawn not yet committed has no row, so writing it raises the same `AttributeError` a read does
  (spawn data belongs in `add_entity`'s kwargs); a field whose component is only *pending* has no column
  yet, so writing it raises until `update()`; and a field whose component is pending *removal* still has
  its column, so the write lands and then goes away with the component.

- **Do not hold a row view across `update()`.** `e.field` is a live view into pool memory, so a view
  stashed across an `update()` writes into whatever row now sits at that index — the entity may have
  migrated or been swap-removed. Same rule a `QueryResult` already has. Re-read after `update()`.

- **Data staged by `add_entity` is held by reference, not snapshotted.** `add_entity(position=arr)`
  remembers *your array*, and `update()` reads whatever it holds at that moment. So the idiomatic numpy
  scratch-buffer loop is **wrong** here:

  ```python
  scratch = np.zeros(2, "float32")
  for i in range(3):
      scratch[:] = i
      world.add_entity([HasPosition], position=scratch)   # WRONG: all three share `scratch`
  world.update()                                           # every entity commits the LAST value
  ```

  Pass a fresh array per call (`np.float32([i, i])`), or `scratch.copy()`. Data *writes* have no such
  trap — they copy into the row at the call.

- **Structural order within a tick is preserved.** The buffer replays in call order at `update()`, so
  `add_component(B)` then `remove_component(B)` nets out correctly. Data writes are not in that buffer at
  all: they have already happened by the time `update()` runs.

- **A repeat despawn is a no-op within the tick; a malformed spawn is never.** `remove_entity(eid)` on an
  id something already killed **since the last `update()`** returns silently and stages nothing. A kill
  decided by three systems at once (damage, a timer, out of bounds) is a normal event, not a programming
  error, so you do not need a kill set to deduplicate it. One `update()` later the same call **raises**: the
  tick is over, system order is no longer arbitrary, and an id you are still holding is a stale reference,
  not a race. An id the world never handed out raises for the same reason — with no id arithmetic, "never
  spawned" and "died three ticks ago" are one answer. The spawn side is the opposite in every case:
  `add_entity([Pos, Pos])` raises, because a duplicated component would build a pool with a duplicated
  field.

### History: `set_data` used to be buffered

Until v0.4.x the line was drawn at *which object you hold* — everything through an `Entity` was staged
(`e.field = x` raised, and a read handed back a read-only view), while `qr.field = ...` was eager. Half of
that was paid for and none of it collected: `update()` is deliberately not atomic and staged values are
references, not copies, so deferral bought no snapshot semantics — while it cost read-modify-write
(`damage(3); damage(4)` left only the last one), cost the ability to read your own write, and put an
`isinstance` + `setflags(write=False)` on every single field read. Data writes are eager now, and
`set_data` kept only the part that was worth keeping: the multi-field transaction.

## `raise` vs `assert`: who made the mistake?

`python -O` deletes every `assert`. So the two are not interchangeable, and which one a check uses says who is
being blamed:

- **`raise` — bad input or bad state from outside the library.** A component definition, a ctor argument, a
  field value, an index. These are the library's contract with its caller, so they must reject under `-O` too:
  `World(extra_metadata=...)`, `Pool(fields=..., shapes=..., dtypes=...)`, `pool.remove_entity(i)`, every check
  in `_validate_component(s)` and in `CommandBuffer.append`.
- **`assert` — our own bug.** Internal bookkeeping that user input cannot reach because it was already
  validated at the call: `_pool_ids` length vs pool size, `entity_ids` count vs pool sizes.
  Free under `-O`, which is why the hot ones (`Pool.add_entity`, per field per spawn) stay asserts.

An assert on a user-reachable path is a **bug**, not a style choice — in production the guard is simply absent
and the bad value flows on until something unrelated breaks.

> **Known gap.** Two guards in `qr_field.py` are still asserts but sit on a *user*-reachable path:
> the axis-0 length check in `_apply_fn_on_parts` (`qr_field.py:44`) and the `out=` check in
> `__array_ufunc__` (`qr_field.py:73`). Both fire on ordinary user code — e.g.
> `np.sum(qr.position, axis=0)` on a multi-pool query. Under `python -O` they vanish and the call
> returns a malformed `QRField` instead of raising. They should be `raise`.

## Lifetimes: how long is a `QueryResult` good for?

**One tick.** A `QueryResult` snapshots a slice of each matching pool at construction
(`query_result.py:32`). `world.update()` may pop-swap rows, delete an emptied pool, or reallocate a
pool's buffer outright — none of which the held object knows about.

```python
qr = world.query(HasPosition)
world.add_entity([HasPosition], position=...)
world.update()              # pool may have reallocated -> qr now points at a freed buffer
qr.position[:] = 5.0        # silently writes nowhere. No error.
```

`world.query(...)` is cached and free to call, so **re-query after every `update()`** rather than
holding one across the boundary. Two related facts: the cache means two `query(A)` calls in the same
tick return the *same object* (callers share it), and `update()` drops the cache only when the command
buffer was non-empty.

> **Known gap.** A stale `QueryResult` is not detected — reads return old data and writes are
> discarded, both silently. Tracked as microecs task 27.

## `object` dtype: you are storing references

A field declared `dtype="object"` holds real Python objects in a numpy object array (that's how you
carry a socket, a texture handle, a dict). Two consequences that numeric fields don't have:

- **A `default=` on an `object` field is shared by every entity.** Defaults are copied with
  `ndarray.copy()`, which is shallow, so the *contained* object is the same one for all of them.
  Use `default=None` and set the value explicitly per entity, or accept the sharing knowingly.
- **Removing an entity does not drop its reference.** `Pool.remove_entity` pop-swaps the live rows but
  leaves the vacated slot pointing at the removed entity's object, so it stays alive in the pool's
  spare capacity until something overwrites it. For plain data that's invisible; for GPU handles,
  sockets or file objects it is a leak. Clear the field before removing the entity if the resource
  matters.

## How much are `Pool` and `QueryResult` numpy-like and corner cases

Given `qr=world.query(A, B)`, then `qr.position` returns a `Field`: a view over the matching pools that behaves like one contiguous-like `(N, *e)`
numpy array (e.g. `(N, 2)` for a `(2,)` field). It applies each op **per pool** and stitches result back.

That covers elementwise math and ufuncs (e.g. `np.where`, `np.linalg.norm(..., axis=1)` etc.), broadcasting (every operand shape numpy accepts, and it raises on the ones numpy rejects). See `test/unit/test_field_numpy_parity.py` for a whole set of operations comparing both.

Edge cases worth knowing:

- **Not a full ndarray — the contract.** Entity-axis *selection* of any kind
  (`qr.f[i]`, `qr.f[2:4]`, `qr.f[::2]`, `qr.f[mask]`, fancy), partial entity writes, and ndarray
  methods/attrs (`.sum()`, `.mean()`, `.dtype`, `.ndim`, `.T`) are **not** part of the contract.
  Keeping every entity is fine, so `qr.f[:]` and `qr.f[...]` read the whole field. Need a single
  entity? Use `world.get_entity(qr.entity_ids[i])`. Need a real array? Materialize first with
  `qr.f.numpy()` (note: `<field>.numpy()`, there is no `qr.numpy()`).
- **Axis-0 ops are per-pool, not global (footgun).** `np.sort` / `np.cumsum` / `np.argsort` /
  `np.diff` over `axis=0` run within each pool and reset at pool boundaries — they do **not** see all
  entities at once. They keep the right *length*, so nothing rejects them, and the numbers are
  quietly wrong across a multi-pool query. If you want a global result, do `qr.f.numpy()` first.
- **Operands must come from the same query.** Alignment is per-pool, not by flat index, so don't
  mix a `Field` from one `world.query(...)` into an op on another.

> **Known gap — the same code can behave differently depending on how many archetypes matched.**
> `qr.f` is not one type. When **exactly one pool** matches, `QueryResult` returns `_QRArray`, a thin
> `np.ndarray` subclass, and *the whole numpy API is live*. When **two or more** match, it returns
> `QRField`, the shim, which enforces the contract above. So today:
>
> | expression | 1 matching pool | ≥2 matching pools |
> |---|---|---|
> | `qr.f[0]`, `qr.f[::2]`, `qr.f[mask]` | works | `TypeError` |
> | `qr.f[0] = x` (single-entity write) | works | `TypeError` |
> | `qr.f.sum()`, `qr.f.dtype` | works | `AttributeError` |
> | `np.sort(qr.f, axis=0)` | correct | silently sorted **per pool** |
> | `np.concatenate([qr.f, qr.f])` | correct | `RecursionError` |
>
> Two things follow. First, a call site that works today starts raising the moment an unrelated spawn
> elsewhere creates a second archetype — so **write to the contract, not to what happens to run**.
> Second, `qr.f[0] = x` indexes the entity axis, which is exactly what has no meaning across pools — the
> timing is fine (a data write is eager either way), the *addressing* is not. For one entity go by id:
> `world.get_entity(eid).f = x`.
- **Reserved field names.** A field is read back as `qr.<field>`, `entity.<field>` and `pool.<field>`,
  all three via `__getattr__` — which Python only calls *after* normal lookup fails. So a field named
  like any member of those classes is shadowed by that member and unreachable. `World(...)` rejects
  such a component at construction (a `raise`, so `python -O` keeps it) instead of silently shadowing
  it. Each class publishes its own reserved set — instance attrs plus its class dict, derived at
  import — and `World._check_components` unions the three:
  - `QUERY_RESULT_RESERVED_NAMES` (`query_result.py`): `pool_list`, `entity_ids`, `fields`, `_data`,
    `_cache`, `_field_shapes`, `_field_dtypes`
  - `ENTITY_RESERVED_NAMES` (`entity.py`): `entity_id`, `_eid_to_pool_ix`, `_pool_to_components`,
    `_world_command_buffer`, plus every method, private ones included (`get_components`, `to_dict`,
    `_locate`, …)
  - `POOL_RESERVED_NAMES` (`pool.py`): `size`, `capacity`, `fields`, `shapes`, `dtypes`, `data`,
    `fields_set`, plus every public method (`add_entity`, `pop_entity`, …) and `INITIAL_CAPACITY`

  Adding an instance attr to one of those classes means adding it to that file's private
  `_*_INTERNAL_ATTRS` — methods are picked up from the class dict automatically, attrs cannot be.
