# Systems & Per-Entity Iteration

`Systems` are a **convention**, not part of the library — a system is just a callable that takes the `World` and does something to it. `World` only knows entities and components; behaviour lives at application level, wired as hooks or callbacks.

```python
class MotionSystem:
    def __call__(self, world: World):
        qr = world.query(HasPosition, HasVelocity)          # 1. select the entities you care about
        qr.position[:] = qr.position + qr.velocity * DT     # 2. operate (here: vectorized)
```

Every system has the same shape: **query, then operate**. There are three ways to operate — pick by what the work needs.

## 1. Vectorized (the default, and the whole point)

Batch the whole query in one numpy op. Two equivalent forms:

```python
qr = world.query(HasPosition, HasVelocity)
qr.position[:] = qr.position + qr.velocity * DT     # writes back to every underlying pool (numpy rules)

# per-pool variant: less ergonomic, but avoids the Field object -- maybe faster in extreme cases
for pool in qr.pool_list:
    pool.position[:] = pool.position + pool.velocity * DT
```

Push branches into numpy too — `np.where` / `np.clip` instead of a per-entity `if`:

```python
qr.velocity[:] = np.where(hit_wall, -qr.velocity, qr.velocity)      # data-parallel bounce, all entities at once
```

This is what microecs is for. Kept vectorized, a system runs at 1–2 ns/entity — see [Benchmarks](benchmarks.md).

## 2. Per-entity via `zip` (when a system can't vectorize)

Not every system vectorizes. A renderer calls a draw function per primitive; same for any per-entity foreign API. The cheapest way to iterate through all entities is `zip()` over the `QueryResult` fields you need:

```python
qr = world.query(HasPosition, HasColor, HasRadius)
for pos, color, radius in zip(qr.position, qr.color, qr.radius):    # one entity per step, fields aligned
    rl.DrawCircle(int(pos[0]), int(pos[1]), float(radius[0]), color)   # per-entity by necessity (no "draw all")
```

There is no `qr.field[i]` shortcut — the entity axis is off-limits on a query result; it raises. For random single-entity access, go by id (next section): `world.get_entity(qr.entity_ids[i])`.

## 3. The `Entity` API (a single entity, structure, foreign formats)

`world.get_entity(eid)` returns an `Entity`: an OOP-like view of **one** row. Reach for it when you address a single entity by id, when you change an entity's **structure** (add/remove a component), or when you convert to a **foreign format** (e.g. JSON). It always re-checks which pool the entity lives in, so it is slower than the vectorized path — keep it out of hot loops (see [Benchmarks](benchmarks.md)).

```python
e = world.get_entity(eid)
e.position += np.float32([1, 0])      # read/write one field -- eager, visible at once (no update() needed)
if e.is_colliding.item():             # pull a python scalar out for control flow / a reply
    ...
```

### Capabilities are additive (add / remove components live)

`has_component` / `add_component` / `remove_component` let a system grant or revoke a capability at runtime. A static object becomes a mover the moment it gains a `HasVelocity`; drop the component and it is static again — there is no `Frozen` flag to juggle, "static" simply means "lacks `HasVelocity`". Structural changes are **buffered**, so call `world.update()` before the same tick reads them:

```python
e = world.get_entity(eid)
if not e.has_component(HasVelocity):
    e.add_component(HasVelocity, velocity=np.float32([0.5, 0]))
else:
    e.remove_component(HasVelocity)
world.update()                        # commit the structural change before this tick uses it
```

This is exactly how a plugin toggles behaviour on a scene object at runtime: press a key → grant `HasVelocity` (and maybe a collider) → the universal motion system starts integrating it; press again → remove the component → it goes static.

### Per-entity serialization (`to_dict`)

Serialization is per-entity and targets a foreign format (JSON), so it is a loop over `get_entity`, not a vectorized op. `entity.to_dict(serialization_field="serializable")` dumps only the fields whose `World(extra_metadata=[...])` flag is set (so a derived, per-frame field can opt out):

```python
def world_to_dict(world: World) -> dict:
    res = {"entities": [], "components": world.component_names, "extra_metadata": world.extra_metadata}
    for eid in world.live_entities:
        res["entities"].append(world.get_entity(eid).to_dict(serialization_field="serializable"))
    return res
```

Runnable in [Example 3 — Serialization](example-3-serialization.md): **F5** saves the world to JSON, **F6** reloads it.

## Which one?

Vectorize by default. Per-entity loops are a **cliff, not a tie** — every per-entity microecs path is 15–30× slower than idiomatic float-based OOP, because microecs is numpy-backed and a per-entity step pays numpy's tiny-array overhead. One unavoidable per-entity pass costs ~500× a vectorized op and will dominate the frame. If a system is *irreducibly* per-entity (data-dependent control flow), plain python objects beat microecs there — use them. microecs is the right tool for **vectorizable** simulation. The [Benchmarks](benchmarks.md) page has the numbers.
