# Serialization (save & load)

[`examples/03-serialization.py`](examples/03-serialization.py) saves the whole world to JSON and loads it back. Run it, then press **F5** to write `state.json` and **F6** to reload it:

```bash
python examples/03-serialization.py --n_objects 10
```

The new idea is **per-field serialization control**, wired through a `World` extra-metadata key.

## Opt in with `extra_metadata`

Ask the `World` to require a `serializable` flag on every field, then each component declares it:

```python
class HasMotion2D(Component):
    velocity:  np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "serializable": True})
    magnitude: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "serializable": False})  # derived
class HasCustom(Component):   # a tag: no fields, just marks an entity
    pass

world = World(components=[...], extra_metadata=["serializable"])   # every field must now carry "serializable"
```

`magnitude` is a **derived** value (recomputed each frame from `velocity`, via `qr.magnitude = np.linalg.norm(qr.velocity, axis=1, keepdims=True)`), so it is flagged `serializable=False` — it never hits the save file and is rebuilt on load. That is the payoff of putting the flag on the *field*, not the component.

## Dump each entity with `to_dict`

Serialization is per-entity (it targets JSON), so it loops over `get_entity` — `to_dict(serialization_field="serializable")` emits only the flagged fields:

```python
def world_to_dict(world) -> dict:
    return {"components": world.component_names, "extra_metadata": world.extra_metadata,
            "entities": [world.get_entity(eid).to_dict(serialization_field="serializable")
                         for eid in world.live_entities]}
```

Loading is the inverse: rebuild the component set from the saved names, then `add_entity` for each row. See [Systems — per-entity serialization](systems.md) for the pattern.

## Variable archetypes round-trip cleanly

Not every ball has the same components — velocity is optional (some are static) and `HasCustom` is a random tag:

```python
components = [HasRadius, HasColor, HasPosition2D]
if velocity is not None: components.append(HasMotion2D)   # movers live in a different pool
if custom:               components.append(HasCustom)
world.add_entity(components=components, **data)
```

Each distinct component set is its own [pool](primitives.md); `world_to_dict` records every entity's `components` list, so `world_from_dict` restores it into the right one. Static balls, movers, and custom-tagged balls all save and load without special-casing.

## See also

- [Primitives](primitives.md) — `World(extra_metadata=[...])` and how archetypes/pools work.
- [Systems & Per-Entity Iteration](systems.md) — `to_dict` and the per-entity serialization pattern.
