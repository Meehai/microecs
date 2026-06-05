# Example: serialization (extra field + entity traversal + JSON round-trip)

**Created**: 2026-06-05
**Priority**: 3

## Why

Task 11 made `World(extra_field_metadata=["serializable"])` possible, but nothing in
the repo *shows* what it's for. The motivating consumer — a serializer — lives only in
robosim (`server/server_ecs.py`). We want a self-contained `examples/03-serialization.py`
that demonstrates the whole design pattern end to end so a reader gets it without robosim:

1. **Extra field metadata** — `serializable: True/False` per field, declared on the world.
   Shows *why* the flag exists: runtime handles (an `object`-dtype live resource) are
   marked `serializable: False` and skipped; data fields are persisted.
2. **Traversing entities** — walk every live entity, `world.get_entity(eid)`, then walk
   each component's `fields()` reading `field.metadata["serializable"]`. This is the
   "object-like" access path (task 10) used for export, not the batched query path.
3. **JSON structure** — `world_to_dict` → `json.dumps` → `json.loads` → `world_from_dict`,
   and assert the round-trip preserves the serializable data.

This is **not** a new library feature. The plan keeps built-in serialization out of scope
(plan §"Don't need": *"`object` dtype + pickle is the escape hatch"*). The pattern lives at
**application level** — this task just documents it as a runnable example, mirroring robosim.

## What to build (dev's call — `examples/03-serialization.py`)

Strip robosim down to the bones. **No raylib, no window, no sockets** — pure `numpy` + `json`
so it runs in CI like `examples/01`. Keep it grug: a couple of components, a couple of
factory functions, the two dict functions, a `main()` that does the round-trip and asserts.

### Components — show both flag values

```python
class HasResource(Component):  # the "why skip" case: a live runtime handle, not data
    handle: np.ndarray = field(metadata={"shape": (1,), "dtype": "object", "serializable": False})
class HasPose(Component):
    pose: np.ndarray = field(metadata={"shape": (4, 4), "dtype": "float32", "serializable": True})
class HasType(Component):  # the reconstruction recipe (mirrors robosim _type/_args)
    _type: np.ndarray = field(metadata={"shape": (1,), "dtype": "object", "serializable": True})
    _args: np.ndarray = field(metadata={"shape": (1,), "dtype": "object", "serializable": True})
```

### The pattern's keystone: `_type` + `_args`

Serializable data alone can't rebuild an entity that owns a non-serializable handle (the
handle is gone). So each entity stores a **recipe**: `_type` (which factory) + `_args` (its
kwargs). Export drops the handle; import calls the factory, which *recreates* a fresh handle.
This is the whole point — copy robosim's `_make_cube`/`_make_robot` shape into 1–2 tiny
`_make_*(world, ...)` factories that build the live handle from `_args`.

### `world_to_dict` — copy robosim `server_ecs.py:164-180`

Iterate `world._live_ids`; per entity walk `fields(component)`, skip `not
metadata["serializable"]`, emit `.item()` for `object` dtype else `.tolist()`. Return
`{"entities": [...], "components": world.component_names,
"extra_field_metadata": world.extra_field_metadata}`.

### `world_from_dict` — copy robosim `server_ecs.py:182-206`, **fix the nesting**

robosim reads `data["scene"]["components"]` because its world dict is nested under `"scene"`
inside `Simulator.to_dict`. Standalone, there is no `Simulator` — read the **top-level** keys
(`data["components"]`, `data["entities"]`, `data["extra_field_metadata"]`) so to/from are
symmetric. Resolve component name → type via a small registry (or `globals()[name]`), build
the world, then dispatch on `entity["data"]["_type"]["type"]` to the right `_make_*`.

### `main()`

Build a world with 2–3 entities, `world.update()`, `d = world_to_dict(world)`,
`s = json.dumps(d, indent=2)`, print it, `w2 = world_from_dict(json.loads(s))`,
`w2.update()`, assert the serializable fields match (e.g. compare `world_to_dict(w2)` to `d`).
Keep components + `world_to_dict`/`world_from_dict` at **module level** and guard the run with
`if __name__ == "__main__":` so the tester can `import` them.

## Done when

- `examples/03-serialization.py` runs standalone (`python examples/03-serialization.py`),
  prints readable JSON, and its internal round-trip assert passes — no raylib, no network.
- `python examples/03-serialization.py --help` exits 0 (CI smoke test, like `examples/01`).
- A `serializable: False` field (the runtime handle) is **absent** from the JSON; every
  `serializable: True` field is present and survives the round-trip.
- `world_from_dict(world_to_dict(w))` rebuilds an equivalent world: same entity count, same
  serializable data; the dropped handle is freshly recreated by the factory (not the old one).
- The example reads its own top-level keys (no leftover `["scene"]` nesting from robosim).

## Tests (tester writes, `test/` — example must be importable)

- `test_serialization_roundtrip` — build world, `d = world_to_dict`, dump+load JSON,
  `world_from_dict`, assert `world_to_dict(rebuilt) == d`.
- `test_non_serializable_field_excluded` — assert the `serializable: False` field name never
  appears in the dumped dict; a `serializable: True` field does.
- `test_handle_is_recreated` — the rebuilt entity's handle is a *new* object (recipe rebuilt
  it), not the original instance, yet the serializable data is identical.
- `test_json_is_valid` — `json.dumps(world_to_dict(world))` is parseable and stable across a
  second round-trip (`to→from→to` is a fixed point).
- `test_example_help_exits_zero` — subprocess `python examples/03-serialization.py --help`
  returns 0 (guards the CI smoke test).
- *(optional)* `test_extra_field_metadata_carried` — dumped `extra_field_metadata ==
  ["serializable"]` and the rebuilt world declares the same, so re-construction re-validates.

## Out of scope

- **Built-in serialization in `microecs/`.** Stays application-level (plan §"Don't need").
  No `World.to_dict`/`from_dict` on the library.
- **pickle / binary formats.** JSON only; `object`-dtype fields that aren't JSON-able must be
  marked `serializable: False` and rebuilt via the recipe (that's the lesson).
- **Partial / incremental save, versioning, schema migration.** Whole-world snapshot only.
- **Camera/plugin/`SimState` serialization** from robosim — not relevant without the simulator.

## Related

- robosim `server/server_ecs.py:164-206` (`world_to_dict`/`world_from_dict`),
  `:71-99` (per-field `serializable` flags), `:116-162` (`_make_*` factories with `_type`/`_args`),
  `:491-493` (`World(..., extra_field_metadata=["serializable"])`). The source to trim down.
- [task 11](../../done/11-extra-field-metadata/TASK.md) — the `extra_field_metadata` kwarg this
  example consumes (`world.py:19-20,198,206`).
- [task 10](../../done/10-single-entity-read/TASK.md) — `world.get_entity` (`world.py:71-77`),
  the object-like read path the exporter walks.
- `world.py:25` `component_names`, `world.py:20` `extra_field_metadata`, `world._live_ids` —
  the fields `world_to_dict` reads.
- `examples/01-hello-world.py` — style/CLI/`--help` reference; CI runs its `--help`
  (`.gitlab-ci.yml`).
- README:16 documents `extra_field_metadata`; once this lands, README could link the example.
