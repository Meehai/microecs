#!/usr/bin/env python3
"""Throwaway probe: does microecs already support field-less (tag) components, and what breaks?

Run: PYTHONPATH=. python test/manual/tag-components-probe/probe.py
Investigates tasks 8 (none_of) and 9 (tag components) — informs the task specs.
"""
from dataclasses import field
import numpy as np
from microecs import World, Component


class HasPosition(Component):
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class Player(Component):  # zero-field tag
    pass


class Frozen(Component):  # zero-field tag
    pass


def banner(s):
    print(f"\n=== {s} ===")


banner("define field-less tag component")
print("Player.__dataclass_fields__ =", Player.__dataclass_fields__)
print("Player() instantiates =", Player())

banner("world with a tag in the component list")
try:
    world = World(components=[HasPosition, Player, Frozen])
    print("OK world created; component_to_field_names =",
          {c.__name__: world.component_to_field_names[c] for c in (HasPosition, Player, Frozen)})
except Exception as e:
    print("FAILED at world creation:", repr(e))
    raise SystemExit

banner("add entity with Position + Player tag (no data for Player)")
try:
    e_tagged = world.add_entity(components=(HasPosition, Player), position=np.array([1.0, 2.0], "float32"))
    e_plain = world.add_entity(components=(HasPosition,), position=np.array([3.0, 4.0], "float32"))
    world.update()
    print("OK added tagged + plain. pools =", len(world.pools))
except Exception as e:
    print("FAILED adding tagged entity:", repr(e))
    raise SystemExit

banner("query_and((Player,)) — should match only the tagged entity")
try:
    qr = world.query_and((Player,))
    print("len =", len(qr), "entity_ids =", qr.entity_ids.tolist(), "fields =", qr._fields)
except Exception as e:
    print("FAILED query (Player,):", repr(e))

banner("query_and((HasPosition, Player)) — tag as a filter, expose position")
try:
    qr = world.query_and((HasPosition, Player))
    print("len =", len(qr), "entity_ids =", qr.entity_ids.tolist())
    print("qr.position.numpy() =", qr.position.numpy().tolist())
except Exception as e:
    print("FAILED query (HasPosition, Player):", repr(e))

banner("get_entity on the tagged entity")
try:
    data, comps = world.get_entity(e_tagged)
    print("data =", {k: v.tolist() for k, v in data.items()}, "comps =", [c.__name__ for c in comps])
except Exception as e:
    print("FAILED get_entity:", repr(e))

banner("PURE tag entity: add_entity((Player,)) with no data at all")
try:
    e_pure = world.add_entity(components=(Player,))
    world.update()
    qr = world.query_and((Player,))
    print("OK pure-tag entity. (Player,) len =", len(qr), "entity_ids =", qr.entity_ids.tolist())
except Exception as e:
    print("FAILED pure-tag entity:", repr(e))

banner("remove a tag via remove_component, then re-query")
try:
    world.remove_component(e_tagged, Player)
    world.update()
    qr = world.query_and((Player,))
    print("OK removed Player from e_tagged. (Player,) entity_ids now =", qr.entity_ids.tolist())
except Exception as e:
    print("FAILED remove tag component:", repr(e))

print("\nprobe done.")
