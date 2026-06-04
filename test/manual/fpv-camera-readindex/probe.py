"""Probe: robosim FPV-camera per-entity read loop against the current QueryResult.

Confirms three things before writing the real test:
  1. an object-dtype component (a camera object) round-trips through a pool + query
  2. `qr.field[i]` (integer entity-axis read) is REJECTED today
  3. iterating the fields (`zip(qr.cam, qr.pose)`) already gives per-entity views -> the loop works
"""
import numpy as np
from microecs import World, Component
from dataclasses import field


class FakeCam:
    """Stand-in for robosim's camera: holds whatever set_position was last handed."""
    def __init__(self, name):
        self.name = name
        self.seen = None

    def set_position(self, position, up, target):
        self.seen = (position.tolist(), up.tolist(), target.tolist())


class HasFPV(Component):
    fpv_camera: np.ndarray = field(metadata={"shape": (), "dtype": "object"})


class HasPose(Component):
    pose: np.ndarray = field(metadata={"shape": (4, 4), "dtype": "float32"})


class HasName(Component):  # only here to force a SECOND archetype/pool
    name: np.ndarray = field(metadata={"shape": (), "dtype": "object"})


def pose_at(tx, ty, tz):
    p = np.eye(4, dtype="float32")
    p[0:3, 3] = (tx, ty, tz)
    return p


w = World([HasFPV, HasPose, HasName])
camA, camB = FakeCam("A"), FakeCam("B")
# two different archetypes so the query stitches across >1 pool
w.add_entity([HasFPV, HasPose], fpv_camera=np.array(camA, dtype=object), pose=pose_at(1, 0, 0))
w.add_entity([HasFPV, HasPose, HasName], fpv_camera=np.array(camB, dtype=object),
             pose=pose_at(0, 2, 0), name=np.array("b", dtype=object))
w.update()

qr = w.query_and([HasPose, HasFPV])
print("len(qr) =", len(qr), "| pools =", len(qr.pool_list))

# (2) the literal robosim spelling -- integer entity-axis read
try:
    _ = qr.fpv_camera[0]
    print("qr.fpv_camera[0] -> OK (returned)", _)
except TypeError as e:
    print("qr.fpv_camera[0] -> REJECTED:", e)

# (3) the idiomatic equivalent that already works: iterate the fields together.
# object fields come back as 0-d arrays -> .item() unwraps (exactly robosim's qr.fpv_camera[i].item())
for cam, pose in zip(qr.fpv_camera, qr.pose):
    cam.item().set_position(position=pose[0:3, 3], up=pose[0:3, 1], target=pose[0:3, 3] + pose[0:3, 2])

print("camA.seen =", camA.seen)
print("camB.seen =", camB.seen)
print("is a view?", "yes" if qr.pose.parts[0].base is not None else "n/a")
