"""Integration: robosim's FPV-camera per-entity read loop over a QueryResult.

robosim drives cameras one entity at a time:

    qr = world.query(HasPose, HasFPV)
    for i in range(len(qr)):
        cam, pose = qr.fpv_camera[i].item(), qr.pose[i]
        cam.set_position(position=pose[0:3, 3], up=pose[0:3, 1],
                         target=pose[0:3, 3] + pose[0:3, 2])

`fpv_camera` is an object-dtype component -- a camera you call methods on -- so the work can't be vectorised;
it's a genuine per-entity loop. These tests drive it over a query that spans two archetypes, both ways: by
iterating the fields (`zip(qr.fpv_camera, qr.pose)`) and by id (`world.get_entity(qr.entity_ids[i])`), with
each `pose` a real (4, 4) view sliced by column. (Task 16 removed the old `qr.pose[i]` int-index spelling;
random per-entity access now goes through `get_entity`.)
"""
from dataclasses import field
import numpy as np
import pytest

from microecs import World, Component

class FakeCamera:
    """Stand-in for robosim's camera: records the last set_position call so the test can read it back."""
    def __init__(self, name: str):
        self.name = name
        self.seen: tuple | None = None

    def set_position(self, position, up, target):
        self.seen = (position.tolist(), up.tolist(), target.tolist())


class HasFPV(Component):  # object-dtype: each entity carries a camera OBJECT, methods called one by one
    fpv_camera: np.ndarray = field(metadata={"shape": (), "dtype": "object"})


class HasPose(Component):  # a 4x4 rigid transform; columns are right/up/forward, last column is translation
    pose: np.ndarray = field(metadata={"shape": (4, 4), "dtype": "float32"})


class HasName(Component):  # only here to force a SECOND archetype so the query spans two pools
    name: np.ndarray = field(metadata={"shape": (), "dtype": "object"})


def _pose(rot3x3: list[list[float]], translation: list[float]) -> np.ndarray:
    """Build a (4, 4) pose from a 3x3 rotation (its columns are the up/forward axes) and a translation."""
    p = np.eye(4, dtype="float32")
    p[0:3, 0:3] = np.array(rot3x3, "float32")
    p[0:3, 3] = np.array(translation, "float32")
    return p


_IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]       # up=+Y (col1), forward=+Z (col2)
_ROT_X_90 = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]      # up=+Z (col1), forward=-Y (col2)


class FPVCameraSystem:
    """robosim's camera driver, via field iteration.

    Object fields come back as 0-d arrays, so `.item()` unwraps to the camera; `pose` is a (4, 4) view sliced
    by column."""
    def __call__(self, world: World):
        qr = world.query(HasPose, HasFPV)
        for cam, pose in zip(qr.fpv_camera, qr.pose):
            cam.item().set_position(position=pose[0:3, 3], up=pose[0:3, 1],
                                    target=pose[0:3, 3] + pose[0:3, 2])


def _world_with_two_cameras() -> tuple[World, FakeCamera, FakeCamera]:
    """One camera in archetype {FPV, Pose}, one in {FPV, Pose, Name} -- the query stitches across both pools."""
    cam_a, cam_b = FakeCamera("a"), FakeCamera("b")
    world = World(components=[HasFPV, HasPose, HasName])
    # cam_a added FIRST -> its pool is created first -> it is entity 0 of the query
    world.add_entity((HasFPV, HasPose),
                     fpv_camera=np.array(cam_a, dtype=object), pose=_pose(_IDENTITY, [1, 0, 0]))
    world.add_entity((HasFPV, HasPose, HasName),
                     fpv_camera=np.array(cam_b, dtype=object), pose=_pose(_ROT_X_90, [0, 2, 0]),
                     name=np.array("b", dtype=object))
    world.update()
    return world, cam_a, cam_b


def test_fpv_camera_per_entity_loop_drives_each_camera_across_two_archetypes():
    """The per-entity camera loop runs through a real World query spanning two pools. Each camera's set_position
    gets the vectors derived from ITS OWN pose -- proving object components + (4, 4) pose views iterate per
    entity, in whichever pool the entity lives, with no .numpy() gather and no vectorisation."""
    world, cam_a, cam_b = _world_with_two_cameras()
    pool_a = world.pools[world._make_key((HasFPV, HasPose))]
    pool_b = world.pools[world._make_key((HasFPV, HasPose, HasName))]
    assert len(pool_a) == 1 and len(pool_b) == 1            # the query really spans two pools

    FPVCameraSystem()(world)

    # cam_a: identity pose at (1,0,0) -> up=+Y, forward=+Z, target = pos + forward
    assert cam_a.seen == ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0])
    # cam_b: rot-X-90 pose at (0,2,0) -> up=+Z (col1), forward=-Y (col2), target = (0,2,0)+(0,-1,0)
    assert cam_b.seen == ([0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])


def test_object_component_unwraps_with_item_during_iteration():
    """Iterating an object-dtype field yields 0-d arrays; `.item()` hands back the very camera object that was
    stored, not a copy, so calling methods on it mutates the real object."""
    world, cam_a, cam_b = _world_with_two_cameras()
    qr = world.query(HasPose, HasFPV)

    cams = [boxed.item() for boxed in qr.fpv_camera]

    assert cams == [cam_a, cam_b]                           # identity preserved, pool-by-pool order
    assert all(c.seen is None for c in cams)                # untouched until a system drives them


def test_fpv_camera_per_entity_read_via_get_entity():
    """robosim's per-entity read after task 16: the old `qr.pose[i]` int-index is forbidden (raises TypeError);
    the entity is fetched by id with `world.get_entity(qr.entity_ids[i])`, which gives the same (4, 4) pose view
    and the same camera object, out of whichever pool the entity lives in."""
    world, cam_a, _ = _world_with_two_cameras()
    qr = world.query(HasPose, HasFPV)

    with pytest.raises(TypeError):
        qr.pose[0]                                          # the old indexed spelling is gone

    e0 = world.get_entity(int(qr.entity_ids[0]))           # entity 0, by id
    assert e0.pose.shape == (4, 4)
    assert e0.pose[0:3, 3].tolist() == [1.0, 0.0, 0.0]     # cam_a's translation, a live (4, 4) view
    assert e0.fpv_camera.item() is cam_a                   # entity 0's camera object
