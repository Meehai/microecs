"""Field-shape invariant for QueryResult / Field.

The contract: a field declared (*e) — e.g. pose=(4,4) — is an (N, *e) array-like. Two rules,
at two layers, that these tests pin:

1. STORED shape is frozen. Once a pool allocates a field as (capacity, *e) from component
   metadata, the feature dims (*e) never change. No op, assignment, realloc or migration can
   turn a (N,4,4) pose into anything else. Enforced structurally by Pool (fixed buffers,
   in-place writes only). You can only break it by trying to *assign* a wrong-shape value,
   which must raise — never silently reshape.

2. TRANSIENT results preserve only the entity axis (N), not the feature dims. Field applies
   a numpy op per-pool and rebuilds; the only invariant it can/should enforce is
   `result.shape[0] == rows` (query_result.py: _apply_fn_on_parts). So np.linalg.norm(v, axis=1)
   is fine ((N,2)->(N,)); a reduction over the entity axis (axis=0) is rejected. Crossing pools
   while keeping N (e.g. np.sort(..., axis=0)) is NOT detectable per-pool and is documented as
   user error — see test_sort_axis0_is_documented_per_pool_footgun.
"""
from dataclasses import field
import numpy as np
import pytest

from microecs import World, Component
from microecs.query_result import Field


class HasPose(Component):  # 2D field: the (4,4) the contract is written around
    pose: np.ndarray = field(metadata={"shape": (4, 4), "dtype": "float32", "default": None})


class HasVelocity(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasScale(Component):  # 0-d field: feature shape () -> field is (N,)
    scale: np.ndarray = field(metadata={"shape": (), "dtype": "float32", "default": None})


def _pose(v: float) -> np.ndarray:
    return (v * np.eye(4)).astype("float32")


# --- rule 1: stored shape matches metadata and survives every mutation path ---

def test_query_field_shape_matches_metadata():
    """Across a multi-pool query, every field is exposed as (N, *e) with e from its metadata."""
    world = World(components=[HasPose, HasVelocity, HasScale])
    world.add_entity(components=[HasPose], pose=_pose(1))                       # pool A
    world.add_entity(components=[HasPose, HasVelocity],                         # pool B
                     pose=_pose(2), velocity=np.array([3, 4], "float32"))
    world.add_entity(components=[HasPose, HasScale],                            # pool C
                     pose=_pose(3), scale=np.array(5.0, "float32"))
    world.update()

    assert world.query(HasPose).pose.shape == (3, 4, 4)       # spans 3 pools
    assert world.query(HasVelocity).velocity.shape == (1, 2)
    assert world.query(HasScale).scale.shape == (1,)          # 0-d feature -> (N,)


def test_field_shape_survives_inplace_arithmetic():
    """qr.field[:] = <expr over qr.field> keeps (N, *e); pool buffer keeps (*e)."""
    world = World(components=[HasPose])
    for v in (1, 2, 3):
        world.add_entity(components=[HasPose], pose=_pose(v))
    world.update()

    qr = world.query(HasPose)
    qr.pose[:] = qr.pose + 1
    world.update()

    qr2 = world.query(HasPose)
    assert qr2.pose.shape == (3, 4, 4)
    for part in qr2.pose.parts:                # every pool's buffer still (.,4,4)
        assert part.shape[1:] == (4, 4)


def test_field_shape_survives_broadcast_assign():
    """Broadcasting a single (4,4) into every row keeps the field (N,4,4)."""
    world = World(components=[HasPose])
    for v in (1, 2):
        world.add_entity(components=[HasPose], pose=_pose(v))
    world.update()

    qr = world.query(HasPose)
    qr.pose[:] = np.zeros((4, 4), "float32")
    world.update()

    qr2 = world.query(HasPose)
    assert qr2.pose.shape == (2, 4, 4)
    assert np.array_equal(qr2.pose.numpy(), np.zeros((2, 4, 4), "float32"))


def test_field_shape_survives_field_to_field_assign():
    """The np.where(...) -> Field path scatters back without changing shape."""
    world = World(components=[HasVelocity])
    for v in ([1, 2], [3, 4], [5, 6]):
        world.add_entity(components=[HasVelocity], velocity=np.array(v, "float32"))
    world.update()

    qr = world.query(HasVelocity)
    mask = np.zeros((len(qr), 2), bool)                       # all-False: a no-op flip
    qr.velocity[:] = np.where(mask, -qr.velocity, qr.velocity)
    world.update()

    qr2 = world.query(HasVelocity)
    assert qr2.velocity.shape == (3, 2)
    assert np.array_equal(qr2.velocity.numpy(), [[1, 2], [3, 4], [5, 6]])


def test_field_shape_survives_realloc():
    """Growing past INITIAL_CAPACITY (and shrinking back) only touches the entity axis."""
    world = World(components=[HasPose])
    ids = [world.add_entity(components=[HasPose], pose=_pose(i)) for i in range(150)]  # > 100 -> grow
    world.update()
    assert world.query(HasPose).pose.shape == (150, 4, 4)

    for eid in ids[40:]:                                      # drop to 40 -> shrink realloc
        world.remove_entity(eid)
    world.update()
    qr = world.query(HasPose)
    assert qr.pose.shape == (40, 4, 4)
    for part in qr.pose.parts:
        assert part.shape[1:] == (4, 4)


def test_field_shape_survives_migration():
    """add_component / remove_component move an entity across pools; pose stays (.,4,4)."""
    world = World(components=[HasPose, HasVelocity])
    eid = world.add_entity(components=[HasPose], pose=_pose(7))
    world.update()
    assert world.query(HasPose).pose.shape == (1, 4, 4)

    world.get_entity(eid).add_component(HasVelocity, velocity=np.array([1, 1], "float32"))   # -> pose+vel pool
    world.update()
    entity = world.get_entity(eid)
    assert entity.pose.shape == (4, 4)
    assert np.array_equal(entity.pose, _pose(7))          # data intact across migration

    world.get_entity(eid).remove_component(HasVelocity)                 # -> back to pose-only pool
    world.update()
    assert world.query(HasPose).pose.shape == (1, 4, 4)


# --- rule 1, negative: a settled field cannot be reshaped by assignment ---

def test_assign_wrong_feature_shape_raises():
    """You cannot write a (.,2) into a (.,4,4) field — must raise, never silently coerce."""
    world = World(components=[HasPose, HasVelocity])
    world.add_entity(components=[HasPose, HasVelocity],
                     pose=_pose(1), velocity=np.array([3, 4], "float32"))
    world.update()

    qr = world.query(HasPose, HasVelocity)
    with pytest.raises(ValueError):                          # broadcast path: (N,2) !-> (N,4,4)
        qr.pose[:] = np.ones((len(qr), 2), "float32")
    with pytest.raises(ValueError):                          # Field path: velocity field into pose
        qr.pose[:] = qr.velocity


# --- rule 2: transient ops preserve N only; entity-axis collapse is rejected ---

def test_transient_op_preserves_entity_axis_not_feature():
    """norm(v, axis=1): (N,2)->(N,) is allowed (feature reduced, N kept) and matches numpy."""
    world = World(components=[HasVelocity])
    for v in ([3, 4], [5, 12]):
        world.add_entity(components=[HasVelocity], velocity=np.array(v, "float32"))
    world.update()

    qr = world.query(HasVelocity)
    mag = np.linalg.norm(qr.velocity, axis=1)
    assert isinstance(mag, Field)
    assert mag.shape == (2,)                                 # N kept, feature collapsed
    assert np.allclose(mag.numpy(), [5.0, 13.0])             # == norm on the materialized array


def test_op_breaking_entity_axis_raises():
    """Reducing over the entity axis violates the (N, ...) contract and is rejected -- BUT only
    because the guard compares result.shape[0] to the pool's row count. With 3 rows, axis=0 sum
    -> (2,) and 2 != 3 trips it. See the coincidence hole below: the guard is necessary, not
    sufficient (same class as the sort footgun)."""
    world = World(components=[HasVelocity])
    for v in ([3, 4], [5, 12], [1, 1]):                      # 3 rows in one pool: reduced len 2 != 3
        world.add_entity(components=[HasVelocity], velocity=np.array(v, "float32"))
    world.update()

    qr = world.query(HasVelocity)
    with pytest.raises(AssertionError):                      # axis=0 -> (2,), rows mismatch -> line 36
        np.sum(qr.velocity, axis=0)
    # NOTE for dev: a full reduction yields a numpy scalar, so `len(part_result)` (query_result.py:36)
    # raises TypeError ("no len()") instead of a clean AssertionError. Guard `ndim >= 1` in
    # _apply_fn_on_parts for a better message; widen/adjust this raises() if you do.
    with pytest.raises((AssertionError, TypeError)):
        np.sum(qr.velocity)


def test_entity_axis_collapse_can_slip_through_when_len_coincides():
    """The guard is shape[0]==rows, not 'is this a reduction'. A single pool whose row count
    equals the reduced feature length passes silently -- documented hole, same family as
    test_sort_axis0_is_documented_per_pool_footgun. Pinned so nobody assumes axis=0 is safe."""
    world = World(components=[HasVelocity])
    for v in ([3, 4], [5, 12]):                              # 2 rows; sum(axis=0) -> (2,), 2 == 2
        world.add_entity(components=[HasVelocity], velocity=np.array(v, "float32"))
    world.update()

    qr = world.query(HasVelocity)
    sneaky = np.sum(qr.velocity, axis=0)                     # does NOT raise: coincidence rows==2==feat
    assert isinstance(sneaky, Field)
    assert sneaky.shape == (2,)                              # presented as N=2, but it's one summed row
    # it is NOT the per-entity data: the materialized field would be the two original rows
    assert not np.array_equal(sneaky.numpy(), qr.velocity.numpy())


def test_axis0_reduction_assign_back_broadcasts_silently():
    """The assign-back does NOT rescue an axis=0 reduction. When N == feature length, sum(axis=0) gives
    (feat,), which numpy then broadcasts across every row on write-back -- no shape mismatch, no raise.
    Pinned because it's the worst case: silent corruption, not an error. (N != feature is caught earlier by
    the per-pool guard; this is the coincidence that slips through.)"""
    world = World(components=[HasVelocity])
    for v in ([5, 1], [1, 5]):                               # N == feature == 2
        world.add_entity(components=[HasVelocity], velocity=np.array(v, "float32"))
    world.update()

    qr = world.query(HasVelocity)
    qr.velocity[:] = np.sum(qr.velocity, axis=0)             # (2,2)->(2,)-> broadcast back, NO raise
    world.update()

    after = world.query(HasVelocity).velocity.numpy()
    assert np.array_equal(after, [[6, 6], [6, 6]])           # column-sum copied into every row -- silently wrong


def test_zero_d_feature_currently_yields_1d_field():
    """Reality check on the ">= 2D" invariant: it is NOT enforced today. shape=() is accepted and the field is
    1D (N,). This test documents current behavior; if _check_components later forbids () (len(shape) >= 1), flip
    this to assert World([HasScale]) raises and drop HasScale's () usage. See HasScale in test_world.py."""
    world = World(components=[HasScale])
    world.add_entity(components=[HasScale], scale=np.array(3.0, "float32"))
    world.add_entity(components=[HasScale], scale=np.array(7.0, "float32"))
    world.update()

    qr = world.query(HasScale)
    assert qr.scale.shape == (2,)                            # 1D field: (N,), not (N, *e)
    assert qr.scale.numpy().ndim == 1


def test_sort_axis0_is_documented_per_pool_footgun():
    """np.sort(axis=0) keeps N so it passes the per-pool check, but sorts WITHIN each pool, not
    globally. This is documented user error, not a library guarantee. Pinned so a future change
    that 'fixes' or forbids it is a deliberate, visible decision."""
    world = World(components=[HasVelocity, HasScale])
    # two pools so global vs per-pool diverge: pool A (vel only) holds the unsorted rows.
    for v in ([5, 1], [1, 5]):
        world.add_entity(components=[HasVelocity], velocity=np.array(v, "float32"))
    world.add_entity(components=[HasVelocity, HasScale],
                     velocity=np.array([3, 3], "float32"), scale=np.array(0.0, "float32"))
    world.update()

    qr = world.query(HasVelocity)
    result = np.sort(qr.velocity, axis=0)                    # does NOT raise
    assert isinstance(result, Field)
    assert result.shape == (3, 2)                            # N preserved -> passes the guard
    # per-pool result differs from the global sort the user probably expected
    assert not np.array_equal(result.numpy(), np.sort(qr.velocity.numpy(), axis=0))
