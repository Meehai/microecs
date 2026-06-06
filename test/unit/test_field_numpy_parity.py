"""_Field numpy-parity contract.

A _Field is a segmented view of a (N, *e) array (one chunk per pool). The promise: for any
*pool-independent* operation it must behave EXACTLY like numpy acting on the concatenated
(N, *e) array -- same values, same shape, and the same exception on bad shapes. These tests
diff _Field against numpy on the materialized array (qr.field.numpy()).

Three buckets, all pinned here:
1. PARITY  -- elementwise/broadcast/feature-axis ops match numpy 1:1 (incl. matching its errors).
2. FOOTGUN -- explicit pool-crossing ops (cumsum/sort over axis 0) differ from numpy on purpose;
              the user opted in. Pinned so the divergence stays a deliberate, visible choice.
3. RAISES  -- entity-axis fancy/range/bool indexing, partial entity writes, and ndarray
              methods/attrs are unsupported and raise (never silently wrong).
"""
from dataclasses import field
import numpy as np
import pytest

from microecs import World, Component
from microecs.query_result import _Field


class HasVel(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


class HasTag(Component):  # forces a second archetype/pool for the velocity query
    tag: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32"})


class HasPose(Component):
    pose: np.ndarray = field(metadata={"shape": (4, 4), "dtype": "float32"})


def _multipool_velocity():
    """A (5,2) velocity field split across two pools (2 rows + 3 rows)."""
    w = World(components=[HasVel, HasTag])
    for v in ([1, 2], [3, 4]):
        w.add_entity(components=[HasVel], velocity=np.array(v, "float32"))
    for v in ([5, 6], [7, 8], [9, 10]):
        w.add_entity(components=[HasVel, HasTag],
                     velocity=np.array(v, "float32"), tag=np.array([0], "float32"))
    w.update()
    return w, w.query(HasVel)


def _multipool_pose():
    """A (5,4,4) pose field split across two pools (2 + 3)."""
    w = World(components=[HasPose, HasTag])
    for i in range(2):
        w.add_entity(components=[HasPose], pose=(i + 1) * np.ones((4, 4), "float32"))
    for i in range(3):
        w.add_entity(components=[HasPose, HasTag],
                     pose=(i + 3) * np.ones((4, 4), "float32"), tag=np.array([0], "float32"))
    w.update()
    return w, w.query(HasPose)


def assert_parity(qr_field, logical, op):
    """op applied to the _Field must equal op applied to the materialized array -- or both raise."""
    try:
        r_np, e_np = op(logical), None
    except Exception as e:                       # noqa: BLE001 - we compare against numpy's own error
        r_np, e_np = None, e
    try:
        r_f, e_f = op(qr_field), None
    except Exception as e:                       # noqa: BLE001
        r_f, e_f = None, e

    if e_np is not None or e_f is not None:
        assert e_np is not None and e_f is not None, (
            f"numpy {'raised '+type(e_np).__name__ if e_np else 'did NOT raise'}, "
            f"_Field {'raised '+type(e_f).__name__ if e_f else 'did NOT raise'}")
        return
    r_f = r_f.numpy() if isinstance(r_f, _Field) else np.asarray(r_f)
    assert r_f.shape == r_np.shape, f"shape {r_f.shape} vs numpy {r_np.shape}"
    assert np.allclose(r_f, r_np), f"values diverge:\n{r_f}\nvs\n{r_np}"


# --- bucket 1: PARITY on a (5,2) field ---

PARITY_OPS_2D = {
    "add_scalar":        lambda x: x + 3.0,
    "add_feature_2":     lambda x: x + np.array([10, 20], "float32"),       # (2,) broadcast
    "add_row_1x2":       lambda x: x + np.array([[10, 20]], "float32"),     # (1,2) broadcast
    "add_entity_5x2":    lambda x: x + np.arange(10, dtype="float32").reshape(5, 2),
    "add_col_5x1":       lambda x: x + np.arange(5, dtype="float32").reshape(5, 1),
    "mul_scalar":        lambda x: x * 2.0,
    "sub_self":          lambda x: x - x,
    "negate":            lambda x: -x,
    "abs":               lambda x: np.abs(x),
    "compare_gt":        lambda x: x > 5,
    "where":             lambda x: np.where(x > 5, -x, x),
    "clip":              lambda x: np.clip(x, 3, 7),
    "maximum":           lambda x: np.maximum(x, 5),
    "norm_axis1":        lambda x: np.linalg.norm(x, axis=1),               # feature reduction -> (5,)
    "sum_axis1":         lambda x: np.sum(x, axis=1),                       # feature reduction -> (5,)
    "feature_index":     lambda x: x[:, 0],
    "feature_slice":     lambda x: x[:, 0:1],
}


@pytest.mark.parametrize("name", list(PARITY_OPS_2D))
def test_parity_2d(name):
    _, qr = _multipool_velocity()
    assert_parity(qr.velocity, qr.velocity.numpy().copy(), PARITY_OPS_2D[name])


# --- bucket 1: numpy's OWN errors are reproduced (not bypassed into a wrong value) ---

BAD_SHAPE_OPS = {
    "add_5_bad":  lambda x: x + np.arange(5, dtype="float32"),              # (5,) !~ (5,2)
    "add_3x2":    lambda x: x + np.arange(6, dtype="float32").reshape(3, 2),  # wrong entity length
}


@pytest.mark.parametrize("name", list(BAD_SHAPE_OPS))
def test_parity_bad_shapes_both_raise(name):
    _, qr = _multipool_velocity()
    assert_parity(qr.velocity, qr.velocity.numpy().copy(), BAD_SHAPE_OPS[name])


# --- bucket 1: 3D field parity (broadcast + batched matmul) ---

PARITY_OPS_3D = {
    "add_scalar":   lambda x: x + 1.0,
    "add_mat_4x4":  lambda x: x + np.eye(4, dtype="float32"),               # (4,4) broadcast over batch
    "matmul_4x4":   lambda x: np.matmul(x, 2 * np.eye(4, dtype="float32")), # batched matmul -> (5,4,4)
    "where":        lambda x: np.where(x > 2, x, -x),
}


@pytest.mark.parametrize("name", list(PARITY_OPS_3D))
def test_parity_3d(name):
    _, qr = _multipool_pose()
    assert_parity(qr.pose, qr.pose.numpy().copy(), PARITY_OPS_3D[name])


# --- bucket 1: assignment broadcasting matches numpy ---

@pytest.mark.parametrize("value_factory,expected_factory", [
    (lambda: 9.0,                                   lambda log: np.full_like(log, 9.0)),
    (lambda: np.array([10, 20], "float32"),         lambda log: np.broadcast_to([10, 20], log.shape).copy()),
    (lambda: np.arange(10, dtype="float32").reshape(5, 2), lambda log: np.arange(10, dtype="float32").reshape(5, 2)),
])
def test_assignment_broadcast_matches_numpy(value_factory, expected_factory):
    """qr.field[:] = value scatters with numpy's broadcasting rules."""
    w, qr = _multipool_velocity()
    logical = qr.velocity.numpy().copy()
    qr.velocity[:] = value_factory()
    w.update()
    assert np.allclose(w.query(HasVel).velocity.numpy(), expected_factory(logical))


# --- bucket 2: explicit pool-crossing ops differ from numpy ON PURPOSE ---

def test_cumsum_axis0_diverges_from_numpy():
    """np.cumsum(axis=0) accumulates WITHIN each pool, resetting at boundaries -- not the global
    scan numpy gives. Documented footgun; pinned so it stays a deliberate divergence."""
    _, qr = _multipool_velocity()
    logical = qr.velocity.numpy().copy()
    field_result = np.cumsum(qr.velocity, axis=0).numpy()
    numpy_result = np.cumsum(logical, axis=0)
    assert not np.array_equal(field_result, numpy_result)   # per-pool != global


def test_sort_axis0_diverges_when_pools_interleave():
    """np.sort(axis=0) sorts within each pool. With values that interleave across pools, that differs
    from the global sort the user might expect."""
    w = World(components=[HasVel, HasTag])
    for v in ([5, 1], [1, 5]):                              # pool A holds the large-then-small rows
        w.add_entity(components=[HasVel], velocity=np.array(v, "float32"))
    w.add_entity(components=[HasVel, HasTag],
                 velocity=np.array([3, 3], "float32"), tag=np.array([0], "float32"))
    w.update()
    qr = w.query(HasVel)
    field_result = np.sort(qr.velocity, axis=0).numpy()
    numpy_result = np.sort(qr.velocity.numpy(), axis=0)
    assert not np.array_equal(field_result, numpy_result)


# --- bucket 3: restricted surface raises (never silently wrong) ---

@pytest.mark.parametrize("indexer", [
    slice(None),                       # qr.f[:]  -- whole-array read
    slice(2, 4),                       # qr.f[2:4] -- entity range
    np.array([1, 0, 1, 0, 1], bool),   # qr.f[mask]
    [0, 2, 4],                         # qr.f[[...]] fancy
])
def test_entity_axis_read_indexing_raises(indexer):
    """Entity-axis indexing beyond a single int crosses pools -> raise, don't return a partial view."""
    _, qr = _multipool_velocity()
    with pytest.raises(TypeError):
        qr.velocity[indexer]


def test_single_int_index_returns_entity_row():
    """The one supported entity index: qr.f[i] -> entity i's row (numpy-consistent value)."""
    _, qr = _multipool_velocity()
    assert np.array_equal(qr.velocity[0], [1, 2])
    assert np.array_equal(qr.velocity[4], [9, 10])          # spans into the second pool


@pytest.mark.parametrize("indexer", [slice(2, 4), np.array([1, 0, 1, 0, 1], bool)])
def test_entity_axis_partial_write_raises(indexer):
    _, qr = _multipool_velocity()
    with pytest.raises(TypeError):
        qr.velocity[indexer] = 0.0


@pytest.mark.parametrize("attr", ["dtype", "ndim", "T", "sum", "mean", "max"])
def test_missing_ndarray_methods_raise(attr):
    """_Field is not a full ndarray: these attrs/methods are absent (AttributeError), so user code
    fails loudly instead of silently. Use .numpy() when a real array is needed."""
    _, qr = _multipool_velocity()
    with pytest.raises(AttributeError):
        getattr(qr.velocity, attr)
