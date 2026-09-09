"""Unit tests for `qr.field[mask]` boolean-mask reads & writes (task 51).

The whole contract is numpy parity: `qr.field[mask]` must behave exactly like
`qr.field.numpy()[mask]` for READS (gather: one copy) and WRITES (scatter into the pool rows) --
for single-pool queries (`_QRArray`, plain ndarray views -- works today) and for multi-pool
queries (`QRField`, the entity-axis bool mask branch that this task adds).

Pins beyond parity:

- mask must be `(n,)` or `(n, 1)` bool (the `(n, 1)` form is the natural `qr.field == v`
  comparison on a `(1, ...)` scalar field; it is SQUEEZED to the entity axis -- numpy's own
  `(n, 1)` boolean indexing is elementwise-flattening and would be wrong here).
- a mask may itself be a `QRField` (e.g. `qr.velocity[:, 0] > 3`) -- coerced via `.numpy()`.
- wrong length / non-bool dtype / wrong value shape raise; a rejected WRITE leaves the pool
  byte-identical (the eager no-partial-write guarantee, as in the set_data tests).
- exception types pinned: non-bool dtype -> TypeError, wrong length -> ValueError,
  wrong value shape -> ValueError.
"""
from dataclasses import field
import numpy as np
import pytest

from microecs import World, Component


class HasVel(Component):
    velocity: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasAcc(Component):
    acceleration: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32", "default": None})


class HasCount(Component):
    count: np.ndarray = field(metadata={"shape": (1,), "dtype": "int32", "default": None})


class HasFlag(Component):
    flag: np.ndarray = field(metadata={"shape": (1,), "dtype": "bool", "default": None})


class HasName(Component):
    name: np.ndarray = field(metadata={"shape": (1,), "dtype": "object", "default": None})


class HasTag(Component):  # forces a second archetype/pool for the HasVel query
    tag: np.ndarray = field(metadata={"shape": (1,), "dtype": "float32", "default": None})


# row values (two pools: 2 rows without HasTag + 3 rows with HasTag)
ROWS = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
NAMES = ["r0", "r1", "r2", "r3", "r4"]
MASK5 = np.array([True, False, True, False, True])          # selects n=3 rows, crosses the pool boundary


def _world(multipool: bool):
    w = World(components=[HasVel, HasAcc, HasCount, HasFlag, HasName, HasTag])
    for i, v in enumerate(ROWS[:2]):
        w.add_entity(components=[HasVel, HasAcc, HasCount, HasFlag, HasName],
                     velocity=np.array(v, "float32"), acceleration=np.array(v, "float32"),
                     count=np.array([i], "int32"), flag=np.array([i % 2 == 0], "bool"),
                     name=np.array([NAMES[i]], "object"))
    for i, v in enumerate(ROWS[2:]):
        comps = [HasVel, HasAcc, HasCount, HasFlag, HasName] + ([HasTag] if multipool else [])
        data = dict(velocity=np.array(v, "float32"), acceleration=np.array(v, "float32"),
                    count=np.array([i + 2], "int32"), flag=np.array([(i + 2) % 2 == 0], "bool"),
                    name=np.array([NAMES[i + 2]], "object"))
        if multipool:
            data["tag"] = np.array([0.0], "float32")
        w.add_entity(components=comps, **data)
    w.update()
    # query every data component (NOT HasTag: it's only on pool B, and querying it would
    # exclude pool A) so the field set covers all dtypes the tests read from one qr
    return w, w.query(HasVel, HasAcc, HasCount, HasFlag, HasName)


# --- READS: `qr.field[mask]` == `qr.field.numpy()[mask]` -----------------------------------------------


@pytest.mark.parametrize("mask", [MASK5, MASK5[:, None]], ids=["n", "n_1"])
def test_read_mask_matches_numpy_multipool(mask):
    _, qr = _world(multipool=True)
    squeezed = mask.reshape(-1)
    assert np.array_equal(qr.velocity[mask], qr.velocity.numpy()[squeezed])
    assert isinstance(qr.velocity[mask], np.ndarray)          # gather returns a plain array, like numpy


def test_read_mask_matches_numpy_single_pool():
    """Single-pool queries return an ndarray view -> masks already work (regression pin)."""
    _, qr = _world(multipool=False)
    assert np.array_equal(qr.velocity[MASK5], qr.velocity.numpy()[MASK5])


def test_read_mask_dtypes_parity():
    """float32 / int32 / bool / object all gather identically to numpy."""
    for name in ("velocity", "count", "flag", "name"):
        _, qr = _world(multipool=True)
        assert np.array_equal(getattr(qr, name)[MASK5], getattr(qr, name).numpy()[MASK5]), name


def test_read_mask_from_field_comparison():
    """The systems.py shape: mask = (qr.physics_level[:, 0] == level) is itself a QRField."""
    _, qr = _world(multipool=True)
    mask = qr.velocity[:, 0] > 3.0                             # QRField bool (5,)
    assert np.array_equal(qr.velocity[mask], qr.velocity.numpy()[mask.numpy()])


def test_read_mask_from_other_query():
    """A mask computed on a DIFFERENT query of the same length is accepted (pure numpy semantics)."""
    w1, qr1 = _world(multipool=True)
    _, qr2 = _world(multipool=True)
    mask = np.array([r0 > 4.0 for r0 in qr2.velocity.numpy()[:, 0]], bool)   # from qr2's data
    assert np.array_equal(qr1.velocity[mask], qr1.velocity.numpy()[mask])


def test_read_mask_all_false_empty_but_shape_ok():
    _, qr = _world(multipool=True)
    assert qr.velocity[np.zeros(5, bool)].shape == (0, 2)


# --- WRITES: `qr.field[mask] = v` scatters into the pool rows -------------------------------------------


def test_write_mask_scatter_matches_numpy_multipool():
    """(k, ...) value spanning the pool boundary: rows 0/2/4 -> one in pool A, two in pool B."""
    w, qr = _world(multipool=True)
    value = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], "float32")
    qr.velocity[MASK5] = value
    logical = np.array(ROWS, "float32")
    logical[MASK5] = value
    assert np.array_equal(qr.velocity.numpy(), logical)
    assert not np.array_equal(qr.velocity.numpy(), np.array(ROWS, "float32"))  # really changed


def test_write_mask_scalar_broadcast():
    _, qr = _world(multipool=True)
    qr.velocity[MASK5] = 0.0
    logical = np.array(ROWS, "float32")
    logical[MASK5] = 0.0
    assert np.array_equal(qr.velocity.numpy(), logical)


def test_write_mask_squeezed_from_2d():
    _, qr = _world(multipool=True)
    qr.velocity[MASK5[:, None]] = np.full((3, 2), 1.0, "float32")
    logical = np.array(ROWS, "float32")
    logical[MASK5] = 1.0
    assert np.array_equal(qr.velocity.numpy(), logical)


def test_write_mask_from_field_read_composition():
    """qr.a[mask] = qr.b[mask]: the read result fed straight back as the write value."""
    _, qr = _world(multipool=True)
    qr.velocity[MASK5] = qr.acceleration[MASK5]                # same shape (3, 2), no copy needed
    logical = np.array(ROWS, "float32")
    logical[MASK5] = np.array(ROWS, "float32")[MASK5]
    assert np.array_equal(qr.velocity.numpy(), logical)


def test_write_mask_all_false_is_noop():
    """Scalar with zero selected rows is a genuine no-op (numpy broadcasts (0, 2) from a scalar)."""
    _, qr = _world(multipool=True)
    before = qr.velocity.numpy().copy()
    qr.velocity[np.zeros(5, bool)] = 99.0
    assert np.array_equal(qr.velocity.numpy(), before)


def test_write_mask_all_false_wrong_shape_raises_like_numpy():
    """numpy: `a[zeros(5, bool)] = full((3, 2))` is a (3,2)-into-(0,2) broadcast error -> ValueError."""
    _, qr = _world(multipool=True)
    before = qr.velocity.numpy().copy()
    with pytest.raises(ValueError):
        qr.velocity[np.zeros(5, bool)] = np.full((3, 2), 99.0, "float32")
    assert np.array_equal(qr.velocity.numpy(), before)   # rejected, so still byte-identical


def test_write_mask_matches_numpy_single_pool():
    _, qr = _world(multipool=False)
    qr.velocity[MASK5] = 1.0
    logical = np.array(ROWS, "float32")
    logical[MASK5] = 1.0
    assert np.array_equal(qr.velocity.numpy(), logical)


# --- RAISES: bad masks/values reject EAGERLY and write NOTHING ------------------------------------------


def test_read_mask_non_bool_raises():
    _, qr = _world(multipool=True)
    with pytest.raises(TypeError):
        qr.velocity[np.array([0, 1, 0, 1, 0])]                 # ints, not bool


def test_read_mask_wrong_length_raises():
    _, qr = _world(multipool=True)
    with pytest.raises(ValueError):
        qr.velocity[np.array([True, False, True])]


def test_write_mask_wrong_value_shape_raises_and_writes_nothing():
    _, qr = _world(multipool=True)
    before = qr.velocity.numpy().copy()
    with pytest.raises(ValueError):
        qr.velocity[MASK5] = np.full((4, 2), 1.0, "float32")   # 4 rows for a 3-row mask
    assert np.array_equal(qr.velocity.numpy(), before)         # byte-identical pool on rejection


def test_write_mask_wrong_length_raises_and_writes_nothing():
    _, qr = _world(multipool=True)
    before = qr.velocity.numpy().copy()
    with pytest.raises(ValueError):
        qr.velocity[np.array([True, False, True])] = 1.0
    assert np.array_equal(qr.velocity.numpy(), before)


def test_write_mask_non_bool_raises():
    _, qr = _world(multipool=True)
    with pytest.raises(TypeError):
        qr.velocity[np.array([0, 1, 0, 1, 0])] = 1.0
