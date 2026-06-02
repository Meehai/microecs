"""Unit tests for ecs.Pool"""
import numpy as np
import pytest

from ecs import Pool


def _pool_pos_vel() -> Pool:
    return Pool(
        fields=["position", "velocity"],
        shapes=[(2,), (2,)],
        dtypes=["float32", "float32"],
    )


def test_add_single_entity():
    pool = _pool_pos_vel()
    pool.add_entity(
        position=np.array([1.0, 2.0], "float32"),
        velocity=np.array([3.0, 4.0], "float32"),
    )
    assert len(pool) == 1
    assert pool.position[0].tolist() == [1.0, 2.0]
    assert pool.velocity[0].tolist() == [3.0, 4.0]


def test_add_multiple_entities():
    pool = _pool_pos_vel()
    for i in range(5):
        pool.add_entity(
            position=np.array([float(i), 0.0], "float32"),
            velocity=np.zeros(2, "float32"),
        )
    assert len(pool) == 5
    assert pool.position[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_remove_swaps_tail_into_slot():
    pool = _pool_pos_vel()
    for i in range(3):
        pool.add_entity(
            position=np.array([float(i), 0.0], "float32"),
            velocity=np.zeros(2, "float32"),
        )
    pool.remove_entity(0)  # tail (i=2) should swap into slot 0
    assert len(pool) == 2
    assert pool.position[0, 0] == 2.0
    assert pool.position[1, 0] == 1.0


def test_dynamic_grow_preserves_data():
    """Forces multiple growths and verifies every entity survives intact."""
    pool = _pool_pos_vel()
    n = Pool.INITIAL_CAPACITY * 3  # triggers at least two growths
    for i in range(n):
        pool.add_entity(
            position=np.array([float(i), float(i) * 2], "float32"),
            velocity=np.zeros(2, "float32"),
        )
    assert len(pool) == n
    assert pool.capacity >= n
    for i in range(n):
        assert pool.position[i, 0] == float(i)
        assert pool.position[i, 1] == float(i) * 2


def test_dynamic_shrink_halves_capacity():
    pool = _pool_pos_vel()
    for i in range(Pool.INITIAL_CAPACITY + 1):  # grows: 100 -> 200
        pool.add_entity(
            position=np.array([float(i), 0.0], "float32"),
            velocity=np.zeros(2, "float32"),
        )
    assert pool.capacity == Pool.INITIAL_CAPACITY * 2
    while len(pool) > Pool.INITIAL_CAPACITY * 2 // 4:
        pool.remove_entity(len(pool) - 1)
    pool.remove_entity(len(pool) - 1)  # crosses size < capacity/4
    assert pool.capacity == Pool.INITIAL_CAPACITY


def test_add_missing_field_raises():
    pool = _pool_pos_vel()
    with pytest.raises(KeyError):
        pool.add_entity(position=np.array([1.0, 2.0], "float32"))


def test_add_wrong_shape_raises():
    pool = _pool_pos_vel()
    with pytest.raises(AssertionError):
        pool.add_entity(
            position=np.array([1.0, 2.0, 3.0], "float32"),
            velocity=np.zeros(2, "float32"),
        )


def test_add_wrong_dtype_raises():
    pool = _pool_pos_vel()
    with pytest.raises(AssertionError):
        pool.add_entity(
            position=np.array([1.0, 2.0], "float64"),
            velocity=np.zeros(2, "float32"),
        )


def test_remove_oob_raises():
    pool = _pool_pos_vel()
    pool.add_entity(
        position=np.array([1.0, 2.0], "float32"),
        velocity=np.zeros(2, "float32"),
    )
    with pytest.raises(AssertionError):
        pool.remove_entity(5)
