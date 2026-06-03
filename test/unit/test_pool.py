"""Unit tests for ecs.Pool"""
import numpy as np
import pytest

from microecs import Pool


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


def test_pop_returns_removed_entity_data():
    pool = _pool_pos_vel()
    pool.add_entity(
        position=np.array([1.0, 2.0], "float32"),
        velocity=np.array([3.0, 4.0], "float32"),
    )
    popped = pool.pop_entity(0)
    assert len(pool) == 0
    assert popped["position"].tolist() == [1.0, 2.0]
    assert popped["velocity"].tolist() == [3.0, 4.0]


def test_pop_swaps_tail_into_slot():
    """pop returns the data at the index, then the tail fills that slot (same swap-remove as remove_entity)."""
    pool = _pool_pos_vel()
    for i in range(3):
        pool.add_entity(
            position=np.array([float(i), 0.0], "float32"),
            velocity=np.zeros(2, "float32"),
        )
    popped = pool.pop_entity(0)  # returns slot 0 (i=0); tail (i=2) swaps into slot 0
    assert popped["position"][0] == 0.0
    assert len(pool) == 2
    assert pool.position[0, 0] == 2.0
    assert pool.position[1, 0] == 1.0


def test_pop_returns_independent_copy():
    """Popped data is a copy: reusing the freed slot must not mutate what pop returned."""
    pool = _pool_pos_vel()
    pool.add_entity(position=np.array([1.0, 2.0], "float32"), velocity=np.zeros(2, "float32"))
    pool.add_entity(position=np.array([5.0, 6.0], "float32"), velocity=np.zeros(2, "float32"))
    popped = pool.pop_entity(0)                          # returns [1,2]; tail [5,6] swaps into slot 0
    pool.position[0] = np.array([9.0, 9.0], "float32")   # overwrite the reused slot
    assert popped["position"].tolist() == [1.0, 2.0]     # copy is unaffected


def test_pop_oob_raises():
    pool = _pool_pos_vel()
    pool.add_entity(position=np.array([1.0, 2.0], "float32"), velocity=np.zeros(2, "float32"))
    with pytest.raises(AssertionError):
        pool.pop_entity(5)


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


def test_rebind_field_raises_and_keeps_storage():
    """Rebinding a field (pool.position = ...) must raise instead of silently detaching from SoA storage."""
    pool = _pool_pos_vel()
    pool.add_entity(
        position=np.array([1.0, 2.0], "float32"),
        velocity=np.zeros(2, "float32"),
    )
    with pytest.raises(ValueError):
        pool.position = np.array([[9.0, 9.0]], "float32")
    assert pool.position[0].tolist() == [1.0, 2.0]  # storage untouched, no shadow attribute

    pool.position[:] = np.array([[3.0, 4.0]], "float32")  # in-place write still goes through
    assert pool.position[0].tolist() == [3.0, 4.0]


def test_reserved_field_names_raise():
    """Field names clashing with Pool internals must be rejected at construction, not fail cryptically later."""
    for reserved in Pool.RESERVED_NAMES:
        with pytest.raises(AssertionError):
            Pool(fields=[reserved], shapes=[(1,)], dtypes=["float32"])


def test_reserved_name_mixed_with_valid_raises():
    with pytest.raises(AssertionError):
        Pool(fields=["position", "size"], shapes=[(2,), (1,)], dtypes=["float32", "float32"])


def test_object_dtype_stores_python_objects_by_reference():
    """An object-dtype field holds arbitrary Python objects, stored by reference (not copied)."""
    pool = Pool(fields=["payload"], shapes=[(1,)], dtypes=["object"])
    obj = {"hp": 10}
    pool.add_entity(payload=np.array([obj], dtype=object))
    assert pool.payload.dtype == object
    assert pool.payload[0, 0] is obj  # the exact same object came back, not a copy


def test_object_dtype_pop_returns_same_object():
    """pop_entity hands back the stored object reference unchanged."""
    pool = Pool(fields=["payload"], shapes=[(1,)], dtypes=["object"])
    obj = object()
    pool.add_entity(payload=np.array([obj], dtype=object))
    popped = pool.pop_entity(0)
    assert popped["payload"][0] is obj
