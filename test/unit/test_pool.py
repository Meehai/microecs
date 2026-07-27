"""Unit tests for ecs.Pool"""
import numpy as np
import pytest

from microecs import Pool
from microecs.pool import POOL_RESERVED_NAMES, _POOL_INTERNAL_ATTRS


def _pool_pos_vel() -> Pool:
    return Pool(
        fields=["position", "velocity"],
        shapes=[(2,), (2,)],
        dtypes=["float32", "float32"],
    )


def test_add_single_entity():
    pool = _pool_pos_vel()
    pool.add_entity({
        "position": np.array([1.0, 2.0], "float32"),
        "velocity": np.array([3.0, 4.0], "float32")})
    assert len(pool) == 1
    assert pool.position[0].tolist() == [1.0, 2.0]
    assert pool.velocity[0].tolist() == [3.0, 4.0]


def test_add_multiple_entities():
    pool = _pool_pos_vel()
    for i in range(5):
        pool.add_entity({
            "position": np.array([float(i), 0.0], "float32"),
            "velocity": np.zeros(2, "float32")})
    assert len(pool) == 5
    assert pool.position[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_remove_swaps_tail_into_slot():
    pool = _pool_pos_vel()
    for i in range(3):
        pool.add_entity({
            "position": np.array([float(i), 0.0], "float32"),
            "velocity": np.zeros(2, "float32")})
    pool.remove_entity(0)  # tail (i=2) should swap into slot 0
    assert len(pool) == 2
    assert pool.position[0, 0] == 2.0
    assert pool.position[1, 0] == 1.0


def test_pop_returns_removed_entity_data():
    pool = _pool_pos_vel()
    pool.add_entity({
        "position": np.array([1.0, 2.0], "float32"),
        "velocity": np.array([3.0, 4.0], "float32")})
    popped = pool.pop_entity(0)
    assert len(pool) == 0
    assert popped["position"].tolist() == [1.0, 2.0]
    assert popped["velocity"].tolist() == [3.0, 4.0]


def test_pop_swaps_tail_into_slot():
    """pop returns the data at the index, then the tail fills that slot (same swap-remove as remove_entity)."""
    pool = _pool_pos_vel()
    for i in range(3):
        pool.add_entity({
            "position": np.array([float(i), 0.0], "float32"),
            "velocity": np.zeros(2, "float32")})
    popped = pool.pop_entity(0)  # returns slot 0 (i=0); tail (i=2) swaps into slot 0
    assert popped["position"][0] == 0.0
    assert len(pool) == 2
    assert pool.position[0, 0] == 2.0
    assert pool.position[1, 0] == 1.0


def test_pop_returns_independent_copy():
    """Popped data is a copy: reusing the freed slot must not mutate what pop returned."""
    pool = _pool_pos_vel()
    pool.add_entity({"position": np.array([1.0, 2.0], "float32"), "velocity": np.zeros(2, "float32")})
    pool.add_entity({"position": np.array([5.0, 6.0], "float32"), "velocity": np.zeros(2, "float32")})
    popped = pool.pop_entity(0)                          # returns [1,2]; tail [5,6] swaps into slot 0
    pool.position[0] = np.array([9.0, 9.0], "float32")   # overwrite the reused slot
    assert popped["position"].tolist() == [1.0, 2.0]     # copy is unaffected


def test_pop_oob_raises():
    """pop_entity delegates the bounds check to remove_entity, so it rejects the same way (task 34)."""
    pool = _pool_pos_vel()
    pool.add_entity({"position": np.array([1.0, 2.0], "float32"), "velocity": np.zeros(2, "float32")})
    with pytest.raises(IndexError, match="OOB"):
        pool.pop_entity(5)


def test_dynamic_grow_preserves_data():
    """Forces multiple growths and verifies every entity survives intact."""
    pool = _pool_pos_vel()
    n = Pool.INITIAL_CAPACITY * 3  # triggers at least two growths
    for i in range(n):
        pool.add_entity({
            "position": np.array([float(i), float(i) * 2], "float32"),
            "velocity": np.zeros(2, "float32")})
    assert len(pool) == n
    assert pool.capacity >= n
    for i in range(n):
        assert pool.position[i, 0] == float(i)
        assert pool.position[i, 1] == float(i) * 2


def test_dynamic_shrink_halves_capacity():
    pool = _pool_pos_vel()
    for i in range(Pool.INITIAL_CAPACITY + 1):  # grows: 100 -> 200
        pool.add_entity({
            "position": np.array([float(i), 0.0], "float32"),
            "velocity": np.zeros(2, "float32")})
    assert pool.capacity == Pool.INITIAL_CAPACITY * 2
    while len(pool) > Pool.INITIAL_CAPACITY * 2 // 4:
        pool.remove_entity(len(pool) - 1)
    pool.remove_entity(len(pool) - 1)  # crosses size < capacity/4
    assert pool.capacity == Pool.INITIAL_CAPACITY


def test_add_missing_field_raises():
    pool = _pool_pos_vel()
    with pytest.raises(KeyError):
        pool.add_entity({"position": np.array([1.0, 2.0], "float32")})


def test_add_wrong_shape_raises():
    """#49 item 2 turned the pool's three per-field asserts into one `raise ValueError`, so this is a raise
    that survives `python -O` now, not an assert. The message must name the field and both expectations."""
    pool = _pool_pos_vel()
    with pytest.raises(ValueError, match="position"):
        pool.add_entity({
            "position": np.array([1.0, 2.0, 3.0], "float32"),
            "velocity": np.zeros(2, "float32")})


def test_add_wrong_dtype_raises():
    """The dtype half of the same guard. This is the case that catches the walrus/short-circuit trap: with
    `if (dtp := ...) != dt or (shp := ...) != shape`, a dtype mismatch makes the first operand True, `or`
    never evaluates the second, and the message's `{shp}` blows up with UnboundLocalError instead of raising
    ValueError -- i.e. the wrong-dtype path, the one thing this guard exists for, is the one that misfires."""
    pool = _pool_pos_vel()
    with pytest.raises(ValueError, match="position"):
        pool.add_entity({
            "position": np.array([1.0, 2.0], "float64"),
            "velocity": np.zeros(2, "float32")})


def test_add_bad_field_error_message_reports_both_dtype_and_shape():
    """Whatever the check's internal form, the message has to carry both facts for either kind of mismatch --
    that is the whole reason the walrus bindings are in the condition. Pins it from the outside."""
    pool = _pool_pos_vel()
    for bad, why in [(np.array([1.0, 2.0], "float64"), "dtype"), (np.array([1.0, 2.0, 3.0], "float32"), "shape")]:
        with pytest.raises(ValueError) as ex:
            pool.add_entity({"position": bad, "velocity": np.zeros(2, "float32")})
        assert "float32" in str(ex.value) and "(2,)" in str(ex.value), f"{why}: {ex.value}"


def test_remove_oob_raises():
    """An out-of-bounds index is caller error on a public method, so it's a real `raise` (task 34), not an
    assert `python -O` would strip. Measured with the check removed (test/manual/34-assert-sweep): the call
    succeeds, a never-live row gets written and `size` goes to -1, after which `len(pool)` itself blows up --
    a corrupt pool complaining somewhere else entirely. Once per removal, so the check is not hot."""
    pool = _pool_pos_vel()
    pool.add_entity({
        "position": np.array([1.0, 2.0], "float32"),
        "velocity": np.zeros(2, "float32")})
    with pytest.raises(IndexError, match="OOB"):
        pool.remove_entity(5)


@pytest.mark.parametrize("bad_index", [-1, -3, -4])
def test_remove_negative_index_raises_and_changes_nothing(bad_index):
    """The bound has a bottom too: a negative index would pass `entity_index >= self.size` and then index numpy
    from the other end, swap-removing the TAIL -- the caller asks to drop an entity that doesn't exist and a
    different one disappears instead. Measured (3 entities, `remove_entity(-7)`): no error, [0,1,2] -> [0,1].
    Assert on the contents, not just `len` -- the size does change, which is what makes it look like it worked."""
    pool = _pool_pos_vel()
    for i in range(3):
        pool.add_entity({"position": np.array([float(i), 0.0], "float32"), "velocity": np.zeros(2, "float32")})

    with pytest.raises(IndexError, match="OOB"):
        pool.remove_entity(bad_index)
    assert pool.position[:, 0].tolist() == [0.0, 1.0, 2.0]  # nothing removed, nothing shuffled


def test_pop_negative_index_raises_and_changes_nothing():
    """pop_entity has the same hole, plus it reads (and would return) the row before the check runs."""
    pool = _pool_pos_vel()
    for i in range(3):
        pool.add_entity({"position": np.array([float(i), 0.0], "float32"), "velocity": np.zeros(2, "float32")})

    with pytest.raises(IndexError, match="OOB"):
        pool.pop_entity(-1)
    assert pool.position[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_rebind_field_raises_and_keeps_storage():
    """Rebinding a field (pool.position = ...) must raise instead of silently detaching from SoA storage."""
    pool = _pool_pos_vel()
    pool.add_entity({
        "position": np.array([1.0, 2.0], "float32"),
        "velocity": np.zeros(2, "float32")})
    with pytest.raises(ValueError):
        pool.position = np.array([[9.0, 9.0]], "float32")
    assert pool.position[0].tolist() == [1.0, 2.0]  # storage untouched, no shadow attribute

    pool.position[:] = np.array([[3.0, 4.0]], "float32")  # in-place write still goes through
    assert pool.position[0].tolist() == [3.0, 4.0]


# Pool resolves data fields through __getattr__, which Python only calls AFTER normal lookup fails -- so a field
# named like ANY member of Pool (instance attr or method) is shadowed by that member and unreachable except via
# pool.data[...]. POOL_RESERVED_NAMES is both halves: the _POOL_INTERNAL_ATTRS attrs plus the class dict.
@pytest.mark.parametrize("reserved", sorted(POOL_RESERVED_NAMES))
def test_reserved_field_names_raise(reserved):
    """Field names clashing with a Pool member must be rejected at construction, not fail cryptically later.
    A field list is user input, so this is a `raise` (survives `python -O`), not an `assert`. The method half was
    measured broken: Pool(fields=['add_entity'], ...) was accepted, then `pool.add_entity` returned the bound
    method and the column was reachable only via `pool.data['add_entity']`."""
    with pytest.raises(ValueError):
        Pool(fields=[reserved], shapes=[(1,)], dtypes=["float32"])


def test_reserved_name_mixed_with_valid_raises():
    with pytest.raises(ValueError):
        Pool(fields=["position", "size"], shapes=[(2,), (1,)], dtypes=["float32", "float32"])


def test_internal_attrs_cover_every_init_attribute():
    """_POOL_INTERNAL_ATTRS must be EXACTLY the attrs __init__ creates. Unlike Entity/QueryResult, nothing inside
    Pool reads it at runtime (__setattr__ guards via the data dict), so a new attr left out of it breaks nothing
    here -- it silently drops out of POOL_RESERVED_NAMES and a component may then be named after it. This is the
    only thing standing between that and a shadowed field. `fields_set` (task 15 part B) is why it exists."""
    pool = _pool_pos_vel()

    assert set(vars(pool)) == _POOL_INTERNAL_ATTRS
    assert _POOL_INTERNAL_ATTRS <= POOL_RESERVED_NAMES


def test_mismatched_field_shape_dtype_lengths_raise():
    """fields/shapes/dtypes are three parallel lists -- a length mismatch means the caller lost track of which
    shape belongs to which field. Rejected at construction (a `raise`: zip() would silently truncate)."""
    with pytest.raises(ValueError, match="Lens not match"):
        Pool(fields=["position", "velocity"], shapes=[(2,)], dtypes=["float32", "float32"])


def test_object_dtype_stores_python_objects_by_reference():
    """An object-dtype field holds arbitrary Python objects, stored by reference (not copied)."""
    pool = Pool(fields=["payload"], shapes=[(1,)], dtypes=["object"])
    obj = {"hp": 10}
    pool.add_entity({"payload": np.array([obj], dtype=object)})
    assert pool.payload.dtype == object
    assert pool.payload[0, 0] is obj  # the exact same object came back, not a copy


def test_object_dtype_pop_returns_same_object():
    """pop_entity hands back the stored object reference unchanged."""
    pool = Pool(fields=["payload"], shapes=[(1,)], dtypes=["object"])
    obj = object()
    pool.add_entity({"payload": np.array([obj], dtype=object)})
    popped = pool.pop_entity(0)
    assert popped["payload"][0] is obj


def test_fields_set_mirrors_fields_for_o1_membership():
    """task 15 part B: Pool exposes fields_set (a set) alongside the ordered fields list, so membership checks
    (used on the per-entity Entity path) are O(1). It must hold exactly the same names as the ordered list."""
    pool = _pool_pos_vel()
    assert pool.fields_set == set(pool.fields)                     # same names, set form
    assert pool.fields == ["position", "velocity"]                # ordered list stays public (serialization order)
    assert "position" in pool.fields_set and "missing" not in pool.fields_set
