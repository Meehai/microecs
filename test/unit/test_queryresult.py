"""Unit tests for QueryResult — a list of pools (+ the queried components) seen as one contiguous run.

QueryResult wraps the pools a query matched, plus the component types that were queried, and makes them
behave like a single sequence of entities. No World here: we build the pools and components directly so
the test pins QueryResult's own behaviour, nothing else.
"""
from dataclasses import field
import numpy as np
import pytest

from microecs import Pool, Component
from microecs.query_result import QueryResult, _Field


class HasPosition(Component):  # owns the `position` field the pools below carry
    position: np.ndarray = field(metadata={"shape": (2,), "dtype": "float32"})


def _pool_with(n: int) -> Pool:
    """A standalone pool of n entities, each with a (2,) float32 `position` starting at [0, 0]."""
    pool = Pool(fields=["position"], shapes=[(2,)], dtypes=["float32"])
    for _ in range(n):
        pool.add_entity(position=np.zeros(2, "float32"))
    return pool


def _moving_pool(n: int, velocity: list[float]) -> Pool:
    """n entities at the origin, each carrying the SAME `velocity` (a (2,) vector)."""
    pool = Pool(fields=["position", "velocity"], shapes=[(2,), (2,)], dtypes=["float32", "float32"])
    for _ in range(n):
        pool.add_entity(position=np.zeros(2, "float32"), velocity=np.array(velocity, "float32"))
    return pool


def _drawable_pool(rows: list[tuple[list[float], float]]) -> Pool:
    """A pool of (position, radius) entities with explicit per-entity values."""
    pool = Pool(fields=["position", "radius"], shapes=[(2,), (1,)], dtypes=["float32", "float32"])
    for xy, r in rows:
        pool.add_entity(position=np.array(xy, "float32"), radius=np.array([r], "float32"))
    return pool


def _pv_pool(rows: list[tuple[list[float], list[float]]]) -> Pool:
    """A pool of (position, velocity) entities with explicit per-entity values."""
    pool = Pool(fields=["position", "velocity"], shapes=[(2,), (2,)], dtypes=["float32", "float32"])
    for xy, v in rows:
        pool.add_entity(position=np.array(xy, "float32"), velocity=np.array(v, "float32"))
    return pool


def _ball_pool(rows: list[tuple[list[float], float]]) -> Pool:
    """A pool of (position, radius, color) entities with explicit position/radius; color starts black."""
    pool = Pool(fields=["position", "radius", "color"], shapes=[(2,), (1,), (4,)],
                dtypes=["float32", "float32", "int32"])
    for xy, r in rows:
        pool.add_entity(position=np.array(xy, "float32"), radius=np.array([r], "float32"),
                        color=np.zeros(4, "int32"))
    return pool


def _pos_pool(rows: list[list[float]]) -> Pool:
    """A pool of entities carrying only an explicit (2,) `position` (no other fields)."""
    pool = Pool(fields=["position"], shapes=[(2,)], dtypes=["float32"])
    for xy in rows:
        pool.add_entity(position=np.array(xy, "float32"))
    return pool


def _all_pairs_collisions(positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """All-pairs collision over flat (N, 2) positions / (N, 1) radii; returns an (N, 1) bool mask of which
    entities overlap at least one other."""
    dists = np.sqrt(((positions[:, None] - positions[None]) ** 2).sum(-1))   # (N, N)
    radii_sum = (radii[None] + radii[:, None])[..., 0]                       # (N, N)
    hit = (dists < radii_sum) - np.eye(len(positions))                       # (N, N), self-pair removed
    return (hit.sum(axis=1) > 0)[..., None]                                  # (N, 1)


_FIELD_SHAPES = {"position": (2,), "velocity": (2,), "radius": (1,), "color": (4,)}
_FIELD_DTYPES = {"position": "float32", "velocity": "float32", "radius": "float32", "color": "int32"}


def _query(pools: list[Pool], *fields: str, entity_ids: list[int] = None) -> QueryResult:
    """A QueryResult over `pools` for the named fields, carrying their (test-fixed) shapes and dtypes. `entity_ids`
    default to 0..N-1 in pool-by-pool row order (World hands out the real ids; here any pool-aligned ids do)."""
    n = sum(len(p) for p in pools)
    ids = np.arange(n, dtype="int64") if entity_ids is None else np.array(entity_ids, dtype="int64")
    return QueryResult(pools, {f: _FIELD_SHAPES[f] for f in fields}, {f: _FIELD_DTYPES[f] for f in fields}, ids)


def test_len_sums_entities_across_pools():
    """len(qr) is the total entity count across all pools (2 + 3 = 5), not the number of pools; entity_ids has
    one entry per entity, so its length tracks len(qr)."""
    qr = _query([_pool_with(2), _pool_with(3)], "position")
    assert len(qr) == 5
    assert len(qr.entity_ids) == 5


def test_field_len_is_total_entities_across_pools():
    """len(qr.position) is the entity count across all pools (2 + 3 = 5): equal to len(qr) and to the number
    of rows iterating the field yields."""
    qr = _query([_pool_with(2), _pool_with(3)], "position")

    assert len(qr.position) == 5
    assert len(qr.position) == len(qr)              # a field's len agrees with the query's len
    assert sum(1 for _ in qr.position) == 5         # and with how many rows iteration hands out


def test_field_write_scatters_per_pool_no_gather():
    """`qr.position[:] = qr.position + 1` computes per pool and writes the result back into each pool.
    Both pools start all-zero and end all-one."""
    pool_of_2 = _pool_with(2)
    pool_of_3 = _pool_with(3)
    qr = _query([pool_of_2, pool_of_3], "position")

    qr.position[:] = qr.position + 1

    assert (pool_of_2.position == 1).all()
    assert (pool_of_3.position == 1).all()


def test_two_field_motion_writes_through_per_pool():
    """`qr.position[:] = qr.position + qr.velocity * 0.5` over two pools with different velocities. Each pool's
    position advances by its own velocity*dt; the two pools end at different positions."""
    slow = _moving_pool(2, [1.0, 0.0])   # +1 in x per unit time
    fast = _moving_pool(3, [0.0, 2.0])   # +2 in y per unit time
    qr = _query([slow, fast], "position", "velocity")

    qr.position[:] = qr.position + qr.velocity * 0.5

    assert (slow.position == [0.5, 0.0]).all()
    assert (fast.position == [0.0, 1.0]).all()


def test_np_where_dispatches_per_pool():
    """`qr.velocity[:] = np.where(qr.velocity > 1.5, -qr.velocity, qr.velocity)` runs per pool and writes back.
    The slow pool ([1, 0]) is below the threshold and is unchanged; the fast pool ([0, 2]) flips its y to [0, -2]."""
    slow = _moving_pool(2, [1.0, 0.0])
    fast = _moving_pool(3, [0.0, 2.0])
    qr = _query([slow, fast], "position", "velocity")

    qr.velocity[:] = np.where(qr.velocity > 1.5, -qr.velocity, qr.velocity)

    assert (slow.velocity == [1.0, 0.0]).all()
    assert (fast.velocity == [0.0, -2.0]).all()


def test_raw_full_n_mask_mixed_with_field_splits_across_pools():
    """`qr.velocity[:] = np.where(mask, -qr.velocity, qr.velocity)` where `mask` is a RAW (N, 2) numpy array
    (not a _Field), over two pools. The mask spans every entity; the velocities are per-pool _Fields. Per-pool
    dispatch splits the full-N mask into per-pool chunks (like a _Field), so each pool's np.where sees its own
    (pool_n, 2) slice of the mask."""
    a = _pv_pool([([0.0, 0.0], [3.0, 3.0])])                       # 1 entity
    b = _pv_pool([([0.0, 0.0], [3.0, 3.0]), ([0.0, 0.0], [3.0, 3.0])])   # 2 entities
    qr = _query([a, b], "position", "velocity")

    mask = np.array([[True, False], [False, True], [True, False]])  # (N=3, 2): flip e0.x, e1.y, e2.x
    qr.velocity[:] = np.where(mask, -qr.velocity, qr.velocity)

    assert a.velocity.tolist() == [[-3.0, 3.0]]                    # e0: x flipped
    assert b.velocity.tolist() == [[3.0, -3.0], [-3.0, 3.0]]       # e1: y flipped, e2: x flipped


def test_inplace_op_with_full_n_raw_operand_splits_across_pools():
    """`qr.velocity += raw` where `raw` is a RAW (N, 2) numpy array, over two pools. The in-place op splits the
    full-N operand into per-pool chunks (like a _Field), so each pool's velocity view is incremented by its own
    (pool_n, 2) slice of `raw`."""
    a = _pv_pool([([0.0, 0.0], [3.0, 3.0])])                       # 1 entity
    b = _pv_pool([([0.0, 0.0], [3.0, 3.0]), ([0.0, 0.0], [3.0, 3.0])])   # 2 entities
    qr = _query([a, b], "position", "velocity")

    bump = np.array([[1.0, 0.0], [0.0, 10.0], [100.0, 0.0]], "float32")  # (N=3, 2): per-entity increment
    qr.velocity += bump

    assert a.velocity.tolist() == [[4.0, 3.0]]                     # e0 += [1, 0]
    assert b.velocity.tolist() == [[3.0, 13.0], [103.0, 3.0]]      # e1 += [0, 10], e2 += [100, 0]


def test_unsupported_ops_are_rejected():
    """Operations with no per-pool meaning raise instead of returning a wrong result: a cross-pool reduction
    (np.sum) and a masked/partial write (qr.velocity[mask] = 0) both raise.

    Note: the whitelist in __array_function__ was removed -- any numpy func now runs per-pool and is policed by
    the `len(result) == rows` guard in _apply_fn_on_parts. A full reduction yields a scalar, so that guard's
    `len(...)` raises TypeError today (not the old NotImplemented->TypeError). A clean AssertionError would need
    an `ndim >= 1` guard -- see test_field_shape_invariant.py::test_op_breaking_entity_axis_raises."""
    qr = _query([_moving_pool(2, [1.0, 0.0]), _moving_pool(3, [0.0, 2.0])], "position", "velocity")

    with pytest.raises((TypeError, AssertionError)):  # reduction collapses the entity axis -> guard rejects it
        np.sum(qr.velocity)

    with pytest.raises(Exception):     # partial write; only whole `[:]` assignment is allowed
        qr.velocity[np.array([True, False, True, False, True])] = 0


def test_inplace_op_writes_through_to_pools():
    """`qr.velocity *= 0.5` mutates the underlying pools in place. Both pools start with non-zero velocity;
    after the halving every entity's velocity is halved."""
    slow = _moving_pool(2, [10.0, 0.0])
    fast = _moving_pool(3, [0.0, 20.0])
    qr = _query([slow, fast], "position", "velocity")

    qr.velocity *= 0.5
    qr.velocity *= np.array([1])
    qr.velocity[:] = qr.velocity * np.array([1])
    qr.velocity = qr.velocity * np.array([1])

    assert (slow.velocity == [5.0, 0.0]).all()
    assert (fast.velocity == [0.0, 10.0]).all()


def test_inplace_op_with_raw_operand_writes_through_and_matches_numpy_on_bad_shapes():
    """`qr.position *= raw` with a RAW numpy operand writes through the pools: the operand broadcasts by numpy's
    rules and the value Python rebinds is the in-place _Field result (not the raw array), so a raw operand never
    looks like a stray field assignment. A non-broadcastable operand -- np.array([]) against (N, 2) -- raises
    ValueError, exactly as `np.zeros((N, 2)) *= np.array([])` does."""
    a = _pos_pool([[10.0, 20.0]])
    b = _pos_pool([[30.0, 40.0], [50.0, 60.0]])
    qr = _query([a, b], "position")

    qr.position *= np.array([2.0, 3.0], "float32")          # raw (2,) row broadcasts onto every entity
    assert a.position.tolist() == [[20.0, 60.0]]
    assert b.position.tolist() == [[60.0, 120.0], [100.0, 180.0]]

    with pytest.raises(ValueError):                         # (N, 2) *= (0,) can't broadcast -> numpy raises, so do we
        qr.position *= np.array([], "float32")


def test_iterating_fields_yields_each_entity_for_rendering():
    """Iterating fields with zip yields one entity per step across all pools in pool-by-pool order, with fields
    aligned -- and `entity_ids` lines up too, so a render/id loop can `zip(qr.entity_ids, qr.position, qr.radius)`
    and get entity k's id with entity k's data."""
    poolA = _drawable_pool([([0.0, 0.0], 5.0), ([1.0, 1.0], 6.0)])
    poolB = _drawable_pool([([2.0, 2.0], 7.0), ([3.0, 3.0], 8.0), ([4.0, 4.0], 9.0)])
    qr = _query([poolA, poolB], "position", "radius", entity_ids=[10, 11, 20, 21, 22])  # A's ids, then B's

    drawn = [(eid.item(), pos.tolist(), rad.item())
             for eid, pos, rad in zip(qr.entity_ids, qr.position, qr.radius)]

    assert drawn == [(10, [0.0, 0.0], 5.0), (11, [1.0, 1.0], 6.0),
                     (20, [2.0, 2.0], 7.0), (21, [3.0, 3.0], 8.0), (22, [4.0, 4.0], 9.0)]


def test_entity_ids_is_a_flat_pool_by_pool_array():
    """`qr.entity_ids` is a flat (N,) array over all matched entities, in the same pool-by-pool order as the
    fields -- NOT a _Field. So entity-axis ops that _Field rejects (integer index, slice, np.isin) work here,
    because the ids are materialized, not a stitched per-pool view."""
    poolA = _drawable_pool([([0.0, 0.0], 5.0), ([1.0, 1.0], 6.0)])   # 2 entities
    poolB = _drawable_pool([([2.0, 2.0], 7.0)])                      # 1 entity
    qr = _query([poolA, poolB], "position", "radius", entity_ids=[10, 11, 20])

    assert isinstance(qr.entity_ids, np.ndarray)
    assert qr.entity_ids.shape == (len(qr),)                         # flat, one entry per entity (== 3)
    assert qr.entity_ids.tolist() == [10, 11, 20]                    # pool A's ids first, then pool B's
    assert int(qr.entity_ids[2]) == 20                               # entity-axis index -> allowed (unlike a _Field)
    assert qr.entity_ids[0:2].tolist() == [10, 11]                   # slicing the entity axis -> allowed
    assert np.isin(qr.entity_ids, [11, 20]).tolist() == [False, True, True]   # set membership -> allowed


def test_component_axis_index_reads_a_per_pool_field():
    """`qr.position[:, 0]` selects one component of every entity and returns a _Field of per-pool column views.
    Checked via .parts: the x column and the y column, each split by pool."""
    poolA = _drawable_pool([([0.0, 10.0], 5.0), ([1.0, 11.0], 6.0)])
    poolB = _drawable_pool([([2.0, 12.0], 7.0), ([3.0, 13.0], 8.0), ([4.0, 14.0], 9.0)])
    qr = _query([poolA, poolB], "position", "radius")

    xs = qr.position[:, 0]
    ys = qr.position[:, 1]

    assert [p.tolist() for p in xs.parts] == [[0.0, 1.0], [2.0, 3.0, 4.0]]
    assert [p.tolist() for p in ys.parts] == [[10.0, 11.0], [12.0, 13.0, 14.0]]


def test_component_axis_index_writes_through_one_column():
    """`qr.position[:, 1] = -1.0` writes one component of every entity across pools and leaves the other
    component untouched."""
    poolA = _drawable_pool([([0.0, 10.0], 5.0), ([1.0, 11.0], 6.0)])
    poolB = _drawable_pool([([2.0, 12.0], 7.0), ([3.0, 13.0], 8.0), ([4.0, 14.0], 9.0)])
    qr = _query([poolA, poolB], "position", "radius")

    qr.position[:, 1] = -1.0

    assert (poolA.position[:, 0] == [0.0, 1.0]).all()        # x untouched
    assert (poolB.position[:, 0] == [2.0, 3.0, 4.0]).all()
    assert (poolA.position[:, 1] == -1.0).all()              # y written
    assert (poolB.position[:, 1] == -1.0).all()


def test_entity_axis_index_stays_rejected():
    """Indexing the entity axis with a range raises: a slice (`qr.position[0:1]`) or a boolean mask both cross
    pool boundaries, so both raise TypeError. (A bare integer is a single entity in one pool, and is allowed.)"""
    qr = _query([_drawable_pool([([0.0, 0.0], 5.0)]), _drawable_pool([([1.0, 1.0], 6.0)])],
                "position", "radius")

    with pytest.raises(TypeError):
        qr.position[0:1]                        # a range of entities -> spans pools
    with pytest.raises(TypeError):
        qr.position[np.array([True, False])]    # a mask over entities


def test_single_entity_read_routes_index_to_right_pool():
    """`qr.position[i]` for a bare integer must return entity i's row, out of whichever pool it lives in. Two
    pools (2 entities, then 3); each entity's position encodes its OWN global index, so position[i] == [i, i]
    proves index i routed to the right pool and offset. Indices 0,1 -> pool A; 2,3,4 -> pool B; -1 -> last."""
    a = _pos_pool([[0.0, 0.0], [1.0, 1.0]])                 # global indices 0, 1
    b = _pos_pool([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])     # global indices 2, 3, 4
    qr = _query([a, b], "position")                          # 5 entities stitched across two pools

    for i in range(5):
        assert qr.position[i].tolist() == [float(i), float(i)]   # value == index -> routed to the right entity

    assert qr.position[-1].tolist() == [4.0, 4.0]           # numpy-parity negative index -> last entity
    assert qr.position.numpy()[3].tolist() == qr.position[3].tolist()   # agrees with the gathered (N, 2) array


def test_single_entity_read_returns_a_writeable_view():
    """`qr.position[i]` is a view into the entity's pool, not a copy: writing through it mutates the pool."""
    a = _pos_pool([[0.0, 0.0], [1.0, 1.0]])
    b = _pos_pool([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    qr = _query([a, b], "position")

    qr.position[3][:] = [9.0, 9.0]                  # entity 3 -> pool b, local index 1
    assert b.position[1].tolist() == [9.0, 9.0]     # the write reached the pool
    assert qr.position[3].tolist() == [9.0, 9.0]    # and re-reading the same entity sees it


def test_single_entity_read_positive_out_of_range_raises():
    """A bare-int read past the end raises IndexError, like numpy on an (N, *) array -- it does not wrap around
    to a valid entity."""
    a = _pos_pool([[0.0, 0.0], [1.0, 1.0]])
    b = _pos_pool([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    qr = _query([a, b], "position")                 # len 5; valid indices -5..4

    with pytest.raises(IndexError):
        qr.position[5]                               # one past the end
    with pytest.raises(IndexError):
        qr.position[99]


def test_single_entity_read_on_empty_query_raises():
    """A bare-int read on a zero-entity query raises IndexError, exactly like `np.empty((0, 2))[0]`."""
    qr = _query([], "position")                     # no pools -> (0, 2) block, len 0
    assert len(qr) == 0

    with pytest.raises(IndexError):
        qr.position[0]


def test_single_entity_read_negative_out_of_range_raises_like_numpy():
    """A bare-int read before the start (`qr.position[-N-1]`) raises IndexError, like numpy -- it does not wrap
    around to the tail."""
    a = _pos_pool([[0.0, 0.0], [1.0, 1.0]])
    b = _pos_pool([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    qr = _query([a, b], "position")                 # len 5; valid indices -5..4

    with pytest.raises(IndexError):
        qr.position[-6]                              # one before the start
    with pytest.raises(IndexError):
        qr.position[-100]


def test_wallbounce_per_axis_via_column_indexing():
    """Flip the x-velocity of entities whose x is past a wall, operating on the x column. The entity at x=0
    flips its vx to -3; the one at x=5 keeps +3. Combines a component-axis read, np.where, and a component-axis
    write across two pools."""
    at_wall = _pv_pool([([0.0, 50.0], [3.0, 0.0])])     # x=0 -> past the left wall
    inside = _pv_pool([([5.0, 50.0], [3.0, 0.0])])      # x=5 -> inside
    qr = _query([at_wall, inside], "position", "velocity")

    hit_x = qr.position[:, 0] < 1.0
    qr.velocity[:, 0] = np.where(hit_x, -qr.velocity[:, 0], qr.velocity[:, 0])

    assert (at_wall.velocity[:, 0] == -3.0).all()       # bounced
    assert (inside.velocity[:, 0] == 3.0).all()         # unchanged


def test_gather_concatenates_all_entities_for_cross_entity_ops():
    """`qr.position.numpy()` concatenates every pool's view into one (N, 2) array in pool-by-pool order."""
    poolA = _drawable_pool([([0.0, 0.0], 5.0), ([1.0, 1.0], 6.0)])
    poolB = _drawable_pool([([2.0, 2.0], 7.0)])
    qr = _query([poolA, poolB], "position", "radius")

    flat = qr.position.numpy()

    assert flat.shape == (3, 2)
    assert flat.tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]     # pool-by-pool order


def test_numpy_single_pool_is_zero_copy_view():
    """`numpy()` over a SINGLE pool must hand back that pool's own view, not a copy -- the docstring promises
    'for len==1, we return the same object'. The single-archetype query is the common case, so gathering a
    contiguous (N, *) array must not pay a full copy. Pinned via np.shares_memory with the underlying part:
    the exact, dtype/layout-independent no-copy check."""
    qr = _query([_drawable_pool([([0.0, 0.0], 5.0), ([1.0, 1.0], 6.0)])], "position", "radius")
    field = qr.position
    assert len(field.parts) == 1                                    # premise: one pool -> one part

    flat = field.numpy()

    assert flat.tolist() == [[0.0, 0.0], [1.0, 1.0]]                # right values, copy or not
    assert np.shares_memory(flat, field.parts[0])                   # ... and the SAME memory: no copy


def test_numpy_multi_pool_is_a_fresh_concatenation():
    """The single-pool no-copy shortcut must stay scoped to len==1: with TWO pools there is no single contiguous
    buffer to view, so numpy() necessarily allocates a fresh (N, *) array. Pinned so a future 'always return
    self.parts[0]' shortcut -- which would silently drop pool B's rows -- can't slip in: the gathered array
    shares memory with NEITHER pool's view."""
    qr = _query([_pos_pool([[0.0, 0.0]]), _pos_pool([[1.0, 1.0]])], "position")
    field = qr.position
    assert len(field.parts) == 2                                    # premise: two pools -> two parts

    flat = field.numpy()

    assert flat.tolist() == [[0.0, 0.0], [1.0, 1.0]]                # both pools' rows, pool-by-pool order
    assert not np.shares_memory(flat, field.parts[0])              # a real concatenation, not pool A's view
    assert not np.shares_memory(flat, field.parts[1])              # ... nor pool B's


def test_collision_round_trips_via_gather_single_archetype():
    """One pool of three balls, two overlapping and one far. gather() feeds the all-pairs collision a contiguous
    (N, 2)/(N, 1); the (N, 1) mask writes back via `qr.color[:] = np.where(mask, RED, BLACK)`. The two
    overlapping balls become red, the lone ball stays black."""
    RED, BLACK = np.array([255, 0, 0, 255], "int32"), np.array([0, 0, 0, 255], "int32")
    pool = _ball_pool([([0.0, 0.0], 5.0), ([1.0, 0.0], 5.0), ([100.0, 100.0], 5.0)])
    qr = _query([pool], "position", "radius", "color")

    hit = _all_pairs_collisions(qr.position.numpy(), qr.radius.numpy())    # (3, 1) flat global mask
    qr.color[:] = np.where(hit, RED, BLACK)

    assert (pool.color[0] == RED).all()     # the two overlapping balls
    assert (pool.color[1] == RED).all()
    assert (pool.color[2] == BLACK).all()   # the lone ball, untouched


def test_cross_archetype_collision_round_trips_via_positional_writeback():
    """Two overlapping balls in separate pools. gather() places every entity adjacent so the all-pairs collision
    detects the cross-pool pair; the (N, 4) colour result writes back with `qr.color[:] = ...`, splitting
    positionally across both pools. Both balls become red."""
    RED, BLACK = np.array([255, 0, 0, 255], "int32"), np.array([0, 0, 0, 255], "int32")
    a = _ball_pool([([0.0, 0.0], 5.0)])
    b = _ball_pool([([1.0, 0.0], 5.0)])
    qr = _query([a, b], "position", "radius", "color")

    hit = _all_pairs_collisions(qr.position.numpy(), qr.radius.numpy())    # (2, 1), catches the cross-pool pair
    qr.color[:] = np.where(hit, RED, BLACK)                                  # (2, 4) -> splits across both pools

    assert (a.color[0] == RED).all()        # cross-archetype neighbour detected AND written back
    assert (b.color[0] == RED).all()


def test_contiguous_array_writes_back_positionally_across_pools():
    """`qr.position[:] = block` assigns an (N, 2) array row i to entity i, split across pools by order (first 2
    rows to pool A, next 3 to pool B). Reversing the block then reverses every entity's position."""
    a, b = _pool_with(2), _pool_with(3)
    qr = _query([a, b], "position")

    block = np.arange(5 * 2, dtype="float32").reshape(5, 2)     # [[0,1],[2,3],[4,5],[6,7],[8,9]]
    qr.position[:] = block
    assert a.position.tolist() == [[0.0, 1.0], [2.0, 3.0]]              # rows 0,1 -> pool A
    assert b.position.tolist() == [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]  # rows 2,3,4 -> pool B

    qr.position[:] = qr.position.numpy()[::-1]                 # reverse the whole (5, 2) block, BY ORDER
    assert a.position.tolist() == [[8.0, 9.0], [6.0, 7.0]]
    assert b.position.tolist() == [[4.0, 5.0], [2.0, 3.0], [0.0, 1.0]]


def test_broadcast_row_or_scalar_writes_to_every_entity():
    """A value that is not (N, *) broadcasts to every entity, decided by shape against the logical (N, 4): a
    (4,) row and a scalar each fill all entities across both pools. N == 4 == the colour length here, and the
    (4,) row still broadcasts to all rather than being split into one value per entity."""
    a = _ball_pool([([0.0, 0.0], 5.0), ([1.0, 1.0], 6.0)])
    b = _ball_pool([([2.0, 2.0], 7.0), ([3.0, 3.0], 8.0)])      # N == 4 == len of a colour row
    qr = _query([a, b], "position", "radius", "color")

    qr.color[:] = np.array([9, 9, 9, 9], "int32")[:, None]      # (4, 1) row -> every entity, NOT split into scalars
    assert a.color.tolist() == [[9, 9, 9, 9], [9, 9, 9, 9]]
    assert b.color.tolist() == [[9, 9, 9, 9], [9, 9, 9, 9]]

    qr.radius[:] = 0.0                                          # scalar -> every entity
    assert (a.radius == 0).all()
    assert (b.radius == 0).all()


def test_row_operand_broadcasts_like_numpy_when_e0_equals_total_entities():
    """A (2,) row added to a 2-entity field broadcasts onto every entity, exactly like a real (2, 2) numpy
    array, even when the row length equals the entity count (N == 2, per-entity dim e0 == 2):

        [[1, 1], [2, 2]] + [10, 20]  ->  [[11, 21], [12, 22]]      (row broadcast onto every entity)

    The row has fewer dims than the field (ndim 1 < 2), so it aligns to the component axis and is not split
    per entity. Per-entity values would need shape (N, 1), as numpy requires."""
    a = _pos_pool([[1.0, 1.0]])
    b = _pos_pool([[2.0, 2.0]])
    qr = _query([a, b], "position")

    got = (qr.position + np.array([10.0, 20.0], "float32")).numpy()

    assert got.tolist() == [[11.0, 21.0], [12.0, 22.0]]     # numpy broadcast: the row lands on every entity


def test_same_row_matches_numpy_when_total_entities_differs_from_e0():
    """A (2,) row added to a field broadcasts onto every entity across uneven pools, exactly like numpy, when
    the row length differs from the entity count (N == 3, per-entity dim e0 == 2). Broadcasting depends on the
    row's ndim (1 < 2), not on the entity count."""
    a = _pos_pool([[1.0, 1.0]])
    b = _pos_pool([[2.0, 2.0], [3.0, 3.0]])             # N == 3, e0 == 2
    qr = _query([a, b], "position")

    row = np.array([10.0, 20.0], "float32")
    got = (qr.position + row).numpy()

    assert got.tolist() == (qr.position.numpy() + row).tolist()          # matches numpy
    assert got.tolist() == [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]


def test_empty_field_writes_behave_like_an_empty_numpy_block():
    """A field with zero entities (a pool that exists but is empty -> a (0, 2) block) behaves exactly like
    `np.empty((0, 2))`: a broadcastable value is a no-op (nothing to write), a wrong-shaped value raises
    ValueError, and .numpy() returns the (0, 2) array. Mirrors numpy:
        A = np.empty((0, 2)); A[:] = (1, 2)    -> no-op
                              A[:] = (1, 2, 3) -> ValueError"""
    qr = _query([_pool_with(0)], "position")

    assert len(qr) == 0
    assert qr.position.numpy().shape == (0, 2)

    qr.position[:] = (1.0, 2.0)                 # broadcastable to (0, 2) -> no-op, no crash
    assert qr.position.numpy().shape == (0, 2)

    with pytest.raises(ValueError):
        qr.position[:] = (1.0, 2.0, 3.0)        # (3,) cannot broadcast into (0, 2)


def test_empty_query_with_no_matched_pool_behaves_like_numpy():
    """A query that matches NO pools is still a (0, 2) block, because the field shape/dtype are threaded in (not
    just names). So it behaves like np.empty((0, 2)), same as a zero-entity pool: .numpy() -> (0, 2), a
    broadcastable write is a no-op, a wrong-shaped value raises ValueError."""
    qr = _query([], "position")

    assert len(qr) == 0
    assert qr.position.numpy().shape == (0, 2)

    qr.position[:] = (1.0, 2.0)                 # broadcastable to (0, 2) -> no-op
    assert qr.position.numpy().shape == (0, 2)

    with pytest.raises(ValueError):
        qr.position[:] = (1.0, 2.0, 3.0)        # (3,) cannot broadcast into (0, 2)


def test_assigning_a_field_scatters_like_a_recarray():
    """Assigning a field attribute writes THROUGH to the pools, exactly like np.recarray: `qr.position = X`
    scatters X into every entity -- whether X is a computed _Field, a raw array, or a scalar -- and a
    non-broadcastable value raises ValueError. Never a silent shadow, never a silent no-op. Each step is
    cross-checked against an actual np.recarray running the same assignment."""
    a, b = _pos_pool([[1.0, 1.0]]), _pos_pool([[2.0, 2.0], [3.0, 3.0]])
    qr = _query([a, b], "position")
    rec = np.rec.array(np.zeros(3, dtype=[("position", "f4", (2,))]))    # the numpy yardstick
    rec.position = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    qr.position = qr.position * 2                       # computed _Field -> writes through (not discarded)
    rec.position = rec.position * 2
    assert a.position.tolist() == [[2.0, 2.0]]
    assert b.position.tolist() == [[4.0, 4.0], [6.0, 6.0]]
    assert qr.position.numpy().tolist() == rec.position.tolist()

    qr.position = np.array([7.0, 8.0], "float32")       # raw (2,) row -> broadcasts to every entity
    rec.position = np.array([7.0, 8.0], "float32")
    assert qr.position.numpy().tolist() == rec.position.tolist() == [[7.0, 8.0]] * 3

    qr.position = 0.0                                   # scalar -> every entity
    assert (a.position == 0).all() and (b.position == 0).all()

    assert "position" not in vars(qr)                   # never shadowed; still served by __getattr__
    assert isinstance(qr.position, _Field)

    with pytest.raises(ValueError):                     # bad shape -> numpy rules, like recarray
        qr.position = np.array([1.0, 2.0, 3.0], "float32")


def test_repr_renders_and_reports_entity_count():
    """repr(QueryResult) must render -- it's used in logs/debugging. Pins that it doesn't crash and shows the
    entity count across pools (2 + 3 == 5)."""
    qr = _query([_pool_with(2), _pool_with(3)], "position")

    text = repr(qr)

    assert "QueryResult" in text
    assert "5" in text                          # total entities across the two pools
