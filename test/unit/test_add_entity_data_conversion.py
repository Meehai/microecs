"""Unit tests for add_entity's array-like data conversion (_convert_components_data).

Contract (broadcastable commit): field values may be lists/tuples/scalars/strings -- they are converted
to the field's dtype/shape at the add_entity call. ndarrays are taken as-is and validated strictly
(dtype/shape), never silently cast. `None` is DATA for object fields (array([None]), e.g. "no texture")
and a caller bug for numeric fields (TypeError, not a silent NaN). Unknown kwargs are preserved through
the conversion so the "Extra fields" guard still fires. Omitted fields fall to their metadata default,
or raise the required-KeyError if default is None.
"""
import numpy as np
import pytest

from microecs import World, Component, f32, i32, obj, zeros


class HasPosition(Component):
    position: np.ndarray = f32((2,))


class HasTag(Component):
    tag: np.ndarray = obj((1,))


class HasLevel(Component):
    level: np.ndarray = i32((1,))


class HasVelDefault(Component):
    velocity: np.ndarray = f32((2,), default=zeros)


def test_array_like_values_converted():
    world = World(components=[HasPosition, HasTag, HasLevel])
    e_tag = world.add_entity((HasTag,), tag="robot")          # string -> object (1,)
    e_pos = world.add_entity((HasPosition,), position=(1.0, 2.0))  # tuple -> float32 (2,)
    e_lvl = world.add_entity((HasLevel,), level=2)            # int scalar -> int32 (1,)
    world.update()

    assert world.get_entity(e_tag).tag.tolist() == ["robot"]
    pos = world.get_entity(e_pos).position
    assert pos.dtype == np.float32 and pos.shape == (2,)
    assert np.array_equal(pos, np.array([1.0, 2.0], "float32"))
    assert world.get_entity(e_lvl).level.tolist() == [2]


def test_none_is_data_for_object_not_omission():
    """'tag: None' (key present) is data -> array([None]). None for a numeric field is a bug -> TypeError.
    Omitted required object fields still raise (None-as-value must not masquerade as omission)."""
    world = World(components=[HasPosition, HasTag])

    e = world.add_entity((HasTag,), tag=None)
    world.update()
    tag = world.get_entity(e).tag
    assert tag.dtype == np.object_ and tag.shape == (1,) and tag[0] is None

    with pytest.raises(TypeError, match="position"):
        world.add_entity((HasPosition,), position=None)

    with pytest.raises(KeyError, match="tag"):
        world.add_entity((HasTag,))
    assert len(world.live_entities) == 1   # only the earlier tag=None entity; the rejected add staged nothing


def test_omitted_field_fills_default():
    world = World(components=[HasVelDefault])
    e = world.add_entity((HasVelDefault,))
    world.update()
    assert np.array_equal(world.get_entity(e).velocity, np.zeros(2, "float32"))


def test_ndarray_fields_validated_strictly():
    """Explicit ndarrays are NOT re-converted: wrong dtype/shape keep the strict validation errors."""
    world = World(components=[HasPosition])

    with pytest.raises(TypeError, match="dtype"):
        world.add_entity((HasPosition,), position=np.array([1, 2], "int32"))
    with pytest.raises(ValueError, match="shape"):
        world.add_entity((HasPosition,), position=np.zeros((3,), "float32"))
    assert list(world.live_entities) == []


def test_non_convertible_and_non_broadcastable_rejected():
    world = World(components=[HasPosition])
    with pytest.raises(TypeError):
        world.add_entity((HasPosition,), position={"x": 1.0})       # not convertible
    with pytest.raises(ValueError):
        world.add_entity((HasPosition,), position=[1.0, 2.0, 3.0])  # (3,) not broadcastable to (2,)
    assert list(world.live_entities) == []


def test_extra_kwargs_still_rejected():
    """Conversion must not swallow fields of unrequested components: extras reach the validation guard."""
    world = World(components=[HasPosition, HasTag])
    with pytest.raises(ValueError, match="tag"):
        world.add_entity((HasPosition,), position=np.zeros(2, "float32"), tag="nope")
    assert list(world.live_entities) == []
