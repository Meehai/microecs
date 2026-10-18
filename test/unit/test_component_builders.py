"""Unit tests for the field builders (f32/i32/obj/boolf/zeros) and broadcastable defaults.

Builders are the ergonomic way to declare a component: they fill the field's metadata dict and turn the
`zeros` sentinel into a concrete default (zeros for numerics/bool, None for object). The World constructor
then broadcasts list/tuple/scalar defaults to the field shape and validates dtype/shape; add_entity fills
omitted fields from them. Only shape/dtype/default are the world's enforced metadata; any other kwarg the
builder is given (e.g. serializable/comment) passes through into the field's metadata -- so a component
declared with such extras REQUIRES a World that declares them (World(extra_metadata=[...])), and crashes
otherwise.
"""
import numpy as np
import pytest

from microecs import World, Component, f32, i32, obj, boolf, zeros


class C1(Component):
    f: np.ndarray = f32((1,), serializable=True, comment="a float", default=zeros)
    i: np.ndarray = i32((2,), serializable=True, comment="an int", default=zeros)
    o: np.ndarray = obj((1,), serializable=True, comment="an obj", default=zeros)
    b: np.ndarray = boolf((1,), serializable=True, comment="a bool", default=zeros)
    d: np.ndarray = f32((3,), serializable=True, comment="list default", default=[1.0, 2.0, 3.0])
    r: np.ndarray = obj((1,), serializable=True, comment="required", default=None)  # required, may be None-valued


def _world():
    return World(components=[C1], extra_metadata=["serializable", "comment"])


def test_builders_fill_metadata():
    fields = C1.__dataclass_fields__
    md = fields["f"].metadata
    assert md["dtype"] == "float32" and md["shape"] == (1,)
    assert md["serializable"] is True and md["comment"] == "a float"
    assert np.array_equal(md["default"], np.zeros(1, "float32"))
    assert fields["i"].metadata["dtype"] == "int32" and fields["i"].metadata["shape"] == (2,)
    assert fields["b"].metadata["dtype"] == "bool"
    assert fields["o"].metadata["dtype"] == "object"
    assert fields["r"].metadata["default"] is None            # default=None still means "required"
    assert fields["d"].metadata["default"] == [1.0, 2.0, 3.0]  # raw default kept; broadcast happens in World


def test_zeros_sentinel_becomes_zeros_array():
    """zeros -> np.zeros for numerics/bool, and array([None]) for object: 'zero' of an object field is None."""
    fields = C1.__dataclass_fields__
    for name in ("f", "i", "o", "b"):
        default = fields[name].metadata["default"]
        assert isinstance(default, np.ndarray)
        assert default.dtype == fields[name].metadata["dtype"]
        assert default.shape == fields[name].metadata["shape"]
        if fields[name].metadata["dtype"] == "object":
            assert default.tolist() == [None]
        else:
            assert np.array_equal(default, np.zeros(default.shape, default.dtype))


def test_extra_metadata_not_declared_in_world_crashes():
    """Only shape/dtype/default are enforced; the rest goes through the builder's kwargs into metadata. A field
    carrying `comment` (or any extra) used in a World that does NOT declare that extra crashes at construction."""
    class C2(Component):
        x: np.ndarray = f32((2,), serializable=True, comment="c", default=zeros)

    with pytest.raises(ValueError, match="extra_metadata"):
        World(components=[C2])                          # comment not declared -> crash

    # same component, world that declares the extras: fine (and defaults/data still work)
    w = World(components=[C2], extra_metadata=["serializable", "comment"])
    e = w.add_entity((C2,))
    w.update()
    assert np.array_equal(w.get_entity(e).x, np.zeros(2, "float32"))


def test_arbitrary_extra_kwarg_passthrough():
    """Any kwarg (not just serializable/comment) passes into metadata; it is enforced by the world's declared set."""
    class C3(Component):
        x: np.ndarray = f32((2,), foo=1, default=zeros)

    with pytest.raises(ValueError, match="extra_metadata"):
        World(components=[C3])
    w = World(components=[C3], extra_metadata=["foo"])
    assert w.component_to_defaults[C3][0].tolist() == [0.0, 0.0]


def test_object_zeros_default_on_omission_is_none():
    """Omitting an object field whose default is zeros fills [None], never [0]."""
    world = _world()
    e = world.add_entity((C1,), r=None)                      # o omitted -> default [None]
    world.update()
    assert world.get_entity(e).o.tolist() == [None]
    assert world.get_entity(e).o.tolist() != [0]


def test_world_broadcasts_and_validates_defaults():
    world = _world()   # list default [1.0, 2.0, 3.0] broadcast to (3,) float32, dtype/shape validated
    d = world.component_to_defaults[C1][4]
    assert d.dtype == np.float32 and d.shape == (3,)
    assert np.array_equal(d, np.array([1.0, 2.0, 3.0], "float32"))


def test_bad_default_dtype_and_shape_raise():
    class C4(Component):
        x: np.ndarray = f32((2,), serializable=True, comment="", default=np.array([1, 2], "int32"))

    class C5(Component):
        x: np.ndarray = f32((2,), serializable=True, comment="", default=[1.0, 2.0, 3.0])  # (3,) not -> (2,)

    with pytest.raises(TypeError, match="dtype"):
        World(components=[C4], extra_metadata=["serializable", "comment"])
    with pytest.raises(ValueError, match="shape"):
        World(components=[C5], extra_metadata=["serializable", "comment"])


def test_omitted_fields_fill_from_defaults():
    world = _world()
    e = world.add_entity((C1,), o=None, r=None)   # only the two object fields supplied
    world.update()
    ent = world.get_entity(e)
    assert np.array_equal(ent.f, np.zeros(1, "float32"))
    assert np.array_equal(ent.i, np.zeros(2, "int32"))
    assert ent.o.tolist() == [None]
    assert ent.r.tolist() == [None]
    assert np.array_equal(ent.d, np.array([1.0, 2.0, 3.0], "float32"))
    assert ent.b.tolist() == [False]


def test_required_object_field_omission_still_raises():
    world = _world()
    with pytest.raises(KeyError, match="r"):
        world.add_entity((C1,), o=None)   # r has default=None (required) and was omitted
    assert list(world.live_entities) == []
