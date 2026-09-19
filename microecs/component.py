"""component.py - submodule holind the base class for all components, which is just a dataclass wrapper"""
from dataclasses import dataclass, field, Field
from typing import Any
import numpy as np

class Component:
    """Base for ECS components: subclassing auto-applies @dataclass(kw_only=True)."""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclass(kw_only=True)(cls)

ComponentType = type[Component]

# helper functions to build components and fields

class zeros: # pylint: disable=all
    """special object so we can instantiate with e.g. f32(..., default=zeros)"""

def _make_field(dtype: str, shape: tuple[int, ...], default: Any = None, **extra) -> Field:
    if default is zeros:
        if dtype == "object":
            default = np.full(shape, fill_value=None, dtype="object")
        else:
            default = np.zeros(shape, dtype=dtype)
    # pylint: disable=invalid-field-call  # builder pattern: Field is consumed by @dataclass on the Component subclass
    return field(metadata={**extra, "shape": shape, "dtype": dtype, "default": default})

def f32(shape: tuple[int, ...], default: Any = None, **extra) -> Field:
    """Builds a float32 field; kwargs (e.g. serializable/comment) go into the field's metadata."""
    return _make_field("float32", shape, default, **extra)

def i32(shape: tuple[int, ...], default: Any = None, **extra) -> Field:
    """Builds a int32 field; kwargs (e.g. serializable/comment) go into the field's metadata."""
    return _make_field("int32", shape, default, **extra)

def obj(shape: tuple[int, ...], default: Any = None, **extra) -> Field:
    """Builds a object field; kwargs (e.g. serializable/comment) go into the field's metadata."""
    return _make_field("object", shape, default, **extra)

def boolf(shape: tuple[int, ...], default: Any = None, **extra) -> Field:
    """Builds a bool field; kwargs (e.g. serializable/comment) go into the field's metadata."""
    return _make_field("bool", shape, default, **extra)
