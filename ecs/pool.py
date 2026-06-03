"""pool.py - A pool of entities of the same type (same list of components). Basically a dynamic array with numpy"""
import numpy as np
from loggez import loggez_logger as logger
from .utils import Shape

PoolKey = int

class Pool:
    """
    Pool is a dynamic array of entities data given a list of fields, shapes and dtypes (from traits).
    Pool has no concept of entity ids.
    """
    INITIAL_CAPACITY = 100
    RESERVED_NAMES = {"size", "capacity", "fields", "shapes", "dtypes", "data"}

    def __init__(self, fields: list[str], shapes: list[Shape], dtypes: list[np.dtype]):
        assert len(fields) == len(shapes) == len(dtypes), (len(fields), len(shapes), len(dtypes))
        assert not (set(fields) & Pool.RESERVED_NAMES), f"One of {fields=} in {Pool.RESERVED_NAMES}"
        self.fields = fields
        self.shapes = shapes
        self.dtypes = dtypes

        self.data: dict[str, np.ndarray] = {} # the actual data is stored in a dynamic arrray, one per field
        self.size = 0
        self.capacity = Pool.INITIAL_CAPACITY
        for _field, shape, dtype in zip(fields, shapes, dtypes):
            self.data[_field] = np.empty(shape=(self.capacity, *shape), dtype=dtype)

    def add_entity(self, **entity_fields) -> int:
        """Adds an entity to the pool. All the fields required by this pool must be provided as kwargs"""
        if self.size == self.capacity:
            self._realloc(self.capacity * 2)
            logger.debug(f"Capacity extended from {self.capacity // 2} to {self.capacity}")

        for _field, field_shape, field_dtype in zip(self.fields, self.shapes, self.dtypes):
            new_item: np.ndarray = entity_fields[_field] # checked in World._get_entity_pool(entity).
            assert new_item.shape == field_shape, f"{_field=} {new_item=}, {new_item.shape=}, {field_shape=}"
            assert np.issubdtype(new_item.dtype, field_dtype), f"{_field=} {new_item=} {new_item.dtype=} {field_dtype=}"
            self.data[_field][self.size] = new_item
        self.size += 1
        return self.size - 1

    def remove_entity(self, entity_index: int):
        """removes an entity given an index (NOT ID) inside this pool"""
        assert entity_index < self.size, f"OOB: {entity_index=}, {self.size=}"
        for _field in self.fields:
            self.data[_field][entity_index] = self.data[_field][self.size - 1]
        self.size -= 1

        if self.size < self.capacity / 4 and self.capacity > Pool.INITIAL_CAPACITY:
            self._realloc(self.capacity // 2)

    def pop_entity(self, entity_index: int) -> dict[str, np.ndarray]:
        """pops an entity given an index (NOT ID) inside this pool and returns the data"""
        res = {_field: self.data[_field][entity_index].copy() for _field in self.fields}
        self.remove_entity(entity_index)
        return res

    def _realloc(self, new_capacity: int):
        for _field, shape, dtype in zip(self.fields, self.shapes, self.dtypes):
            old_data = self.data[_field]
            self.data[_field] = np.empty(shape=(new_capacity, *shape), dtype=dtype)
            self.data[_field][0:self.size] = old_data[0:self.size]
        self.capacity = new_capacity

    def __getattr__(self, name):
        if (data := self.__dict__.get("data")) is not None and name in data:
            return data[name][0: self.size]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if (data := self.__dict__.get("data")) is not None and name in data:
            raise ValueError(f"Cannot explicitly set anything to Pool. Use `pool.component[:] = ...` ({name=})")
        super().__setattr__(name, value)

    def __len__(self):
        return self.size
