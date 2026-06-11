"""entity.py - A view of an entity with all its fields from the pool it belongs to in the world"""
from dataclasses import fields
from typing import Any
import numpy as np
from .pool import Pool
from .component import ComponentType
from .utils import EntityId

# Note: if Entity gets new fields, add them here! Otherwise the user code may overwrite them e.g. ent._eid_to_pool_ix=xx
ENTITY_INTERNAL_ATTRS = {"entity_id", "_eid_to_pool_ix", "_pool_to_components"}

class Entity:
    """
    A view of an entity with all its fields from the pool it belongs to in the world.
    Note: Consistent to internal pool changes, however it always must check where it belongs so it's slow!!
    """
    def __init__(self, entity_id: EntityId, eid_to_pool_ix: dict[EntityId, tuple[Pool, int]],
                 pool_to_components: dict[Pool, list[ComponentType]]):
        self.entity_id = entity_id
        self._eid_to_pool_ix = eid_to_pool_ix
        self._pool_to_components = pool_to_components

    def get_components(self) -> list[ComponentType]:
        """get the components of this entity. Note: they may change, so call this every time, don't store it"""
        pool, _ = self._eid_to_pool_ix[self.entity_id]
        return self._pool_to_components[pool]

    def get_fields(self) -> list[str]:
        """gets the fields of this entity. Note: they may change, so call this every time, don't store it."""
        pool, _ = self._eid_to_pool_ix[self.entity_id]
        return pool.fields

    def to_dict(self, serialization_field: str | None = None) -> dict[str, Any]:
        """
        Serializes a single entity. Assumes fields are numpy. numerics are converted via `.tolist()`. objects are
        converted via `.item()`.
        Parameters:
        - `serialization_field` An optional special field added at World-level (e.g.: 'serializable'). If set, then we
        only serialize this entity's fields where the serialization_field is True. If not set, all fields are dumped.
        """
        components = self.get_components()
        res = {"components": [c.__name__ for c in components], "data": {}}
        for component in components:
            for field in fields(component):
                # the magic key that we have added in extra_metadata at World level. If not set, all fields are dumped.
                if serialization_field is not None and field.metadata[serialization_field] is False:
                    continue
                if field.metadata["dtype"] == "object": # dtype=object is for... non-numeric data (mostly dicts)
                    res["data"][field.name] = self.__getattr__(field.name).item()
                else:
                    res["data"][field.name] = self.__getattr__(field.name).tolist()
        return res

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            pool, pool_index = self._eid_to_pool_ix[self.entity_id]
        except KeyError:
            raise AttributeError(f"Entity {self.entity_id} not committed yet. Call `world.update()` (reading '{name}')")

        if name not in (_fields := pool.fields_set):
            raise AttributeError(f"Attribute '{name}' not in entity fields: {_fields} (entity id: {self.entity_id})")

        return pool.data[name][pool_index]

    def __setattr__(self, name: str, value: np.ndarray):
        if name in ENTITY_INTERNAL_ATTRS:
            super().__setattr__(name, value)
            return
        pool, pool_index = self._eid_to_pool_ix[self.entity_id]
        if name not in (_fields := pool.fields_set):
            raise AttributeError(f"Attribute '{name}' not in entity fields: {_fields} (entity id: {self.entity_id})")
        pool.data[name][pool_index] = value
