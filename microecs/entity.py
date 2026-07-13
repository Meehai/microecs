"""entity.py - A view of an entity with all its fields from the pool it belongs to in the world"""
from __future__ import annotations
from dataclasses import fields
from typing import Any
import numpy as np
from .pool import Pool
from .component import ComponentType
from .utils import EntityId
from .command_buffer import CommandBuffer, Command, CommandType

# Note: if Entity gets new fields, add them here! Otherwise the user code may overwrite them e.g. ent._eid_to_pool_ix=xx
ENTITY_INTERNAL_ATTRS = {"entity_id", "_eid_to_pool_ix", "_pool_to_components", "_world_command_buffer"}

class Entity:
    """
    A view of an entity with all its fields from the pool it belongs to in the world.
    Note: Consistent to internal pool changes, however it always must check where it belongs so it's slow!!
    """
    def __init__(self, entity_id: EntityId, eid_to_pool_ix: dict[EntityId, tuple[Pool, int]],
                 pool_to_components: dict[Pool, list[ComponentType]], world_command_buffer: CommandBuffer):
        self.entity_id = entity_id
        self._eid_to_pool_ix = eid_to_pool_ix
        self._pool_to_components = pool_to_components
        self._world_command_buffer = world_command_buffer # the world command buffer, needed for add/remove_component

    def add_component(self, component: ComponentType, **kwargs):
        """Adds a component to an entity. Component data is sent to kwargs. Lazy; call world.update()"""
        self._world_command_buffer.append(Command(CommandType.ADD_COMPONENT, self.entity_id,
                                          args={"component": component, **kwargs}))

    def remove_component(self, component: ComponentType):
        """Removes a component from an entity given its id. Lazy; call update()"""
        self._world_command_buffer.append(Command(CommandType.REMOVE_COMPONENT, self.entity_id, args=component))

    def has_component(self, component: ComponentType) -> bool:
        """Checks if this entity has a component"""
        return component in self.get_components()

    def get_components(self) -> list[ComponentType]:
        """get the components of this entity. Note: they may change, so call this every time, don't store it"""
        pool, _ = self._eid_to_pool_ix[self.entity_id]
        return self._pool_to_components[pool]

    def get_fields(self) -> list[str]:
        """gets the fields of this entity. Note: they may change, so call this every time, don't store it."""
        pool, _ = self._eid_to_pool_ix[self.entity_id]
        return pool.fields

    def set_component_data(self, component: ComponentType, data: dict[str, np.ndarray]):
        """Sets the data of a single component as a transaction. If the set fails, we try to revert the entity's data"""
        self._world_command_buffer.append(
            Command(CommandType.SET_DATA, self.entity_id, args={"component": component, "data": data}))

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
            raise AttributeError(f"Attribute '{name}' not found (entity id: {self.entity_id}). "
                                 f"\n- Components: {[c.__name__ for c in self.get_components()]}\n- Fields: {_fields}")
        return pool.data[name][pool_index]

    def __setattr__(self, name: str, value: np.ndarray):
        if name in ENTITY_INTERNAL_ATTRS:
            super().__setattr__(name, value)
            return
        pool, pool_index = self._eid_to_pool_ix[self.entity_id]
        if name not in (_fields := pool.fields_set):
            raise AttributeError(f"Attribute '{name}' not in entity fields: {_fields} (entity id: {self.entity_id})")
        pool.data[name][pool_index] = value

    def __repr__(self):
        return f"EID-{self.entity_id}"
