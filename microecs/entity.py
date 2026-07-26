"""entity.py - A view of an entity with all its fields from the pool it belongs to in the world"""
from __future__ import annotations
from dataclasses import fields
from typing import Any, Iterable
import numpy as np
from .pool import Pool
from .component import ComponentType
from .utils import EntityId
from .command_buffer import CommandBuffer, Command, CommandType

# Note: if Entity gets new fields, add them here! Otherwise the user code may overwrite them e.g. ent._eid_to_pool_ix=xx
_ENTITY_INTERNAL_ATTRS = {"entity_id", "_eid_to_pool_ix", "_pool_to_components", "_world_command_buffer"}

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
        pool, _ = self._locate(names=[])
        return self._pool_to_components[pool]

    def get_fields(self) -> set[str]:
        """gets the fields of this entity. Note: they may change, so call this every time, don't store it."""
        pool, _ = self._locate(names=[])
        return pool.fields_set

    def set_data(self, **data):
        """Eagerly (no world.update()) sets the data of this entity. Multiple columns can be updated at once e.g.
        entity.set_data(a=x, b=y). Checks are done so object doesn't crash midway (e.g. bad dtype/shape etc.)."""
        pool, pool_index = self._locate(names=data.keys())

        if len(data) == 1: # fast path because we don't need to do exhaustive tests on a single k->v data set
            k, v = next(iter(data.items()))
            pool.data[k][pool_index] = v
            return

        # convert the data in numpy array of proper shape before calling pool.data[k][ix]=v so we don't have to revert
        ready: list[tuple[str, np.ndarray]] = []
        for k, v in data.items():
            col = pool.data[k]
            # no need to call the slow np.broadcast_to if v is already in proper shape and broadcastable and all.
            if not (isinstance(v, np.ndarray) and v.dtype == col.dtype and v.shape == col.shape[1:]):
                v = np.broadcast_to(np.asarray(v, dtype=col.dtype), col.shape[1:])
            ready.append((k, v))

        # finally set the data after the conversion was done
        for k, v in ready:
            pool.data[k][pool_index] = v

    def _locate(self, names: Iterable[str]) -> tuple[Pool, int]:
        try:
            pool, index = self._eid_to_pool_ix[self.entity_id]
        except KeyError:
            raise AttributeError(f"Entity {self.entity_id} not in world. Call `world.update()` if it was just added.")

        if not (flds := pool.fields_set).issuperset(names):
            raise AttributeError(f"Not all of {list(names)} are fields (entity id: {self.entity_id}). "
                                 f"\n- Components: {[c.__name__ for c in self.get_components()]}\n- Fields: {flds}")
        return pool, index

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
        pool, pool_index = self._locate(names=[name])
        return pool.data[name][pool_index]

    def __setattr__(self, name: str, value: np.ndarray):
        if name in _ENTITY_INTERNAL_ATTRS:
            super().__setattr__(name, value)
            return

        pool, pool_index = self._locate(names=[name])
        pool.data[name][pool_index] = value

    def __repr__(self):
        return f"EID-{self.entity_id}"

ENTITY_RESERVED_NAMES = _ENTITY_INTERNAL_ATTRS | {n for n in vars(Entity) if not n.startswith("__")}
