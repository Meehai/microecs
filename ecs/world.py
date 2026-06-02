"""world.py - The world container for ECS. It manages all the pools (one per archetype). Entities are id-based."""
import numpy as np
from loggez import loggez_logger as logger

from .pool import Pool, PoolKey
from .utils import Shape, EntityId
from .component import Component

class World:
    """Generic container for pools of components. Newly added components are assigned a unique id and go in a pool"""
    def __init__(self, components: list[type[Component]]):
        self._check_components(components)
        self.pools: dict[PoolKey, Pool] = {}
        # components management
        self.component_names = [x.__name__ for x in components]
        self.component_types = list(components)
        self.component_to_bit: dict[type, int] = {t: 2**i for i, t in enumerate(components)} # unique bit for querying
        self.component_to_shapes: dict[type, list[Shape]] = {
            t: [f.metadata["shape"] for f in t.__dataclass_fields__.values()] for t in components}
        self.component_to_dtypes: dict[type, list[str]] = {
            t: [f.metadata["dtype"] for f in t.__dataclass_fields__.values()] for t in components}
        self.component_to_field_names: dict[type, list[str]] = {t: list(t.__dataclass_fields__) for t in components}
        # entity id management
        self._eid_to_pool_ix: dict[EntityId, tuple[Pool, int]] = {}
        self._pool_ix_to_eid: dict[tuple[Pool, int], EntityId] = {}
        self._last_id: EntityId = -1
        logger.debug(f"Created scene with components: {self.component_names}")

    def add_entity(self, components: list[type], **kwargs) -> EntityId:
        """adds an entity to the right pool based on its traits. Returns an entity id. Components are sent to kwargs"""
        pool = self._get_entity_pool(components, **kwargs)
        pool_index = pool.add_entity(**kwargs)
        self._last_id += 1
        self._eid_to_pool_ix[self._last_id] = (pool, pool_index)
        self._pool_ix_to_eid[(pool, pool_index)] = self._last_id
        return self._last_id

    def remove_entity(self, entity_id: EntityId):
        """removes an entity based on its unique entity id"""
        pool, pool_ix = self._eid_to_pool_ix.pop(entity_id)
        pool.remove_entity(pool_ix)
        id_which_was_last_in_pool = self._pool_ix_to_eid.pop((pool, len(pool)))
        if entity_id != id_which_was_last_in_pool:
            self._eid_to_pool_ix[id_which_was_last_in_pool] = (pool, pool_ix) # we re-use the popped id (it's swapped)
            self._pool_ix_to_eid[(pool, pool_ix)] = id_which_was_last_in_pool

    def query_and(self, component_types: list[type]) -> list[Pool]:
        """returns all the entities that have all the components"""
        key = self._make_key(component_types)
        res = []
        for archetype_key, archetype_pool in self.pools.items():
            if (archetype_key & key) == key: # key is subset of archetype_key
                res.append(archetype_pool)
        return res

    def _get_entity_pool(self, components: list[type], **entity_fields) -> Pool:
        assert len(components) > 0, f"Entity has no components: {self.component_names}"
        expected_fields = set()
        for component in components:
            for _field in self.component_to_field_names[component]:
                expected_fields.add(_field)
                assert _field in entity_fields, f"Entity doenst't have '{component}/{_field}'"
        assert (extra := (set(entity_fields) - expected_fields)) == set(), f"Extra fields: {extra}; {expected_fields=}"

        if (key := self._make_key(components)) not in self.pools:
            fields = sum([self.component_to_field_names[component] for component in components], []) # merge fields
            shapes = sum([self.component_to_shapes[component] for component in components], []) # merge shapes
            dtypes = sum([self.component_to_dtypes[component] for component in components], []) # merge dtypes
            self.pools[key] = Pool(fields, shapes, dtypes)
        return self.pools[key]

    def _make_key(self, components: list[type]) -> PoolKey:
        key = 0
        for component in components:
            assert component in self.component_types, f"Component '{component.__name__}' not in {self.component_names}"
            key |= self.component_to_bit[component]
        return key

    def _check_components(self, components: list[type]):
        _dtypes = {"float32", "int32", "bool", "str"}
        for component in components:
            assert hasattr(component, "__dataclass_fields__"), f"Component '{component}' is not a dataclass"
            for field_name, _field in component.__dataclass_fields__.items():
                assert _field.type == np.ndarray, f"Field '{field_name}' of '{component=}' not an array: {_field}"
                assert _field.metadata.keys() == {"shape", "dtype"}, _field.metadata
                assert isinstance(_field.metadata["shape"], tuple), _field.metadata["shape"]
                assert isinstance(fmd := _field.metadata["dtype"], str) and fmd in _dtypes, f"{fmd} not in {_dtypes}"
