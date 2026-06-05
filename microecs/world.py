"""world.py - The world container for ECS. It manages all the pools (one per archetype). Entities are id-based."""
from typing import Callable, get_type_hints
from functools import partial
import numpy as np

from .pool import Pool
from .utils import Shape, EntityId, PoolKey, EntityData, logger
from .component import ComponentType
from .query_result import QueryResult

class World:
    """
    Generic container for pools of components. Newly added components are assigned a unique id and go in a pool
    Parameters
    - components The list of components that the world accepts
    - extra_field_metadata The list of required extra metadata for each field besides shape and dtype.
    """
    def __init__(self, components: list[ComponentType], extra_field_metadata: list[str] | None = None):
        self.extra_field_metadata = extra_field_metadata or []
        self._check_components(components)
        self.pools: dict[PoolKey, Pool] = {}
        self.pool_to_components: dict[Pool, list[ComponentType]] = {}
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
        self._pool_ids: dict[Pool, list[EntityId]] = {}
        self._last_id: EntityId = -1
        self._live_ids: set[EntityId] = set() # 'eager' mode ids so e.g. calling remove_entity twice before update fails
        # command buffer management. {add/remove}_{entity/component} are lazy. Taken into account after update().
        self._command_buffer: list[Callable] = []
        self._cache: dict[PoolKey, QueryResult] = {}
        logger.debug(f"Created scene with components: {self.component_names}")

    # public api

    def update(self):
        """commits the underlying pool changes from the systems between two updates. Should be called in main loop."""
        for fn in self._command_buffer:
            fn()

        if len(self._command_buffer) > 0:
            self._command_buffer.clear()
            self._cache.clear()

    def add_entity(self, components: list[ComponentType], **kwargs) -> EntityId:
        """Adds an entity to the world based on components (data->kwargs). Returns an entity id. Lazy; call update()"""
        self._check_components_against_pool(components, **kwargs)
        self._last_id += 1
        self._live_ids.add(self._last_id)
        self._command_buffer.append(partial(self._add_to_pool, entity_id=self._last_id,
                                            components=components, **kwargs))
        return self._last_id

    def remove_entity(self, entity_id: EntityId):
        """Removes an entity based on its unique entity id. Lazy; call update()"""
        assert entity_id in self._live_ids, f"Entity id: {entity_id} not in {self._live_ids=}"
        self._live_ids.remove(entity_id)
        self._command_buffer.append(partial(self._pop_from_pool, entity_id=entity_id))

    def get_entity(self, entity_id: EntityId) -> tuple[EntityData, list[ComponentType]]:
        """Gets the entity (data) and its components (list of types) given an entity id. Used for 'object-like' ops"""
        assert entity_id in self._eid_to_pool_ix, f"Entity id {entity_id} not found. Perhaps you didn't world.update()"
        pool, pool_ix = self._eid_to_pool_ix[entity_id]
        entity = {k: pool.data[k][pool_ix] for k in pool.fields}
        components = self.pool_to_components[pool]
        return entity, components

    def add_component(self, entity_id: EntityId, component: ComponentType, **kwargs):
        """Adds a component to an entity given its id. Component data is sent to kwargs. Lazy; call update()"""
        assert entity_id in self._live_ids, f"Entity id: {entity_id} not in {self._live_ids=}"
        assert component in self.component_types, f"Component '{component}' not in {self.component_types}"
        self._command_buffer.append(partial(self._do_add_component, entity_id=entity_id, component=component, **kwargs))

    def remove_component(self, entity_id: EntityId, component: ComponentType):
        """Removes a component from an entity given its id. Lazy; call update()"""
        assert entity_id in self._live_ids, f"Entity id: {entity_id} not in {self._live_ids=}"
        assert component in self.component_types, f"Component '{component}' not in {self.component_types}"
        self._command_buffer.append(partial(self._do_remove_component, entity_id=entity_id, component=component))

    def query_and(self, component_types: list[ComponentType]) -> QueryResult:
        """returns A QueryResult object with the entities that have all the requested components (entity ids too)."""
        # Note: we can cache the queries. The only time it can get invalidated (via public API) is at update().
        if (key := self._make_key(component_types)) in self._cache:
            return self._cache[key]

        res = []
        for archetype_key, archetype_pool in self.pools.items():
            if (archetype_key & key) == key: # key is subset of archetype_key
                res.append(archetype_pool)

        field_names  = sum([self.component_to_field_names[c] for c in component_types], [])
        field_shapes = dict(zip(field_names, sum([self.component_to_shapes[c] for c in component_types], [])))
        field_dtypes = dict(zip(field_names, sum([self.component_to_dtypes[c] for c in component_types], [])))
        entity_ids = np.array(sum((self._pool_ids[p] for p in res), []), dtype="int64")

        self._cache[key] = QueryResult(res, field_shapes=field_shapes, field_dtypes=field_dtypes, entity_ids=entity_ids)
        return self._cache[key]

    # private stuff

    # eager mode methods equivalent to add/remove entities and add/remove_components

    def _add_to_pool(self, entity_id: EntityId, components: list[ComponentType], **kwargs):
        """adds the item to the pool"""
        pool = self._get_entity_pool(components)
        pool_index = pool.add_entity(**kwargs)
        self._eid_to_pool_ix[entity_id] = (pool, pool_index)
        self._pool_ids.setdefault(pool, []).append(entity_id)
        assert len(self._pool_ids[pool]) == len(pool), (pool, len(self._pool_ids[pool]), len(pool))

    def _pop_from_pool(self, entity_id: EntityId) -> tuple[EntityData, list[ComponentType]]:
        """common function that updates the entities inside a pool (after popswap) and removes them if they get empty"""
        old_pool, pool_ix = self._eid_to_pool_ix.pop(entity_id)
        entity = old_pool.pop_entity(pool_ix)
        components = self.pool_to_components[old_pool]
        id_which_was_last_in_pool = self._pool_ids[old_pool].pop()
        if entity_id != id_which_was_last_in_pool:
            self._eid_to_pool_ix[id_which_was_last_in_pool] = (old_pool, pool_ix) # we re-use the popped id (swapped)
            self._pool_ids[old_pool][pool_ix] = id_which_was_last_in_pool
        if len(old_pool) == 0:
            del self.pools[self._make_key(components)]
            del self.pool_to_components[old_pool]
            del self._pool_ids[old_pool]
        return entity, components

    def _do_add_component(self, entity_id: EntityId, component: ComponentType, **kwargs):
        entity, components = self._pop_from_pool(entity_id)
        new_components = [*components, component]
        assert entity.keys().isdisjoint(kwargs), f"Duplicate keys: {entity.keys()} vs {kwargs.keys()}"
        self._check_components_against_pool(new_components, **entity, **kwargs)
        self._add_to_pool(entity_id, new_components, **entity, **kwargs)

    def _do_remove_component(self, entity_id: EntityId, component: ComponentType):
        entity, components = self._pop_from_pool(entity_id)
        for _field in self.component_to_field_names[component]:
            assert _field in entity.keys(), f"Field {component}/{_field} not in components: {components} ({entity_id=})"
            entity.pop(_field)
        new_components = [c for c in components if c != component]
        self._check_components_against_pool(new_components, **entity)
        self._add_to_pool(entity_id, new_components, **entity)

    # other low-level methods

    def _check_components_against_pool(self, components: list[ComponentType], **entity_fields):
        assert len(cs := components) > 0, f"Entity has no components: {self.component_names}"
        assert all(c in self.component_types for c in cs), f"Components '{cs}' not in {self.component_types}"
        expected_fields = set()
        for component in components:
            for _field in self.component_to_field_names[component]:
                expected_fields.add(_field)
                assert _field in entity_fields, f"Entity doesn't have '{component}/{_field}'"
        assert (extra := set(entity_fields) - expected_fields) == set(), f"Extra fields: {extra}; {expected_fields=}"

    def _get_entity_pool(self, components: list[ComponentType]) -> Pool:
        if (key := self._make_key(components)) not in self.pools:
            fields = sum([self.component_to_field_names[component] for component in components], []) # merge fields
            shapes = sum([self.component_to_shapes[component] for component in components], []) # merge shapes
            dtypes = sum([self.component_to_dtypes[component] for component in components], []) # merge dtypes
            self.pools[key] = Pool(fields, shapes, dtypes)
            self.pool_to_components[self.pools[key]] = components
        return self.pools[key]

    def _make_key(self, components: list[ComponentType]) -> PoolKey:
        key = 0
        for component in components:
            assert component in self.component_types, f"Component '{component.__name__}' not in {self.component_names}"
            key |= self.component_to_bit[component]
        return key

    def _check_components(self, components: list[ComponentType]):
        _query_result_reserved_names = _qres = sorted(vars(QueryResult([], {}, {}, [])))
        _dtypes = {"float32", "int32", "bool", "str", "object"}

        for component in components:
            assert hasattr(component, "__dataclass_fields__"), f"Component '{component}' is not a dataclass"
            hints = get_type_hints(component) # make it work with from __future__ import annotations
            for field_name, _field in component.__dataclass_fields__.items():
                assert hints[field_name] is np.ndarray, f"Field '{field_name}' of '{component=}' not an array: {_field}"
                assert _field.metadata.keys() == {"shape", "dtype", *self.extra_field_metadata}, _field.metadata
                assert isinstance(_field.metadata["shape"], tuple), _field.metadata["shape"]
                assert isinstance(fmd := _field.metadata["dtype"], str) and fmd in _dtypes, f"{fmd} not in {_dtypes}"
                assert field_name not in _query_result_reserved_names, f"Field '{field_name}' in {_qres}"

    def __len__(self):
        return len(self._live_ids)

    def __repr__(self):
        return (f"[World]\n- Entities: {len(self)} (last id: {self._last_id})"
                f"\n- Components ({len(self.component_names)}): {self.component_names}"
                f"\n- Pools: {len(self.pools)}\n- Command buffer: {len(self._command_buffer)}")
