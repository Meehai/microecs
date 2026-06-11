"""world.py - The world container for ECS. It manages all the pools (one per archetype). Entities are id-based."""
from typing import Callable, get_type_hints
from functools import partial
from dataclasses import fields
import numpy as np

from .pool import Pool
from .utils import Shape, EntityId, PoolKey, logger
from .component import ComponentType
from .query_result import QueryResult, QUERY_RESULT_INTERNAL_ATTRS
from .entity import Entity, ENTITY_INTERNAL_ATTRS

class World:
    """
    Generic container for pools of components. Newly added components are assigned a unique id and go in a pool
    Parameters
    - components The list of components that the world accepts
    - extra_metadata The list of required extra metadata for each field besides shape and dtype.
    """
    def __init__(self, components: list[ComponentType], extra_metadata: list[str] | None = None):
        self.extra_metadata = extra_metadata or []
        assert isinstance(self.extra_metadata, list), type(self.extra_metadata)
        self._check_components(components)

        # pools management
        self.pools: dict[PoolKey, Pool] = {}
        self.pool_to_components: dict[Pool, list[ComponentType]] = {}

        # components management
        self.component_names = [x.__name__ for x in components]
        self.component_types = set(components)
        self.component_to_bit: dict[ComponentType, int] = {t: 2**i for i, t in enumerate(components)} # bit for querying
        self.component_to_shapes: dict[ComponentType, list[Shape]] = {
            c: [f.metadata["shape"] for f in fields(c)] for c in components}
        self.component_to_dtypes: dict[ComponentType, list[str]] = {
            c: [f.metadata["dtype"] for f in fields(c)] for c in components}
        self.component_to_field_names: dict[ComponentType, list[str]] = {
            c: [f.name for f in fields(c)] for c in components}
        # entities management
        self._eid_to_pool_ix: dict[EntityId, tuple[Pool, int]] = {}
        self._pool_ids: dict[Pool, list[EntityId]] = {}
        self._last_id: EntityId = -1
        # a dictionary of all live entities in 'eager' mode (before update()). The actual entity is created at request
        # in get_entity, so we don't pay for the Entity object unless it's explicitly requested by the user.
        self.live_entities: dict[EntityId, Entity | None] = {}

        # command buffer management. {add/remove}_{entity/component} are lazy. Taken into account after update().
        self._command_buffer: list[Callable] = []
        self._cache: dict[tuple[PoolKey, PoolKey], QueryResult] = {} # include+exclude key
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
        self.live_entities[self._last_id] = None # add the id the live_entities, but the object is created in get_entity
        self._command_buffer.append(partial(self._add_to_pool, entity_id=self._last_id,
                                            components=components, **kwargs))
        logger.debug(f"Created entity. ID: {self._last_id}. Components: {[c.__name__ for c in components]}")
        return self._last_id

    def remove_entity(self, entity_id: EntityId):
        """Removes an entity based on its unique entity id. Lazy; call update()"""
        assert entity_id in self.live_entities, f"Entity id: {entity_id} not in the world"
        del self.live_entities[entity_id]
        self._command_buffer.append(partial(self._pop_from_pool, entity_id=entity_id))

    def get_entity(self, entity_id: EntityId) -> Entity:
        """Gets the entity reference given an entity id. Used for 'object-like' ops. Lazy; call world.update()"""
        assert entity_id in self.live_entities, f"Entity id: {entity_id} not in the world"
        if self.live_entities[entity_id] is None:
            # only instantiate on first request, so the object is not created for no reason at add_entity time.
            self.live_entities[entity_id] = Entity(entity_id, self._eid_to_pool_ix, self.pool_to_components)
        return self.live_entities[entity_id]

    def set_entity_data(self, entity_id: EntityId, field_name: str, value: np.ndarray):
        """Sets the value of one specific entity's field given an enttiy id. Non-vectorized operation, use rarely"""
        assert entity_id in self._eid_to_pool_ix, f"Entity id {entity_id} not found. Perhaps you didn't world.update()"
        pool, pool_ix = self._eid_to_pool_ix[entity_id]
        pool.data[field_name][pool_ix] = value

    def add_component(self, entity_id: EntityId, component: ComponentType, **kwargs):
        """Adds a component to an entity given its id. Component data is sent to kwargs. Lazy; call update()"""
        assert entity_id in self.live_entities, f"Entity id: {entity_id} not in the world"
        assert component in self.component_types, f"Component '{component}' not in {self.component_types}"
        self._command_buffer.append(partial(self._do_add_component, entity_id=entity_id, component=component, **kwargs))

    def remove_component(self, entity_id: EntityId, component: ComponentType):
        """Removes a component from an entity given its id. Lazy; call update()"""
        assert entity_id in self.live_entities, f"Entity id: {entity_id} not in the world"
        assert component in self.component_types, f"Component '{component}' not in {self.component_types}"
        self._command_buffer.append(partial(self._do_remove_component, entity_id=entity_id, component=component))

    def query(self, *include: ComponentType, exclude: list[ComponentType] | None = None) -> QueryResult:
        """
        Queries the world for entities that match the include set of components.
        Syntax: `world.query(A, B, exclude=[C, D])` is, in logic form, a chain of 'ands': `A & B & ~C & ~D`.
        Return: A `QueryResult` object with the entities that have all the requested components. Has `EntityIds` too.
        """

        # Note: we can cache the queries. The only time it can get invalidated (via public API) is at update().
        include_key = self._make_key(include)
        exclude_key = self._make_key(exclude or [])
        if (key := (include_key, exclude_key)) in self._cache:
            return self._cache[key]

        # archetype_key = (1 0 0 1 1) &
        #           key = (1 0 0 0 1)
        #              -> (1 0 0 0 1) OK
        # but
        # archetype_key = (1 0 0 1 1) &
        #           key = (1 0 1 0 0)
        #              -> (1 0 0 0 0) NOT OK
        res = []
        for archetype_key, archetype_pool in self.pools.items():
            if (archetype_key & include_key) == include_key and (archetype_key & exclude_key) == 0:
                res.append(archetype_pool)

        field_names = sum([self.component_to_field_names[c] for c in include], [])
        field_shapes = dict(zip(field_names, sum([self.component_to_shapes[c] for c in include], [])))
        field_dtypes = dict(zip(field_names, sum([self.component_to_dtypes[c] for c in include], [])))
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

    def _pop_from_pool(self, entity_id: EntityId) -> tuple[dict[str, np.ndarray], list[ComponentType]]:
        """common function that updates the entities inside a pool (after popswap) and removes them if they get empty"""
        old_pool, pool_ix = self._eid_to_pool_ix.pop(entity_id)
        entity_data = old_pool.pop_entity(pool_ix)
        components = self.pool_to_components[old_pool]
        id_which_was_last_in_pool = self._pool_ids[old_pool].pop()
        if entity_id != id_which_was_last_in_pool:
            self._eid_to_pool_ix[id_which_was_last_in_pool] = (old_pool, pool_ix) # we re-use the popped id (swapped)
            self._pool_ids[old_pool][pool_ix] = id_which_was_last_in_pool
        if len(old_pool) == 0:
            del self.pools[self._make_key(components)]
            del self.pool_to_components[old_pool]
            del self._pool_ids[old_pool]
        return entity_data, components

    def _do_add_component(self, entity_id: EntityId, component: ComponentType, **kwargs):
        entity_data, components = self._pop_from_pool(entity_id)
        new_components = [*components, component]
        assert entity_data.keys().isdisjoint(kwargs), f"Duplicate keys: {entity_data.keys()} vs {kwargs.keys()}"
        self._check_components_against_pool(new_components, **entity_data, **kwargs)
        self._add_to_pool(entity_id, new_components, **entity_data, **kwargs)

    def _do_remove_component(self, entity_id: EntityId, component: ComponentType):
        entity_data, components = self._pop_from_pool(entity_id)
        for _field in self.component_to_field_names[component]:
            assert _field in entity_data, f"Field {component}/{_field} not in components: {components} ({entity_id=})"
            entity_data.pop(_field)
        new_components = [c for c in components if c != component]
        self._check_components_against_pool(new_components, **entity_data)
        self._add_to_pool(entity_id, new_components, **entity_data)

    # other low-level methods

    def _check_components_against_pool(self, components: list[ComponentType], **entity_fields):
        assert len(cs := set(components)) > 0, f"Entity has no components: {self.component_names}"
        assert (diff := cs - self.component_types) == set(), f"Missing comps:\n{cs=}\n{self.component_types=}\n{diff=}"
        expected_fields = set()
        for component in components:
            for _field in self.component_to_field_names[component]:
                expected_fields.add(_field)
                assert _field in entity_fields, f"Entity doesn't have '{component.__name__}/{_field}'"
        assert (extra := set(entity_fields) - expected_fields) == set(), f"Extra fields: {extra}; {expected_fields=}"

    def _get_entity_pool(self, components: list[ComponentType]) -> Pool:
        if (key := self._make_key(components)) not in self.pools:
            _fields = sum([self.component_to_field_names[component] for component in components], []) # merge fields
            shapes = sum([self.component_to_shapes[component] for component in components], []) # merge shapes
            dtypes = sum([self.component_to_dtypes[component] for component in components], []) # merge dtypes
            self.pools[key] = Pool(_fields, shapes, dtypes)
            self.pool_to_components[self.pools[key]] = components
        return self.pools[key]

    def _make_key(self, components: list[ComponentType]) -> PoolKey:
        key = 0
        for component in components:
            assert component in self.component_types, f"Component '{component.__name__}' not in {self.component_names}"
            key |= self.component_to_bit[component]
        return key

    def _check_components(self, components: list[ComponentType]):
        reserved_names = (ENTITY_INTERNAL_ATTRS | {n for n in vars(Entity) if not n.startswith("__")} |
                          QUERY_RESULT_INTERNAL_ATTRS | {n for n in vars(QueryResult) if not n.startswith("__")})
        dtypes = {"float32", "int32", "bool", "str", "object"}
        expected_meta = {"shape", "dtype", *self.extra_metadata}

        for component in components:
            cn = component.__name__
            assert hasattr(component, "__dataclass_fields__"), f"Component '{cn}' is not a dataclass"
            hints = get_type_hints(component) # make it work with from __future__ import annotations
            for f in fields(component):
                assert hints[f.name] is np.ndarray, f"Field '{cn}/{f.name}' not an array: {f}"
                assert f.name not in reserved_names, f"Field '{cn}/{f.name}' in {reserved_names}"
                assert f.metadata.keys() == expected_meta, f"Field '{cn}/{f.name}':\n{f.metadata} vs\n{expected_meta}"
                assert isinstance(f.metadata["shape"], tuple), f.metadata["shape"]
                assert isinstance(fmd := f.metadata["dtype"], str) and fmd in dtypes, f"{fmd} not in {dtypes}"

    def __len__(self):
        return len(self.live_entities)

    def __repr__(self):
        return (f"[World]\n- Entities: {len(self)} (last id: {self._last_id})"
                f"\n- Components ({len(self.component_names)}): {self.component_names}"
                f"\n- Pools: {len(self.pools)}\n- Command buffer: {len(self._command_buffer)}")
