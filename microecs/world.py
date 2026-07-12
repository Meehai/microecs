"""world.py - The world container for ECS. It manages all the pools (one per archetype). Entities are id-based."""
from typing import get_type_hints
from dataclasses import fields
import numpy as np

from .pool import Pool
from .utils import Shape, EntityId, PoolKey, Command, CommandType, logger
from .component import ComponentType
from .query_result import QueryResult, QUERY_RESULT_INTERNAL_ATTRS
from .entity import Entity, ENTITY_INTERNAL_ATTRS
from .command_buffer import CommandBuffer

class World:
    """
    Generic container for pools of components. Newly added components are assigned a unique id and go in a pool
    Parameters
    - components The list of components that the world accepts
    - extra_metadata The list of required extra metadata for each field besides shape and dtype.
    """
    def __init__(self, components: list[ComponentType], extra_metadata: list[str] | None = None):
        self._default_metadata = {"shape", "dtype", "default"}
        self.extra_metadata = extra_metadata or []
        assert isinstance(self.extra_metadata, list), type(self.extra_metadata)
        self._check_components(components)

        # pools management
        self.pools: dict[PoolKey, Pool] = {}
        self.pool_to_components: dict[Pool, list[ComponentType]] = {}

        # components management
        self.component_names = [x.__name__ for x in components]
        self.component_types = set(components)
        self.component_name_to_type = {x.__name__: x for x in components}
        self.component_to_bit: dict[ComponentType, int] = {t: 2**i for i, t in enumerate(components)} # bit for querying
        self.component_to_field_names: dict[ComponentType, list[str]] = {c: [] for c in components}
        self.component_to_shapes: dict[ComponentType, list[Shape]] = {c: [] for c in components}
        self.component_to_dtypes: dict[ComponentType, list[str]] = {c: [] for c in components}
        self.component_to_defaults: dict[ComponentType, list[np.ndarray]] = {c: [] for c in components}
        # setup the obligatory metadata at each fields
        for c in components:
            for f in fields(c):
                self.component_to_field_names[c].append(f.name)
                self.component_to_shapes[c].append(field_shape := f.metadata["shape"])
                self.component_to_dtypes[c].append(field_dtype := f.metadata["dtype"])
                self.component_to_defaults[c].append(field_default := f.metadata["default"])
                if field_default is not None:
                    if (dt := field_default.dtype) != field_dtype:
                        raise TypeError(f"'{c.__name__}/{f.name}'. Expected dtype: {field_dtype}. Got: {dt}")
                    if (sh := field_default.shape) != field_shape:
                        raise ValueError(f"'{c.__name__}/{f.name}'. Expected shape: {field_shape}. Got: {sh}")

        # entities management
        self._eid_to_pool_ix: dict[EntityId, tuple[Pool, int]] = {}
        self._pool_ids: dict[Pool, list[EntityId]] = {}
        self._last_id: EntityId = -1
        # a dictionary of all live entities in 'eager' mode (before update()). The actual entity is created at request
        # in get_entity, so we don't pay for the Entity object unless it's explicitly requested by the user.
        self.live_entities: dict[EntityId, Entity | None] = {}

        # command buffer management. {add/remove}_{entity/component} are lazy. Taken into account after update().
        self._command_buffer = CommandBuffer(self)
        self._cache: dict[tuple[PoolKey, PoolKey], QueryResult] = {} # include+exclude key
        logger.debug(f"Created scene with components: {self.component_names}")

    # public api

    def add_entity(self, components: list[ComponentType], **kwargs) -> EntityId:
        """Adds an entity to the world based on components (data->kwargs). Returns an entity id. Lazy; call update()"""
        self._validate_components(components, **kwargs)
        default_kwargs = self._defaults_for(components, **kwargs)
        self._last_id += 1
        self.live_entities[self._last_id] = None # add the id the live_entities, but the object is created in get_entity
        self._command_buffer.append(Command(CommandType.ADD_ENTITY, self._last_id,
                                            args={"components": components, **kwargs, **default_kwargs}))
        return self._last_id

    def remove_entity(self, entity_id: EntityId):
        """Removes an entity based on its unique entity id. Lazy; call update()"""
        self._command_buffer.append(Command(CommandType.REMOVE_ENTITY, entity_id))
        del self.live_entities[entity_id]

    def get_entity(self, entity_id: EntityId) -> Entity:
        """Gets the entity reference given an entity id. Used for 'object-like' ops. Lazy; call world.update()"""
        if entity_id not in self.live_entities:
            raise ValueError(f"Entity id: {entity_id} not in the world")

        if self.live_entities[entity_id] is None:
            # only instantiate on first request, so the object is not created for no reason at add_entity time.
            self.live_entities[entity_id] = Entity(entity_id, self._eid_to_pool_ix, self.pool_to_components,
                                                   world_command_buffer=self._command_buffer)

        return self.live_entities[entity_id]

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

    def update(self):
        """commits the underlying pool changes from the systems between two updates. Should be called in main loop."""
        for command in self._command_buffer:
            if command.command_type == CommandType.ADD_ENTITY:
                components = command.args.pop("components")
                self._add_to_pool(command.entity_id, components=components, **command.args)
            elif command.command_type == CommandType.REMOVE_ENTITY:
                self._pop_from_pool(command.entity_id)
            elif command.command_type == CommandType.ADD_COMPONENT:
                component = command.args.pop("component")
                self._do_add_component(command.entity_id, component=component, **command.args)
            else: # CommandType.REMOVE_COMPONENT
                self._do_remove_component(command.entity_id, component=command.args)

        if len(self._command_buffer) > 0:
            self._command_buffer.clear()
            self._cache.clear()

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
        default_kwargs = self._defaults_for(new_components, **entity_data, **kwargs)
        self._add_to_pool(entity_id, new_components, **entity_data, **kwargs, **default_kwargs)

    def _do_remove_component(self, entity_id: EntityId, component: ComponentType):
        entity_data, components = self._pop_from_pool(entity_id)
        for _field in self.component_to_field_names[component]:
            assert _field in entity_data, f"Field {component}/{_field} not in components: {components} ({entity_id=})"
            entity_data.pop(_field)
        new_components = [c for c in components if c != component]
        self._add_to_pool(entity_id, new_components, **entity_data)

    # other low-level methods

    def _validate_components(self, components: list[ComponentType], **kwargs):
        """Pure check. Raises on: no components, unknown component, missing-required (default=None),
            wrong dtype/shape, extra field. No mutation, no return. kwargs == fields data."""
        if len(cs := set(components)) == 0:
            raise ValueError(f"Entity has no components: {self.component_names}")
        if diff := cs - self.component_types:
            raise ValueError(f"Unknown components: {diff}")
        expected = set()
        for c in components:
            for name, shape, dtype, default in zip(self.component_to_field_names[c], self.component_to_shapes[c],
                                                    self.component_to_dtypes[c], self.component_to_defaults[c]):
                expected.add(name)
                if name not in kwargs:
                    if default is None:
                        raise KeyError(f"'{c.__name__}/{name}' required (default=None) but not supplied")
                    continue                      # omitted but has a default -> fine
                if not isinstance(field := kwargs[name], np.ndarray):
                    raise TypeError(f"'{c.__name__}/{name}'. Expected np.ndarray, got {type(field)}")
                if (dt := field.dtype) != dtype:
                    raise TypeError(f"'{c.__name__}/{name}'. Expected dtype {dtype}, got {dt}")
                if (sh := field.shape) != shape:
                    raise ValueError(f"'{c.__name__}/{name}'. Expected shape {shape}, got {sh}")
        if extra := set(kwargs) - expected:
            raise ValueError(f"Extra fields: {extra}; expected {expected}")

    def _defaults_for(self, components: list[ComponentType], **kwargs) -> dict[str, np.ndarray]:
        """Defaults for omitted fields. Assumes already validated. No mutation of `kwargs` (data) is done."""
        res = {}
        for c in components:
            for name, default in zip(self.component_to_field_names[c], self.component_to_defaults[c]):
                if name not in kwargs and default is not None:
                    res.setdefault(c, {})[name] = default.copy()
        return res

    def _get_entity_pool(self, components: list[ComponentType]) -> Pool:
        if (key := self._make_key(components)) not in self.pools:
            _fields = sum([self.component_to_field_names[c] for c in components], []) # merge fields
            shapes = sum([self.component_to_shapes[c] for c in components], []) # merge shapes
            dtypes = sum([self.component_to_dtypes[c] for c in components], []) # merge dtypes
            self.pools[key] = Pool(_fields, shapes, dtypes)
            self.pool_to_components[self.pools[key]] = components
        return self.pools[key]

    def _make_key(self, components: list[ComponentType]) -> PoolKey:
        key = 0
        for c in components:
            assert c in self.component_types, f"c '{c.__name__}' not in {self.component_names}"
            key |= self.component_to_bit[c]
        return key

    def _check_components(self, components: list[ComponentType]):
        reserved_names = (ENTITY_INTERNAL_ATTRS | {n for n in vars(Entity) if not n.startswith("__")} |
                          QUERY_RESULT_INTERNAL_ATTRS | {n for n in vars(QueryResult) if not n.startswith("__")})
        dtypes = {"float32", "int32", "bool", "object"}
        expected_meta = {*self._default_metadata, *self.extra_metadata}

        for c in components:
            cn = c.__name__
            assert hasattr(c, "__dataclass_fields__"), f"c '{cn}' is not a dataclass"
            hints = get_type_hints(c) # make it work with from __future__ import annotations
            for f in fields(c):
                assert hints[f.name] is np.ndarray, f"Field '{cn}/{f.name}' not an array: {f}"
                assert f.name not in reserved_names, f"Field '{cn}/{f.name}' in {reserved_names}"
                assert f.metadata.keys() == expected_meta, (
                    f"Field '{cn}/{f.name}'\n{list(f.metadata.keys())}\nvs\n{expected_meta}\n"
                    "Perhaps missing World(extra_metadata=[...])?")
                assert isinstance(f.metadata["shape"], tuple), f.metadata["shape"]
                assert isinstance(fmd := f.metadata["dtype"], str) and fmd in dtypes, f"{fmd} not in {dtypes}"

    def __len__(self):
        return len(self.live_entities)

    def __repr__(self):
        return (f"[World]\n- Entities: {len(self)} (last id: {self._last_id})"
                f"\n- Components ({len(self.component_names)}): {self.component_names}"
                f"\n- Pools: {len(self.pools)}\n- Command buffer: {len(self._command_buffer)}")
