"""world.py - The world container for ECS. It manages all the pools (one per archetype). Entities are id-based."""
import numpy as np
from loggez import loggez_logger as logger

from .pool import Pool, PoolKey, Shape

class World:
    """Generic container for pools of components. Newly added components are assigned a unique id and go in a pool"""
    def __init__(self, traits: list[type]):
        self._check_traits(traits)
        self.trait_names = [x.__name__ for x in traits]
        self.trait_types = list(traits)
        self.trait_to_bit: dict[type, int] = {t: 2**i for i, t in enumerate(traits)} # assign unique bit for querying
        self.trait_to_shapes: dict[type, list[Shape]] = {
            t: [f.metadata["shape"] for f in t.__dataclass_fields__.values()] for t in traits}
        self.trait_to_dtypes: dict[type, list[str]] = {
            t: [f.metadata["dtype"] for f in t.__dataclass_fields__.values()] for t in traits}
        self.trait_to_field_names: dict[type, list[str]] = {t: list(t.__dataclass_fields__) for t in traits}
        self.pools: dict[PoolKey, Pool] = {}
        logger.info(f"Created scene with traits: {self.trait_names}")

    def add_entity(self, traits: list[type], **kwargs):
        self._get_entity_pool(traits, **kwargs).add_entity(**kwargs)

    def query_and(self, trait_types: list[type]) -> list[Pool]:
        """returns all the entities that have all the traits"""
        key = self._make_key(trait_types)
        res = []
        for archetype_key, archetype_pool in self.pools.items():
            if (archetype_key & key) == key: # key is subset of archetype_key
                res.append(archetype_pool)
        return res

    def _get_entity_pool(self, traits: list[type], **entity_fields) -> Pool:
        assert len(traits) > 0, f"Entity has no traits: {self.trait_names}"
        expected_fields = set()
        for trait in traits:
            for _field in self.trait_to_field_names[trait]:
                expected_fields.add(_field)
                assert _field in entity_fields, f"Entity doenst't have '{trait}/{_field}'"
        assert (extra := (set(entity_fields) - expected_fields)) == set(), f"Extra fields: {extra}; {expected_fields=}"

        if (key := self._make_key(traits)) not in self.pools:
            fields = sum([self.trait_to_field_names[trait] for trait in traits], []) # merge of all the fields
            shapes = sum([self.trait_to_shapes[trait] for trait in traits], []) # merge of all the shapes
            dtypes = sum([self.trait_to_dtypes[trait] for trait in traits], []) # merge of all the dtypes
            self.pools[key] = Pool(fields, shapes, dtypes)
        return self.pools[key]

    def _make_key(self, traits: list[type]) -> PoolKey:
        key = 0
        for trait in traits:
            assert trait in self.trait_types, f"Trait '{trait.__name__}' not in {self.trait_names}"
            key |= self.trait_to_bit[trait]
        return key

    def _check_traits(self, traits: list[type]):
        _dtypes = {"float32", "int32", "bool", "str"}
        for trait in traits:
            assert hasattr(trait, "__dataclass_fields__"), f"Trait '{trait}' is not a dataclass"
            for field_name, _field in trait.__dataclass_fields__.items():
                assert _field.type == np.ndarray, f"Field '{field_name}' of '{trait=}' not an array: {_field}"
                assert _field.metadata.keys() == {"shape", "dtype"}, _field.metadata
                assert isinstance(_field.metadata["shape"], tuple), _field.metadata["shape"]
                assert isinstance(fmd := _field.metadata["dtype"], str) and fmd in _dtypes, f"{fmd} not in {_dtypes}"
