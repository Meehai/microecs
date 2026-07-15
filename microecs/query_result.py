"""query_result.py - A list of pools seen as a contiguous view. Implements array interface to look like numpy"""
import numpy as np

from .utils import Shape
from .pool import Pool
from .qr_field import QRField

# Note: if QueryResult gets new fields, add them here! Otherwise the user code may overwrite them e.g. qr._len=xxx
QUERY_RESULT_INTERNAL_ATTRS = {"pool_list", "entity_ids", "fields", "_field_shapes",
                               "_field_dtypes", "_data", "_len", "_cache"}

class _QRArray(np.ndarray):
    """small shim array so we don't instantiate QRField which is more expensive (contiguous view for >=2 pools)"""
    def numpy(self) -> np.ndarray:
        """for compatibility with QRField.numpy()"""
        return np.asarray(self)

    @property
    def parts(self) -> list[np.ndarray]:
        """for compatibility with QRField.numpy()"""
        return [np.asarray(self)]

class QueryResult:
    """A query result containing entities. Fields (e.g. qr.position) implement array interface to look like numpy"""
    def __init__(self, pool_list: list[Pool], field_shapes: dict[str, Shape], field_dtypes: dict[str, np.dtype],
                 entity_ids: np.ndarray):
        self.pool_list = pool_list
        self.entity_ids = entity_ids
        self.fields = list(field_shapes)
        self._field_shapes = field_shapes
        self._field_dtypes = field_dtypes
        self._data: dict[str, list[np.ndarray]] = {f: [p.data[f][0:len(p)] for p in pool_list]
                                                   for f in field_shapes.keys()}
        self._len = sum(len(pool) for pool in self.pool_list)
        self._cache: dict[str, QRField] = {}

    def __getattr__(self, name):
        data: dict[str, list[np.ndarray]]
        if (data := self.__dict__.get("_data")) is None or name not in data:
            raise AttributeError(f"'{name}' not part of {self.fields}")

        if name not in self._cache:
            if len(parts := data[name]) in (0, 1): # optimized path for a single pool -> return an actual np array
                # the 'or' part is in case no pools match the query so we create a (0, k) array for that field.
                arr = parts[0] if parts else np.empty((0, *self._field_shapes[name]), self._field_dtypes[name])
                self._cache[name] = arr.view(_QRArray)
            else:
                self._cache[name] = QRField(parts)
        return self._cache[name]

    def __setattr__(self, name, value):
        if (data := self.__dict__.get("_data")) is not None and name in data:
            getattr(self, name)[:] = value   # recarray semantics: assigning a field scatters into it
            return
        super().__setattr__(name, value)

    def __iter__(self):
        raise TypeError(("QueryResult is not iterable. Use `qr.field[:] = ..` that applies to all items at once.\n"
                         "Common pattern: `for e in world.query(..): e.f = ..` -> `qr=world.query(..); qr.f[:] = ..`"))

    def __len__(self):
        return self._len

    def __repr__(self):
        return (f"[QueryResult]\n- Entities: {len(self.entity_ids)} (pools: {len(self.pool_list)})"
                f"\n- Fields: {self.fields}"
                f"\n- Shapes: {list(self._field_shapes.values())}\n- Dtypes: {list(self._field_dtypes.values())}")
