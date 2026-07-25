"""query_result.py - A list of pools seen as a contiguous view. Implements array interface to look like numpy"""
import numpy as np

from .utils import Shape
from .pool import Pool
from .qr_field import QRField

# Note: if QueryResult gets new fields, add them here! Otherwise the user code may overwrite them e.g. qr._data=xxx
_QR_INTERNAL_ATTRS = {"pool_list", "entity_ids", "fields", "_field_shapes", "_field_dtypes", "_data", "_cache"}

class _QRArray(np.ndarray):
    """small shim array so we don't instantiate QRField which is more expensive (contiguous view for >=2 pools)"""
    def numpy(self) -> np.ndarray:
        """for compatibility with QRField.numpy()"""
        return np.asarray(self)

    @property
    def parts(self) -> list[np.ndarray]:
        """for compatibility with QRField.parts"""
        return [np.asarray(self)]

class QueryResult:
    """A query result containing entities. Fields (e.g. qr.position) implement array interface to look like numpy"""
    def __init__(self, pool_list: list[Pool], field_shapes: dict[str, Shape], field_dtypes: dict[str, np.dtype],
                 entity_ids: np.ndarray):
        assert len(entity_ids) == sum(len(p) for p in pool_list), (entity_ids, pool_list)
        self.pool_list = pool_list
        self.entity_ids = entity_ids
        self.fields = list(field_shapes)
        self._field_shapes = field_shapes
        self._field_dtypes = field_dtypes
        self._data: dict[str, list[np.ndarray]] = {f: [p.data[f][0:len(p)] for p in pool_list] for f in field_shapes}
        self._cache: dict[str, QRField | _QRArray] = {}

    def __getattr__(self, name):
        data: dict[str, list[np.ndarray]]
        # .get and not self._data: on an instance built without __init__ (copy/deepcopy/pickle.loads probe
        # hasattr(__setstate__)), _data is MISSING -- and self._data would re-enter __getattr__ -> RecursionError
        if name not in (data := self.__dict__.get("_data", {})):
            raise AttributeError(f"'{name}' not part of {self.__dict__.get('fields')}")

        if name not in self._cache:
            if len(parts := data[name]) in (0, 1): # optimized path for a single pool -> return an actual np array
                # the if/else part is in case no pools match the query so we create a (0, k) array for that field.
                arr = parts[0] if parts else np.empty((0, *self._field_shapes[name]), self._field_dtypes[name])
                self._cache[name] = arr.view(_QRArray)
            else:
                self._cache[name] = QRField(parts)

        return self._cache[name]

    def __setattr__(self, name, value):
        if name in _QR_INTERNAL_ATTRS:
            super().__setattr__(name, value)
            return

        # When is _data None and __setattr__ called? On any instance built without __init__ (copy/deepcopy/pickle.loads)
        if name not in self.__dict__.get("_data", {}): # note: self._data bounces to getattr (recursion).
            raise AttributeError(f"Attribute '{name}' not in query result fields: {self.__dict__.get('fields')}")

        getattr(self, name)[:] = value

    def __iter__(self):
        raise TypeError(("QueryResult is not iterable. Use `qr.field = ..` that applies to all items at once.\n"
                         "Common pattern: `for e in world.query(..): e.attr = X` -> `qr=world.query(..); qr.attr = X`"))

    def __len__(self):
        return len(self.entity_ids)

    def __repr__(self):
        return (f"[QueryResult]\n- Entities: {len(self.entity_ids)} (pools: {len(self.pool_list)})"
                f"\n- Fields: {self.fields}"
                f"\n- Shapes: {list(self._field_shapes.values())}\n- Dtypes: {list(self._field_dtypes.values())}")

QUERY_RESULT_RESERVED_NAMES = _QR_INTERNAL_ATTRS | {n for n in vars(QueryResult) if not n.startswith("__")}
