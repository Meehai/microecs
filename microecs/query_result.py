"""query_result.py - A list of pools seen as a contiguous view. Implements array interface to look like numpy"""
from typing import Callable, T
import numpy as np

from .utils import Shape
from .pool import Pool

class _Field(np.lib.mixins.NDArrayOperatorsMixin):
    _PER_POOL_OK = {np.where, np.clip}

    def __init__(self, parts: list[np.ndarray]):
        self.parts = parts
        self._lens = [len(p) for p in self.parts]
        self.len = sum(self._lens)
        self.shape: Shape = (len(self), *self.parts[0].shape[1:])
        self._bounds = np.cumsum([0, *self._lens])

    def numpy(self) -> np.ndarray:
        """Creates a numpy array from the underlying pool parts"""
        return np.concatenate(self.parts)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """wrapper forelementwise (python) primitives, e.g. qr.position[:] += 1"""
        if method != "__call__":
            return NotImplemented
        if out is None:
            return self._apply_fn_on_parts(ufunc, inputs, **kwargs)

        assert len(out) == 1 and isinstance(out[0], _Field), out
        for i in range(len(self.parts)):
            pool_args = [self._chunk(x, i) for x in inputs]
            ufunc(*pool_args, out=out[0].parts[i], **kwargs)
        return out[0]

    def __array_function__(self, func: Callable, _types, args: list, kwargs: dict):
        """wrapper for elementwise numpy functions, e.g. qr.velocity[:] = np.where(mask, -qr.velocity, qr.velocity)"""
        if func not in self._PER_POOL_OK:
            return NotImplemented # add them manually
        return self._apply_fn_on_parts(func, args, **kwargs)

    def _chunk(self, x: T, i: int) -> T:
        if isinstance(x, _Field):
            return x.parts[i]
        if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == self.len:   # full-N raw -> slice per pool
            return x[self._bounds[i]:self._bounds[i + 1]]
        return x

    def _apply_fn_on_parts(self, fn: Callable, op_args: list, **kwargs):
        results = []
        for i in range(len(self.parts)):
            pool_args = [self._chunk(x, i) for x in op_args]
            results.append(fn(*pool_args, **kwargs))
        return _Field(results)

    # qr.position[:] = <field | scalar | per-entity broadcast>   -> scatter through the views
    def __setitem__(self, key, value):
        if (not (isinstance(key, slice) and key == slice(None)) and
            not (isinstance(key, tuple) and key and key[0] == slice(None))):
            raise TypeError("entity-axis assignment crosses pools; use [:] or [:, k]")

        if isinstance(value, _Field):
            for i, part in enumerate(self.parts):
                part[key] = value.parts[i]
            return

        # follow numpy's rules for broadcasting
        views = [part[key] for part in self.parts] # per-pool destinations (views)
        logical = (self.len, *views[0].shape[1:]) # the (N, *e) the user "sees"
        full = np.broadcast_to(value, logical) # numpy rules: (*e,)/scalar fill, (N,*e) positional; raises otherwise
        for v, chunk in zip(views, np.split(full, np.cumsum(self._lens)[:-1])):
            v[:] = chunk

    # qr.position[:, 0] (or any >1 axis as well as slices not exact indices) is allowed. qr.position[0] is not.
    def __getitem__(self, key):
        if not (isinstance(key, tuple) and key and key[0] == slice(None)):
            raise TypeError("entity-axis indexing crosses pools; use [:, k]. Use .numpy() to get a proper np.ndarray.")
        return _Field([part[key] for part in self.parts])

    def __iter__(self):
        for part in self.parts:
            yield from part

    def __len__(self):
        return self.len

class QueryResult:
    """A list of pools seen as a contiguous view. Fields (qr.position) implement array interface to look like numpy"""
    def __init__(self, pool_list: list[Pool], field_shapes: dict[str, Shape], field_dtypes: dict[str, np.dtype]):
        self.pool_list = pool_list
        self.field_shapes = field_shapes
        self.field_dtypes = field_dtypes
        self.fields = list(field_shapes)
        self.data: dict[str, np.ndarray] = {f: [p.data[f][0:len(p)] for p in pool_list] for f in field_shapes.keys()}
        self.len = sum(len(pool) for pool in self.pool_list)

    def __getattr__(self, name):
        if (data := self.__dict__.get("data")) is not None and name in data:
            # the 'or' part is in case no pools match the query and we want qr.position[:] += 1 still to work (noop)
            return _Field(data[name] or [np.empty((0, *self.field_shapes[name]), self.field_dtypes[name])])
        raise AttributeError(name)

    def __len__(self):
        return self.len

    def __repr__(self):
        return (f"[QueryResult]\n- Fields: {self.fields}\n- Pools: {len(self.pool_list)}\n- Len: {self.len}"
                f"\n- Shapes: {list(self.field_shapes.values())}\n- Dtypes: {list(self.field_dtypes.values())}")
