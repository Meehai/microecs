"""query_result.py - A list of pools seen as a contiguous view. Implements array interface to look like numpy"""
from typing import Callable, T
import numpy as np

from .utils import Shape
from .pool import Pool

class _Field(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, parts: list[np.ndarray]):
        self.parts = parts
        self._lens = [len(p) for p in self.parts]
        self.len = sum(self._lens)
        self.shape: Shape = (len(self), *self.parts[0].shape[1:])
        self._bounds = np.cumsum([0, *self._lens])

    def numpy(self) -> np.ndarray:
        """Creates a numpy array from the underlying pool parts"""
        return np.concatenate(self.parts)

    def _chunk(self, x: T, i: int) -> T:
        if isinstance(x, _Field):
            return x.parts[i]
        if isinstance(x, np.ndarray) and x.ndim == len(self.shape) and x.shape[0] == self.len:
            return x[self._bounds[i]:self._bounds[i + 1]]
        return x

    def _apply_fn_on_parts(self, fn: Callable, op_args: list, **kwargs):
        # op args can be 1 element (-qr.velocity), 2 elements (qr.position * 0.1), 3 elements (np.where(a, b, c)), etc.
        # all of them must be chunked based on how many we have in this _Field so each subpart is called independently.
        results = []
        for i, part in enumerate(self.parts):
            pool_args = [self._chunk(x, i) for x in op_args]
            part_result: _Field = fn(*pool_args, **kwargs)
            # we expect f(arr(N, ...)) -> arr(N, ...) where N = number of items in the pool
            # for e.g. np.linalg.norm(velocity, axis=1) should do (N, 2) -> (N, 1) so the first axis is preserved
            assert len(part_result) == part.shape[0], f"Result: {part_result.shape} vs {part.shape}"
            results.append(part_result)
        return _Field(results)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """wrapper for elementwise (python) primitives, e.g. qr.position[:] += 1"""
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
        return self._apply_fn_on_parts(func, args, **kwargs)

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

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):       # qr.position[i] -> entity i's row, from its own pool
            key = range(self.len)[key]               # numpy's exact int-index rule, for free: wraps neg, raises on OOB
            pool_ix = int(np.searchsorted(self._bounds, key, side="right")) - 1
            return self.parts[pool_ix][key - self._bounds[pool_ix]]

        if not (isinstance(key, tuple) and key and key[0] == slice(None)):
            raise TypeError("Unsupported indexing. Use .numpy() for a proper array. To set items, use qr.attr[:, k]=xxx"
                            ". For fancy indexing qr[i, 0:3, k], use qr[i][0:3, k].")

        return _Field([part[key] for part in self.parts])

    def __iter__(self):
        for part in self.parts:
            yield from part

    def __len__(self):
        return self.len

    def __repr__(self):
        return f"[Field] Shape: {self.shape} (across {len(self.parts)} pools)"

class QueryResult:
    """A list of pools seen as a contiguous view. Fields (qr.position) implement array interface to look like numpy"""
    def __init__(self, pool_list: list[Pool], field_shapes: dict[str, Shape], field_dtypes: dict[str, np.dtype],
                 entity_ids: np.ndarray):
        self.pool_list = pool_list
        self.entity_ids = entity_ids
        self._field_shapes = field_shapes
        self._field_dtypes = field_dtypes
        self._fields = list(field_shapes)
        self._data: dict[str, np.ndarray] = {f: [p.data[f][0:len(p)] for p in pool_list] for f in field_shapes.keys()}
        self._len = sum(len(pool) for pool in self.pool_list)

    def __getattr__(self, name):
        if (data := self.__dict__.get("_data")) is not None and name in data:
            # the 'or' part is in case no pools match the query and we want qr.position[:] += 1 still to work (noop)
            return _Field(data[name] or [np.empty((0, *self._field_shapes[name]), self._field_dtypes[name])])
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if (data := self.__dict__.get("_data")) is not None and name in data:
            getattr(self, name)[:] = value   # recarray semantics: assigning a field scatters into it
            return
        super().__setattr__(name, value)

    def __len__(self):
        return self._len

    def __repr__(self):
        return (f"[QueryResult]\n- Entities: {len(self.entity_ids)} (pools: {len(self.pool_list)})"
                f"\n- Fields: {self._fields}"
                f"\n- Shapes: {list(self._field_shapes.values())}\n- Dtypes: {list(self._field_dtypes.values())}")
