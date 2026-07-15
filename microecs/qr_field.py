"""qr_field.py - Query Result Field. A single field that implements numpy's interface for interop"""
from typing import Callable, T
import numpy as np
from microecs.utils import Shape

class QRField(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Field is a single field (column) from a QueryResult object obtained from world.query(...).
    Parts may come from different Pools, so we try to make a contiguius-like view from discountinous arrays.
    """
    def __init__(self, parts: list[np.ndarray]):
        assert len(parts) >= 2, "Cannot instantiate QRField with a single part. Use QRArray for that optimized path"
        self.parts = parts
        self._lens = [len(p) for p in self.parts]
        self.len = sum(self._lens)
        self.shape: Shape = (len(self), *self.parts[0].shape[1:])
        self.dtype = self.parts[0].dtype
        self._bounds: np.ndarray | None = None

    def numpy(self) -> np.ndarray:
        """Creates a numpy array from the underlying pool parts. For len==1, we return the same object!"""
        return np.concatenate(self.parts) if len(self.parts) != 1 else self.parts[0]

    def _chunk(self, x: T, i: int) -> T:
        if isinstance(x, QRField):
            return x.parts[i]
        if isinstance(x, np.ndarray) and x.ndim == len(self.shape) and x.shape[0] == self.len:
            if self._bounds is None: # lazy instantiate because it's expensive to do this unless needed.
                self._bounds = np.cumsum([0, *self._lens])
            return x[self._bounds[i]:self._bounds[i + 1]]
        return x

    def _apply_fn_on_parts(self, fn: Callable, op_args: list, **kwargs):
        # op args can be 1 element (-qr.velocity), 2 elements (qr.position * 0.1), 3 elements (np.where(a, b, c)), etc.
        # all of them must be chunked based on how many we have in this Field so each subpart is called independently.

        results = []
        for i, part in enumerate(self.parts):
            pool_args = [self._chunk(x, i) for x in op_args]
            part_result: QRField = fn(*pool_args, **kwargs)
            # we expect f(arr(N, ...)) -> arr(N, ...) where N = number of items in the pool
            # for e.g. np.linalg.norm(velocity, axis=1) should do (N, 2) -> (N, 1) so the first axis is preserved
            assert len(part_result) == part.shape[0], f"Result: {part_result.shape} vs {part.shape}"
            results.append(part_result)
        return QRField(results)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """wrapper for elementwise (python) primitives, e.g. qr.position[:] += 1"""
        if method != "__call__":
            return NotImplemented
        if out is None:
            return self._apply_fn_on_parts(ufunc, inputs, **kwargs)

        assert len(out) == 1 and isinstance(out[0], QRField), out
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
            raise TypeError("entity-axis assignment crosses pools; use [:] or [:, k] or [i][...]")

        if isinstance(value, QRField):
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
        if key is Ellipsis or (isinstance(key, tuple) and key and (key[0] is Ellipsis or key[0] == slice(None))):
            return QRField([part[key] for part in self.parts])

        raise TypeError(("Only batch updates are supported, e.g. `qr.attr[:]=xxx` or `qr.attr[:, k]=xxx`. "
                         "Use .numpy() for a proper array. For entity-level ops use `world.get_entity(eid).attr=xxx`"))

    def __iter__(self):
        for part in self.parts:
            yield from part

    def __len__(self):
        return self.len

    def __repr__(self):
        return f"[Field] Shape: {self.shape} (across {len(self.parts)} pools)"
