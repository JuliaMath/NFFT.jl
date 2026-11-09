"""The non-equidistant fast Fourier transform (NFFT) of NFFT.jl, from a library compiled with
`juliac --trim`; no Julia installation needed.

    import numpy as np, nfftjl
    k = np.random.rand(10_000, 2) - 0.5          # nodes in [-1/2, 1/2)^2, one per row
    p = nfftjl.Plan(k, (256, 256), reltol=1e-8)
    fhat = p.trafo(np.random.randn(256, 256) + 0j)
    f = p.adjoint(fhat)

Direct transform fhat_j = sum_l f_l exp(-2 pi i l.k_j), adjoint f_l = sum_j fhat_j exp(2 pi i l.k_j),
with l_d = -N_d//2, ..., N_d - N_d//2 - 1 along axis d of f. RNFFTPlan, NFCTPlan and NFSTPlan are the
transforms with real values at the nodes, the cosine and the sine transform, NNFFTPlan the
transform with arbitrary real nodes in both domains.
"""

import ctypes
import os
import sys

import numpy as np

__all__ = ["Plan", "RNFFTPlan", "NFCTPlan", "NFSTPlan", "NNFFTPlan", "num_threads"]

_ext = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "_lib", "lib", "libnfftjl" + _ext))

_p = ctypes.c_void_p
_i64 = ctypes.c_int64


# The C functions of a transform in one precision: nfftjl_… (double) or nfftjlf_… (single), with
# the transform in the name except for the NFFT (nfftjl_rnfft_trafo, nfftjl_trafo).
def _functions(kind, real):
    prefix = ("nfftjl_" if real == np.float64 else "nfftjlf_") + ("" if kind == "nfft" else kind + "_")
    f = {name: getattr(_lib, prefix + name) for name in ("plan_create_many", "trafo", "adjoint")}
    f["plan_create_many"].argtypes = ([ctypes.c_int, _i64, _p, _i64, _p, _i64, _i64, ctypes.c_double, ctypes.c_int]
                                      if kind == "nnfft" else
                                      [ctypes.c_int, _p, _i64, _p, _i64, _i64, ctypes.c_double, ctypes.c_int])
    f["plan_create_many"].restype = _i64
    for name in ("trafo", "adjoint"):
        f[name].argtypes = [_i64, _p, _p]
        f[name].restype = ctypes.c_int
    return f


_lib.nfftjl_plan_destroy.argtypes = [_i64]
_lib.nfftjl_plan_destroy.restype = ctypes.c_int
_lib.nfftjl_last_error.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
_lib.nfftjl_last_error.restype = ctypes.c_size_t
_lib.nfftjl_num_threads.argtypes = []
_lib.nfftjl_num_threads.restype = ctypes.c_int

# The Julia runtime of the library starts on the first call into it; two threads making that first
# call at the same time abort the process. Importing makes the first call.
_lib.nfftjl_num_threads()


def num_threads():
    """The default number of threads of a plan: the environment variable NFFTJL_NUM_THREADS if
    set, otherwise the number of performance cores (Apple Silicon) or of all CPU threads."""
    return _lib.nfftjl_num_threads()


class _Plan:
    # set by the subclasses: the transform, and the element types of input and output for the
    # real type (float64 or float32)
    _kind = None

    @staticmethod
    def _eltypes(real):
        raise NotImplementedError

    def __init__(self, k, shape, reltol, nthreads, real, axes):
        self.dtype = np.dtype(real)
        if self.dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
            raise ValueError(f"precision must be float64 or float32, got {self.dtype}")
        self.in_dtype, self.out_dtype = (np.dtype(t) for t in self._eltypes(self.dtype.type))
        self._f = _functions(self._kind, self.dtype.type)
        self.shape = (int(shape),) if np.isscalar(shape) else tuple(int(n) for n in shape)
        n = len(self.shape)
        if axes is None:
            axes = tuple(range(n))
        axes = (axes,) if np.isscalar(axes) else tuple(axes)
        axes = tuple(a % n for a in axes)
        if not axes or list(axes) != list(range(axes[0], axes[0] + len(axes))):
            raise ValueError(f"axes must be consecutive and increasing, got {axes}")
        self.axes = axes
        a0, a1 = axes[0], axes[-1]
        self.N = self.shape[a0:a1 + 1]
        D = len(self.N)
        k = np.asarray(k, dtype=np.float64)
        if k.ndim == 1:
            k = k[:, None]
        if k.ndim != 2 or k.shape[1] != D:
            raise ValueError(f"nodes must have shape (J, {D}), got {k.shape}")
        self.J = k.shape[0]
        self.in_shape = self._in_shape(self.shape, a1)
        self.out_shape = self.shape[:a0] + (self.J,) + self.shape[a1 + 1:]
        nthreads = 0 if nthreads is None else int(nthreads)
        # The library is column major: a C-ordered array of shape s is its array of shape
        # reversed(s). The transformed sizes and the node coordinates are reversed to match (rows of
        # k in C order are then the columns of its D x J node matrix), the axes after the
        # transformed ones become the batch before them (L1) and vice versa (L3). No copy of the
        # data per transform.
        k = np.ascontiguousarray(k[:, ::-1])
        Nc = (ctypes.c_int64 * D)(*reversed(self.N))
        L1 = int(np.prod(self.shape[a1 + 1:], dtype=np.int64))
        L3 = int(np.prod(self.shape[:a0], dtype=np.int64))
        self._h = self._f["plan_create_many"](D, ctypes.addressof(Nc), self.J, k.ctypes.data, L1, L3,
                                              reltol, nthreads)
        self._check_plan()

    def _check_plan(self):
        if self._h < 0:
            buf = ctypes.create_string_buffer(512)
            _lib.nfftjl_last_error(buf, len(buf))
            raise ValueError(buf.value.decode() or "plan creation failed")

    @staticmethod
    def _in_shape(shape, a1):
        return shape

    def trafo(self, f, out=None):
        """fhat (shape `out_shape`: the transformed axes replaced by the J nodes) from f (shape
        `in_shape`)."""
        f = np.ascontiguousarray(f, dtype=self.in_dtype)
        if f.shape != self.in_shape:
            raise ValueError(f"f must have shape {self.in_shape}, got {f.shape}")
        out = np.empty(self.out_shape, dtype=self.out_dtype) if out is None else out
        if out.shape != self.out_shape or out.dtype != self.out_dtype or not out.flags.c_contiguous:
            raise ValueError(f"out must be a contiguous {self.out_dtype} array of shape {self.out_shape}")
        self._check(self._f["trafo"](self._h, f.ctypes.data, out.ctypes.data))
        return out

    def adjoint(self, fhat, out=None):
        """f (shape `in_shape`) from fhat (shape `out_shape`); for the NFCT and NFST the
        transpose."""
        fhat = np.ascontiguousarray(fhat, dtype=self.out_dtype)
        if fhat.shape != self.out_shape:
            raise ValueError(f"fhat must have shape {self.out_shape}, got {fhat.shape}")
        out = np.empty(self.in_shape, dtype=self.in_dtype) if out is None else out
        if out.shape != self.in_shape or out.dtype != self.in_dtype or not out.flags.c_contiguous:
            raise ValueError(f"out must be a contiguous {self.in_dtype} array of shape {self.in_shape}")
        self._check(self._f["adjoint"](self._h, fhat.ctypes.data, out.ctypes.data))
        return out

    def _check(self, rc):
        if rc != 0:
            raise RuntimeError("transform failed (plan destroyed?)")

    def close(self):
        """Release the plan (also done when the plan is garbage collected)."""
        if getattr(self, "_h", -1) >= 0:
            _lib.nfftjl_plan_destroy(self._h)
            self._h = -1

    def __del__(self):
        self.close()


class Plan(_Plan):
    """NFFT plan for the nodes `k` (shape (J, D), or (J,) in 1D; entries in [-1/2, 1/2)) and data
    of shape `shape` (an integer or a tuple), transformed along `axes` (default: all axes; D = 1, 2
    or 3 consecutive axes, column d of k belongs to axes[d]). The other axes are batch axes: every
    slice along them is transformed with the same nodes, and they appear unchanged in the result.

    `reltol` is the requested relative error, `nthreads` the number of threads (default:
    `num_threads()`), `dtype` the element type of the data, `numpy.complex128` or
    `numpy.complex64` (the nodes are given in double precision either way). One plan must not be
    executed by two threads at once; separate plans can run concurrently.
    """
    _kind = "nfft"

    @staticmethod
    def _eltypes(real):
        return (np.complex128, np.complex128) if real == np.float64 else (np.complex64, np.complex64)

    def __init__(self, k, shape, reltol=1e-8, nthreads=None, dtype=np.complex128, axes=None):
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.complex128), np.dtype(np.complex64)):
            raise ValueError(f"dtype must be complex128 or complex64, got {dtype}")
        super().__init__(k, shape, reltol, nthreads, np.float64 if dtype == np.complex128 else np.float32, axes)
        self.dtype = dtype


class RNFFTPlan(_Plan):
    """RNFFT plan (real values at the nodes): fhat_j = Re sum_n h_n exp(-2 pi i n.k_j), with
    n = 0, ..., N//2 along the last transformed axis and n = -N//2, ..., N - N//2 - 1 along the
    others; the adjoint is h_n = sum_j fhat_j exp(2 pi i n.k_j). h has the shape `shape` with the
    last transformed axis replaced by N//2 + 1 (as `numpy.fft.rfftn`), complex; fhat is real.
    Nodes in [-1/2, 1/2). `dtype` is the real type, `numpy.float64` or `numpy.float32`; the other
    arguments as for `Plan`.
    """
    _kind = "rnfft"

    @staticmethod
    def _eltypes(real):
        return (np.complex128, np.float64) if real == np.float64 else (np.complex64, np.float32)

    @staticmethod
    def _in_shape(shape, a1):
        return shape[:a1] + (shape[a1] // 2 + 1,) + shape[a1 + 1:]

    def __init__(self, k, shape, reltol=1e-8, nthreads=None, dtype=np.float64, axes=None):
        super().__init__(k, shape, reltol, nthreads, dtype, axes)


class NFCTPlan(_Plan):
    """NFCT plan: fhat_j = sum_n f_n prod_d cos(2 pi n_d k_dj), n_d = 0, ..., N_d - 1, for real f;
    `adjoint` is the transpose. Nodes in [0, 1/2]. `dtype` is the real type, `numpy.float64` or
    `numpy.float32`; the other arguments as for `Plan`.
    """
    _kind = "nfct"

    @staticmethod
    def _eltypes(real):
        return (real, real)

    def __init__(self, k, shape, reltol=1e-8, nthreads=None, dtype=np.float64, axes=None):
        super().__init__(k, shape, reltol, nthreads, dtype, axes)


class NFSTPlan(NFCTPlan):
    """NFST plan: fhat_j = sum_n f_n prod_d sin(2 pi n_d k_dj), n_d = 1, ..., N_d, for real f;
    `adjoint` is the transpose. Nodes in [0, 1/2]. `dtype` is the real type, `numpy.float64` or
    `numpy.float32`; the other arguments as for `Plan`.
    """
    _kind = "nfst"


class NNFFTPlan(_Plan):
    """NNFFT plan (nonequispaced in both domains): fhat_j = sum_l f_l exp(-2 pi i k_j.y_l) for
    targets `k` (shape (J, D), or (J,) in 1D) and sources `y` (shape (N, D) or (N,)), arbitrary
    real coordinates, D = 1, 2 or 3; the adjoint is f_l = sum_j fhat_j exp(2 pi i k_j.y_l). The cost
    grows with the product of the extents of k and y, not with J and N.

    `shape` is the shape of the data (default (N,)) and `axis` the axis of the sources in it; the
    other axes are batch axes, and the result has the axis of length J in its place. `reltol`,
    `nthreads` and `dtype` as for `Plan`.
    """
    _kind = "nnfft"

    @staticmethod
    def _eltypes(real):
        return (np.complex128, np.complex128) if real == np.float64 else (np.complex64, np.complex64)

    def __init__(self, k, y, shape=None, reltol=1e-8, nthreads=None, dtype=np.complex128, axis=-1):
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.complex128), np.dtype(np.complex64)):
            raise ValueError(f"dtype must be complex128 or complex64, got {dtype}")
        self.dtype = dtype
        real = np.float64 if dtype == np.complex128 else np.float32
        self.in_dtype = self.out_dtype = dtype
        self._f = _functions(self._kind, real)
        k = np.asarray(k, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        k = k[:, None] if k.ndim == 1 else k
        y = y[:, None] if y.ndim == 1 else y
        if k.ndim != 2 or y.ndim != 2 or k.shape[1] != y.shape[1] or not 1 <= k.shape[1] <= 3:
            raise ValueError(f"targets and sources must have shapes (J, D) and (N, D), D = 1, 2 or 3, "
                             f"got {k.shape} and {y.shape}")
        D = k.shape[1]
        self.J, self.N = k.shape[0], y.shape[0]
        self.shape = (self.N,) if shape is None else ((int(shape),) if np.isscalar(shape) else tuple(int(n) for n in shape))
        a = int(axis) % len(self.shape)
        if self.shape[a] != self.N:
            raise ValueError(f"axis {a} of the shape {self.shape} must have the N = {self.N} sources")
        self.axis = a
        self.in_shape = self.shape
        self.out_shape = self.shape[:a] + (self.J,) + self.shape[a + 1:]
        # Column major in the library (see _Plan): the axes after `axis` are the batch L1, those
        # before it L3; the D x J and D x N node matrices are the rows of k and y in C order.
        k = np.ascontiguousarray(k)
        y = np.ascontiguousarray(y)
        L1 = int(np.prod(self.shape[a + 1:], dtype=np.int64))
        L3 = int(np.prod(self.shape[:a], dtype=np.int64))
        nthreads = 0 if nthreads is None else int(nthreads)
        self._h = self._f["plan_create_many"](D, self.J, k.ctypes.data, self.N, y.ctypes.data, L1, L3, reltol, nthreads)
        self._check_plan()
