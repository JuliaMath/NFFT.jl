"""The non-equidistant fast Fourier transform (NFFT) of NFFT.jl, from a library compiled with
`juliac --trim`; no Julia installation needed.

    import numpy as np, nfftjl
    k = np.random.rand(10_000, 2) - 0.5          # nodes in [-1/2, 1/2)^2, one per row
    p = nfftjl.Plan(k, (256, 256), reltol=1e-8)
    fhat = p.trafo(np.random.randn(256, 256) + 0j)
    f = p.adjoint(fhat)

Direct transform fhat_j = sum_l f_l exp(-2 pi i l.k_j), adjoint f_l = sum_j fhat_j exp(2 pi i l.k_j),
with l_d = -N_d//2, ..., N_d - N_d//2 - 1 along axis d of f.
"""

import ctypes
import os
import sys

import numpy as np

__all__ = ["Plan", "num_threads"]

_ext = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "_lib", "lib", "libnfftjl" + _ext))

_p = ctypes.c_void_p
_lib.nfftjl_plan_create.argtypes = [ctypes.c_int, _p, ctypes.c_int64, _p, ctypes.c_double, ctypes.c_int]
_lib.nfftjl_plan_create.restype = ctypes.c_int64
_lib.nfftjl_trafo.argtypes = [ctypes.c_int64, _p, _p]
_lib.nfftjl_trafo.restype = ctypes.c_int
_lib.nfftjl_adjoint.argtypes = [ctypes.c_int64, _p, _p]
_lib.nfftjl_adjoint.restype = ctypes.c_int
_lib.nfftjl_plan_destroy.argtypes = [ctypes.c_int64]
_lib.nfftjl_plan_destroy.restype = ctypes.c_int
_lib.nfftjl_num_threads.argtypes = []
_lib.nfftjl_num_threads.restype = ctypes.c_int

# The Julia runtime of the library starts on the first call into it; two threads making that first
# call at the same time abort the process. Importing makes the first call.
_lib.nfftjl_num_threads()


def num_threads():
    """Threads of the library, the default and upper limit of `nthreads`."""
    return _lib.nfftjl_num_threads()


class Plan:
    """NFFT plan for the nodes `k` (shape (J, D), or (J,) in 1D; entries in [-1/2, 1/2)) and the
    mode array size `N` (an integer or a tuple of D = 1, 2, 3 integers).

    `reltol` is the requested relative error, `nthreads` the number of threads (default: all
    threads of the library). One plan must not be executed by two threads at once; separate
    plans can run concurrently.
    """

    def __init__(self, k, N, reltol=1e-8, nthreads=None):
        self.N = (int(N),) if np.isscalar(N) else tuple(int(n) for n in N)
        D = len(self.N)
        k = np.asarray(k, dtype=np.float64)
        if k.ndim == 1:
            k = k[:, None]
        if k.ndim != 2 or k.shape[1] != D:
            raise ValueError(f"nodes must have shape (J, {D}), got {k.shape}")
        self.J = k.shape[0]
        nthreads = num_threads() if nthreads is None else int(nthreads)
        # The library is column major: a C-ordered array of shape N is its array of shape
        # reversed(N), with the node coordinates reversed to match; rows of k (C order) are then
        # the columns of its D x J node matrix. No copy of the data per transform.
        k = np.ascontiguousarray(k[:, ::-1])
        Nc = (ctypes.c_int64 * D)(*reversed(self.N))
        self._h = _lib.nfftjl_plan_create(D, ctypes.addressof(Nc), self.J, k.ctypes.data, reltol, nthreads)
        if self._h < 0:
            raise ValueError("plan creation failed (D must be 1, 2 or 3)")

    def trafo(self, f, out=None):
        """fhat (shape (J,)) from the mode array f (shape N)."""
        f = np.ascontiguousarray(f, dtype=np.complex128)
        if f.shape != self.N:
            raise ValueError(f"f must have shape {self.N}, got {f.shape}")
        out = np.empty(self.J, dtype=np.complex128) if out is None else out
        if out.shape != (self.J,) or out.dtype != np.complex128 or not out.flags.c_contiguous:
            raise ValueError("out must be a contiguous complex128 array of shape (J,)")
        self._check(_lib.nfftjl_trafo(self._h, f.ctypes.data, out.ctypes.data))
        return out

    def adjoint(self, fhat, out=None):
        """The mode array f (shape N) from fhat (shape (J,))."""
        fhat = np.ascontiguousarray(fhat, dtype=np.complex128)
        if fhat.shape != (self.J,):
            raise ValueError(f"fhat must have shape ({self.J},), got {fhat.shape}")
        out = np.empty(self.N, dtype=np.complex128) if out is None else out
        if out.shape != self.N or out.dtype != np.complex128 or not out.flags.c_contiguous:
            raise ValueError(f"out must be a contiguous complex128 array of shape {self.N}")
        self._check(_lib.nfftjl_adjoint(self._h, fhat.ctypes.data, out.ctypes.data))
        return out

    def _check(self, rc):
        if rc != 0:
            raise RuntimeError("NFFT failed (plan destroyed?)")

    def close(self):
        """Release the plan (also done when the plan is garbage collected)."""
        if getattr(self, "_h", -1) >= 0:
            _lib.nfftjl_plan_destroy(self._h)
            self._h = -1

    def __del__(self):
        self.close()
