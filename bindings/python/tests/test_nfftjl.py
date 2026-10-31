# Tests of nfftjl against the direct sum and FINUFFT, and with several Python threads at once.
# Run with `python tests/test_nfftjl.py` or pytest.
import threading

import numpy as np

import nfftjl


def ndft(k, f):
    """direct transform, k of shape (J, D), f of shape N"""
    N = f.shape
    grids = np.meshgrid(*[np.arange(-(n // 2), n - n // 2) for n in N], indexing="ij")
    l = np.stack([g.ravel() for g in grids], axis=1)              # (M, D)
    return np.exp(-2j * np.pi * k @ l.T) @ f.ravel()


def ndft_adjoint(k, fhat, N):
    grids = np.meshgrid(*[np.arange(-(n // 2), n - n // 2) for n in N], indexing="ij")
    l = np.stack([g.ravel() for g in grids], axis=1)
    return (np.exp(2j * np.pi * l @ k.T) @ fhat).reshape(N)


def relerr(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)


def test_accuracy():
    rng = np.random.default_rng(1)
    for N in [(64,), (16, 12), (8, 6, 10)]:
        for tol in (1e-4, 1e-9):
            J = 300
            k = rng.random((J, len(N))) - 0.5
            f = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            fhat = rng.standard_normal(J) + 1j * rng.standard_normal(J)
            p = nfftjl.Plan(k, N, reltol=tol)
            assert relerr(p.trafo(f), ndft(k, f)) < 10 * tol
            assert relerr(p.adjoint(fhat), ndft_adjoint(k, fhat, N)) < 10 * tol


def test_axes():
    # batch axes before, after and around the transformed ones; every slice against the direct sum
    rng = np.random.default_rng(6)
    for shape, axes in (((5, 300), 1), ((40, 7), 0), ((3, 20, 22), (1, 2)), ((20, 22, 4), (0, 1)),
                        ((2, 12, 3), 1), ((6, 8, 9), (0, 1, 2)), ((9, 64, 8), -2)):
        k = rng.random((150, 1 if np.isscalar(axes) else len(axes))) - 0.5
        p = nfftjl.Plan(k, shape, reltol=1e-9, axes=axes)
        a0, a1 = p.axes[0], p.axes[-1]
        f = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        fhat = p.trafo(f)
        assert fhat.shape == shape[:a0] + (150,) + shape[a1 + 1:]
        g = p.adjoint(fhat)
        assert g.shape == shape
        fb = np.moveaxis(f.reshape(shape[:a0] + (-1,) + shape[a1 + 1:]), a0, -1)
        hb = np.moveaxis(fhat, a0, -1)
        gb = np.moveaxis(g.reshape(shape[:a0] + (-1,) + shape[a1 + 1:]), a0, -1)
        for idx in np.ndindex(hb.shape[:-1]):
            assert relerr(hb[idx], ndft(k, fb[idx].reshape(p.N))) < 1e-8
            assert relerr(gb[idx].reshape(p.N), ndft_adjoint(k, hb[idx], p.N)) < 1e-8


def real_direct(kind, k, x, N, adjoint=False):
    """direct sums of the RNFFT (h: last axis N[-1]//2+1), NFCT and NFST, one slice"""
    D = len(N)
    if kind == "rnfft":
        ranges = [np.arange(-(n // 2), n - n // 2) for n in N[:-1]] + [np.arange(N[-1] // 2 + 1)]
    else:
        ranges = [np.arange(n) + (kind == "nfst") for n in N]
    n = np.stack([g.ravel() for g in np.meshgrid(*ranges, indexing="ij")], axis=1)   # (M, D)
    if kind == "rnfft":
        E = np.exp(-2j * np.pi * k @ n.T)                                             # (J, M)
        return (E.conj().T @ x).reshape([len(r) for r in ranges]) if adjoint else (E @ x.ravel()).real
    f = np.cos if kind == "nfct" else np.sin
    E = np.prod(f(2 * np.pi * k[:, None, :] * n[None, :, :]), axis=2)                # (J, M)
    return (E.T @ x).reshape(N) if adjoint else E @ x.ravel()


def test_real_transforms():
    rng = np.random.default_rng(7)
    for cls, kind in ((nfftjl.RNFFTPlan, "rnfft"), (nfftjl.NFCTPlan, "nfct"), (nfftjl.NFSTPlan, "nfst")):
        for shape, axes, dtype, tol in (((40,), None, np.float64, 1e-9), ((12, 10), None, np.float64, 1e-9),
                                        ((6, 5, 4), None, np.float64, 1e-9), ((3, 12, 10), (1, 2), np.float64, 1e-9),
                                        ((12, 10, 2), (0, 1), np.float32, 1e-5)):
            D = len(shape) if axes is None else len(axes)
            J = 120
            k = rng.random((J, D)) - 0.5 if kind == "rnfft" else rng.random((J, D)) / 2
            p = cls(k, shape, reltol=tol, axes=axes, dtype=dtype)
            a0, a1 = p.axes[0], p.axes[-1]
            if kind == "rnfft":
                f = rng.standard_normal(p.in_shape) + 1j * rng.standard_normal(p.in_shape)
            else:
                f = rng.standard_normal(p.in_shape)
            fhat = p.trafo(f)
            assert fhat.shape == p.out_shape and fhat.dtype == p.out_dtype
            back = p.adjoint(fhat)
            assert back.shape == p.in_shape and back.dtype == p.in_dtype
            fb = np.moveaxis(f.reshape(p.in_shape[:a0] + (-1,) + p.in_shape[a1 + 1:]), a0, -1)
            hb = np.moveaxis(fhat, a0, -1)
            bb = np.moveaxis(back.reshape(p.in_shape[:a0] + (-1,) + p.in_shape[a1 + 1:]), a0, -1)
            slice_shape = p.in_shape[a0:a1 + 1]
            for idx in np.ndindex(hb.shape[:-1]):
                x = fb[idx].reshape(slice_shape).astype(np.complex128 if kind == "rnfft" else np.float64)
                assert relerr(hb[idx], real_direct(kind, k, x, p.N)) < 10 * tol
                y = hb[idx].astype(np.float64)
                assert relerr(bb[idx].reshape(slice_shape), real_direct(kind, k, y, p.N, adjoint=True)) < 10 * tol
    # a plan rejects data of the wrong shape, nodes outside the range of the transform fail
    p = nfftjl.RNFFTPlan(rng.random((10, 1)) - 0.5, 16)
    assert p.in_shape == (9,)
    try:
        nfftjl.NFCTPlan(rng.random((10, 1)) - 0.5, 16)
        assert False
    except ValueError as e:
        assert "[0, 1/2]" in str(e)


def test_nnfft():
    # direct sums with batch axes, single precision, and FINUFFT's type 3
    rng = np.random.default_rng(8)
    for D, shape, axis, dtype, tol in ((1, (150,), -1, np.complex128, 1e-9), (2, (3, 150, 2), 1, np.complex128, 1e-9),
                                       (3, (120, 4), 0, np.complex128, 1e-9), (2, (150,), 0, np.complex64, 1e-4)):
        k = 6 * (rng.random((200, D)) - 0.5) - 1                 # targets
        y = 20 * (rng.random((150 if D < 3 else 120, D)) - 0.5) + 3   # sources
        p = nfftjl.NNFFTPlan(k, y, shape, reltol=tol, axis=axis, dtype=dtype)
        E = np.exp(-2j * np.pi * k @ y.T)                          # (J, N)
        f = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
        fhat = p.trafo(f)
        assert fhat.shape == p.out_shape and fhat.dtype == dtype
        ref = np.moveaxis(np.tensordot(E, f.astype(np.complex128), axes=([1], [p.axis])), 0, p.axis)
        assert relerr(fhat, ref) < 10 * tol
        g = p.adjoint(ref.astype(dtype))
        refa = np.moveaxis(np.tensordot(E.conj().T, ref, axes=([1], [p.axis])), 0, p.axis)
        assert g.shape == shape and relerr(g, refa) < 10 * tol
    import finufft
    y = 30 * rng.random((2000, 2)); k = 4 * rng.random((3000, 2)) - 2
    c = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
    ref = finufft.nufft2d3(*(np.ascontiguousarray(a) for a in (2 * np.pi * y[:, 0], 2 * np.pi * y[:, 1], c, k[:, 0], k[:, 1])),
                           isign=-1, eps=1e-12)
    assert relerr(nfftjl.NNFFTPlan(k, y, reltol=1e-10).trafo(c), ref) < 1e-9


def test_float32():
    rng = np.random.default_rng(5)
    for N, tol in (((64,), 1e-5), ((16, 12), 1e-5), ((8, 6, 10), 1e-3)):
        J = 300
        k = rng.random((J, len(N))) - 0.5
        f = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        p = nfftjl.Plan(k, N, reltol=tol, dtype=np.complex64)
        fhat = p.trafo(f)
        assert fhat.dtype == np.complex64
        assert relerr(fhat, ndft(k, f.astype(np.complex128))) < 10 * tol
        g = p.adjoint(fhat)
        assert g.dtype == np.complex64
        assert relerr(g, ndft_adjoint(k, fhat.astype(np.complex128), N)) < 10 * tol


def test_finufft():
    import finufft
    rng = np.random.default_rng(2)
    N = (64, 48)
    J = 5000
    k = rng.random((J, 2)) - 0.5
    f = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x, y = 2 * np.pi * k[:, 0], 2 * np.pi * k[:, 1]
    ref = finufft.nufft2d2(x, y, f, isign=-1, eps=1e-12)
    assert relerr(nfftjl.Plan(k, N, reltol=1e-10).trafo(f), ref) < 1e-9


def test_threads():
    # four Python threads, each with its own threaded plan, plus numpy (BLAS threads) at the
    # same time
    rng = np.random.default_rng(3)
    N = (128, 128)
    J = 20000
    errors = []

    def work(seed):
        r = np.random.default_rng(seed)
        k = r.random((J, 2)) - 0.5
        f = r.standard_normal(N) + 0j
        p = nfftjl.Plan(k, N, reltol=1e-8, nthreads=2)
        ref = p.trafo(f)
        for _ in range(20):
            if relerr(p.trafo(f), ref) > 1e-14:
                errors.append(seed)
            p.adjoint(ref)

    def blas():
        a = rng.standard_normal((800, 800))
        for _ in range(10):
            a = a @ a.T / 800

    ts = [threading.Thread(target=work, args=(s,)) for s in range(4)] + [threading.Thread(target=blas)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


def test_threaded_plan():
    # one caller, all threads of the library; same result as single-threaded
    rng = np.random.default_rng(4)
    N = (256, 256)
    k = rng.random((100_000, 2)) - 0.5
    f = rng.standard_normal(N) + 0j
    a = nfftjl.Plan(k, N, reltol=1e-8).trafo(f)
    b = nfftjl.Plan(k, N, reltol=1e-8, nthreads=1).trafo(f)
    assert relerr(a, b) < 1e-13


if __name__ == "__main__":
    test_accuracy()
    test_axes()
    test_real_transforms()
    test_nnfft()
    test_float32()
    test_finufft()
    test_threads()
    test_threaded_plan()
    print("passed")
