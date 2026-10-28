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
    test_finufft()
    test_threads()
    test_threaded_plan()
    print("passed")
