# nfftjl against FINUFFT from Python: plan time, transform and adjoint (fastest of 5), relative
# error against FINUFFT at eps = 1e-14. J = 2^18 nodes, all threads.
import sys
import time

import finufft
import numpy as np

import nfftjl

nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else nfftjl.num_threads()
rng = np.random.default_rng(0)
J = 2**18


def tmin(f, n=5):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def relerr(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)


print(f"{'problem':<18}{'tol':>7} | {'library':<8}{'plan ms':>9}{'trafo ms':>10}{'adj ms':>9}{'err':>10}")
for N in [(262144,), (512, 512), (64, 64, 64)]:
    D = len(N)
    k = rng.random((J, D)) - 0.5
    f = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    fhat = rng.standard_normal(J) + 1j * rng.standard_normal(J)
    xs = [2 * np.pi * np.ascontiguousarray(k[:, d]) for d in range(D)]
    ref = finufft.Plan(2, N, eps=1e-14, isign=-1, nthreads=nthreads)
    ref.setpts(*xs)
    ref_t = ref.execute(f)
    for tol in (1e-4, 1e-8):
        t0 = time.perf_counter()
        p = nfftjl.Plan(k, N, reltol=tol, nthreads=nthreads)
        tp = time.perf_counter() - t0
        tt = tmin(lambda: p.trafo(f))
        ta = tmin(lambda: p.adjoint(fhat))
        e = relerr(p.trafo(f), ref_t)
        print(f"{str(N):<18}{tol:>7.0e} | {'nfftjl':<8}{1e3*tp:>9.1f}{1e3*tt:>10.2f}{1e3*ta:>9.2f}{e:>10.1e}")

        t0 = time.perf_counter()
        p2 = finufft.Plan(2, N, eps=tol, isign=-1, nthreads=nthreads)
        p2.setpts(*xs)
        p1 = finufft.Plan(1, N, eps=tol, isign=1, nthreads=nthreads)
        p1.setpts(*xs)
        tp = time.perf_counter() - t0
        tt = tmin(lambda: p2.execute(f))
        ta = tmin(lambda: p1.execute(fhat))
        e = relerr(p2.execute(f), ref_t)
        print(f"{'':<18}{'':>7} | {'finufft':<8}{1e3*tp:>9.1f}{1e3*tt:>10.2f}{1e3*ta:>9.2f}{e:>10.1e}")
