# The transforms for real data: an NFCT in 2D checked against its direct sum, and an RNFFT whose
# coefficients have the layout of numpy.fft.rfftn.
import numpy as np
import nfftjl

rng = np.random.default_rng(2)
N = (32, 24)
k = rng.random((500, 2)) / 2                       # NFCT/NFST nodes in [0, 1/2]
f = rng.standard_normal(N)

p = nfftjl.NFCTPlan(k, N, reltol=1e-9)
fhat = p.trafo(f)                                  # sum_n f_n cos(2π n_1 k_1) cos(2π n_2 k_2)
n1, n2 = np.meshgrid(np.arange(N[0]), np.arange(N[1]), indexing="ij")
ref = np.array([np.sum(f * np.cos(2 * np.pi * n1 * x) * np.cos(2 * np.pi * n2 * y)) for x, y in k])
err = np.linalg.norm(fhat - ref) / np.linalg.norm(ref)
print(f"NFCT: relative error {err:.1e}")
assert err < 1e-8

q = nfftjl.RNFFTPlan(rng.random((500, 2)) - 0.5, N)
print("RNFFT: coefficients", q.in_shape, "complex, values at the nodes", q.out_shape, "real")
assert q.in_shape == (32, 13) and q.trafo(np.ones(q.in_shape, dtype=complex)).dtype == np.float64
