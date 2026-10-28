# A 2D NFFT and its adjoint with nfftjl, checked against the direct sum.
import numpy as np
import nfftjl

rng = np.random.default_rng(0)
N = (64, 48)                                      # size of the equidistant array
k = rng.random((1000, 2)) - 0.5                   # 1000 nodes in [-1/2, 1/2)², one per row
f = rng.standard_normal(N) + 1j * rng.standard_normal(N)

p = nfftjl.Plan(k, N, reltol=1e-9)                # nthreads: all threads of the library
fhat = p.trafo(f)                                 # fhat_j = Σ_l f_l exp(-2πi l·k_j)
g = p.adjoint(fhat)                               # g_l = Σ_j fhat_j exp(+2πi l·k_j)

# the direct sum at the first ten nodes; l_d = -N_d//2, …, N_d - N_d//2 - 1 along axis d of f
grid = np.meshgrid(*[np.arange(-(n // 2), n - n // 2) for n in N], indexing="ij")
l = np.stack(grid, axis=-1)                       # shape N + (2,)
ref = np.array([np.sum(f * np.exp(-2j * np.pi * (l @ kj))) for kj in k[:10]])
err = np.linalg.norm(fhat[:10] - ref) / np.linalg.norm(ref)
print(f"relative error {err:.1e}")
assert err < 1e-8
