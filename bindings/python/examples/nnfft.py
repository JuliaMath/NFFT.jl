# NNFFT: sums of exponentials at arbitrary real frequencies and positions, here a 2D signal of
# point sources sampled at scattered frequencies, checked against the direct sum.
import numpy as np
import nfftjl

rng = np.random.default_rng(3)
y = 50 * rng.random((2000, 2))                     # source positions
k = 3 * rng.random((5000, 2)) - 1.5                # frequencies
f = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)

p = nfftjl.NNFFTPlan(k, y, reltol=1e-9)
fhat = p.trafo(f)                                  # sum_l f_l exp(-2πi k_j·y_l)
ref = np.exp(-2j * np.pi * k[:100] @ y.T) @ f
err = np.linalg.norm(fhat[:100] - ref) / np.linalg.norm(ref)
print(f"NNFFT: {len(f)} sources, {len(fhat)} targets, relative error {err:.1e}")
assert err < 1e-8
