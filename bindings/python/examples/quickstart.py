import numpy as np
import nfftjl

k = np.random.rand(10_000, 2) - 0.5               # 10 000 nodes in [-1/2, 1/2)², one per row
f = np.random.randn(256, 256) + 0j                # equidistant data

p = nfftjl.Plan(k, f.shape)                       # relative error 1e-8 by default
fhat = p.trafo(f)                                 # NFFT: equidistant → nodes
g = p.adjoint(fhat)                               # adjoint NFFT: nodes → equidistant
