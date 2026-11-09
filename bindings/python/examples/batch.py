# Many transforms with the same nodes: eight coil images of 128 × 128, transformed along axes 1 and 2
# with one plan, compared with one plan per coil.
import numpy as np
import nfftjl

rng = np.random.default_rng(1)
k = rng.random((20_000, 2)) - 0.5                  # nodes, one per row
images = rng.standard_normal((8, 128, 128)) + 0j   # coil × x × y

p = nfftjl.Plan(k, images.shape, axes=(1, 2))      # axis 0 is a batch axis
data = p.trafo(images)                             # shape (8, 20000)
back = p.adjoint(data)                             # shape (8, 128, 128)

single = nfftjl.Plan(k, (128, 128))
ref = np.stack([single.trafo(images[c]) for c in range(8)])
err = np.linalg.norm(data - ref) / np.linalg.norm(ref)
print(f"shape {data.shape}, difference to single plans {err:.1e}")
assert data.shape == (8, 20_000) and back.shape == images.shape and err < 1e-12
