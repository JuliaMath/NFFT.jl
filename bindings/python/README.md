# nfftjl: NFFT.jl for Python

The NFFT, RNFFT, NFCT, NFST and NNFFT of NFFT.jl as a Python package (experimental): a ctypes
wrapper around `libnfftjl` (built from NFFT.jl with `juliac --trim`, see
`../../libnfftjl/README.md`), with the library and its Julia runtime inside the wheel.
No Julia installation is needed to use it.

```python
import numpy as np, nfftjl
k = np.random.rand(10_000, 2) - 0.5          # nodes in [-1/2, 1/2)^2, one per row
p = nfftjl.Plan(k, (256, 256), reltol=1e-8)  # nthreads: default NFFTJL_NUM_THREADS or P-cores
fhat = p.trafo(np.random.randn(256, 256) + 0j)
f = p.adjoint(fhat)
```

fhat_j = Σ_l f_l exp(−2πi l·k_j), l_d = −N_d//2 … N_d − N_d//2 − 1 along axis d of f (the
convention of NFFT.jl and of FINUFFT's type 2 with x = 2πk, isign = −1). Arrays in numpy's
C order are passed without a copy.

Build and test (macOS arm64 so far):

```
../../libnfftjl/build.sh                     # the library
python3 -m venv .venv && .venv/bin/pip install numpy build wheel setuptools finufft
PYTHON=.venv/bin/python ./build_wheel.sh     # → dist/nfftjl-…-macosx_…_arm64.whl (≈ 20 MB)
pip install dist/*.whl
python tests/test_nfftjl.py                  # direct sum, FINUFFT, Python threads
python tests/bench.py 8                      # against FINUFFT
```

Threads: one plan must not run on two threads at once; separate plans run concurrently, each with
its own `nthreads`.
