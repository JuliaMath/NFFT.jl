# Python

!!! note "Experimental"
    The Python package `nfftjl` wraps `libnfftjl`, NFFT.jl compiled into a library with
    `juliac --trim` (see [C](c.md)). The wheel carries the library and its Julia runtime; no Julia
    installation is needed to use it. It covers the NFFT, RNFFT, NFCT, NFST and NNFFT and their
    adjoints in 1–3 dimensions, with batches, in single and double precision; the table on the
    [home page](index.md) lists what the Julia package offers beyond that. The interface may still change.

## Installation

`nfftjl` is not on PyPI yet. A wheel for your platform is built from the repository (building
needs Julia ≥ 1.12, using the wheel does not):

```
git clone https://github.com/JuliaMath/NFFT.jl && cd NFFT.jl
libnfftjl/build.sh                                  # the library, ≈ 2 min
python3 -m venv .venv && .venv/bin/pip install numpy build wheel setuptools
PYTHON=.venv/bin/python bindings/python/build_wheel.sh
.venv/bin/pip install bindings/python/dist/*.whl
```

## A first transform

The NFFT evaluates ``\hat f_j = \sum_{\bm l} f_{\bm l} \, e^{-2\pi i \bm l \cdot \bm k_j}`` at nodes
``\bm k_j \in [-1/2, 1/2)^D`` for the equidistant array ``f`` (see
[Mathematical Background](@ref)); the adjoint is the sum with the opposite sign. This example
computes both and checks the first values against the direct sum:

```python
{{include bindings/python/examples/getting_started.py}}
```

## Interface

`nfftjl.Plan(k, shape, reltol=1e-8, nthreads=None, dtype=numpy.complex128, axes=None)`
: Plan for the nodes `k`, an array of shape `(J, D)` (one node per row; shape `(J,)` in 1D) with
  entries in ``[-1/2, 1/2)``, and data of shape `shape` (an integer or a tuple), transformed along
  `axes`: all axes by default, otherwise `D` = 1, 2 or 3 consecutive axes (column `d` of `k`
  belongs to `axes[d]`). The other axes are batch axes, see [Batches](#Batches). `reltol` is the
  requested relative error: window, window width and oversampling factor are chosen for it at the
  lowest predicted cost (see [Accuracy and Parameters](@ref)).
  `nthreads` defaults to `num_threads()`. `dtype` is the element type of the data,
  `numpy.complex128` or `numpy.complex64`; the nodes are given in double precision either way, and
  grid positions are computed from them in double precision.

`p.trafo(f, out=None)`
: The NFFT of `f` (shape `p.shape`, converted to `dtype`): an array of shape `p.out_shape`, the
  transformed axes replaced by one axis of length `J`.

`p.adjoint(fhat, out=None)`
: The adjoint NFFT of `fhat` (shape `p.out_shape`): an array of shape `p.shape`.

`p.close()`
: Releases the plan; garbage collection does the same.

`nfftjl.num_threads()`
: The default number of threads of a plan: the environment variable `NFFTJL_NUM_THREADS` if set,
  otherwise the number of performance cores (Apple Silicon) or of all CPU threads. The library
  has one thread per CPU thread, the upper limit of `nthreads`.

Along axis `d` of `f`, index `i` stands for the frequency ``l_d = i - N_d/2`` (``N_d/2`` rounded
down), in increasing order, as `numpy.fft.fftshift` orders them. Arrays in numpy's default (C)
order are passed to the library without a copy. With `out` the result goes into an existing
array.

The transform equals FINUFFT's type 2 with `x = 2π k`, `isign = -1` and the default mode ordering;
the adjoint equals type 1 with `isign = +1`.

## Real data: RNFFT, NFCT, NFST

`nfftjl.RNFFTPlan`, `nfftjl.NFCTPlan` and `nfftjl.NFSTPlan` take the same arguments as `Plan`
(`dtype` is the real type, `numpy.float64` or `numpy.float32`); `p.in_shape` and `p.out_shape`
give the shapes of their data:

- **RNFFT** (real values at the nodes): ``\hat f_j = \mathrm{Re} \sum_{\bm n} h_{\bm n}
  e^{-2\pi i \bm n \cdot \bm k_j}`` with ``n = 0, \dots, N/2`` along the last transformed axis
  (the layout of `numpy.fft.rfftn`) and as for the NFFT along the others; `h` is complex, the
  values at the nodes are real. The adjoint is ``h_{\bm n} = \sum_j \hat f_j
  e^{2\pi i \bm n \cdot \bm k_j}``. Nodes in ``[-1/2, 1/2)``.
- **NFCT** and **NFST**: ``\hat f_j = \sum_{\bm n} f_{\bm n} \prod_d \cos(2\pi n_d k_{dj})``
  over ``n_d = 0, \dots, N_d - 1``, or with ``\sin`` over ``n_d = 1, \dots, N_d``, for real `f`;
  `adjoint` is the transpose. Nodes in ``[0, 1/2]``.

```python
{{include bindings/python/examples/real.py}}
```

A plan whose nodes lie outside the range of its transform raises a `ValueError` with the reason.

## Non-equidistant in both domains: NNFFT

`nfftjl.NNFFTPlan(k, y, shape=None, reltol=1e-8, nthreads=None, dtype=numpy.complex128, axis=-1)`
computes ``\hat f_j = \sum_l f_l e^{-2\pi i \bm k_j \cdot \bm y_l}`` for targets `k` (shape
`(J, D)`) and sources `y` (shape `(N, D)`) with arbitrary real coordinates, and the adjoint. The
cost grows with the product of the extents of `k` and `y`, not with `J` and `N`. With `shape`, the
data have batch axes around `axis`, the axis of the sources, which the transform replaces by the
`J` targets. It equals FINUFFT's type 3 with `x = 2π y`, `s = k`, `isign = -1`.

```python
{{include bindings/python/examples/nnfft.py}}
```

## Batches

Every slice along the batch axes is transformed with the same nodes, for example the coil images
in MRI. One plan for all slices is faster than a loop over slices: the nodes are binned once, and
the batch kernels of NFFT.jl process several slices per pass over the nodes.

```python
{{include bindings/python/examples/batch.py}}
```

## Threads

- One plan must not be executed by two threads at the same time: create one plan per thread.
  Separate plans run concurrently, each with its own `nthreads`.
- The default number of threads of a plan is the environment variable `NFFTJL_NUM_THREADS` if
  set, otherwise the number of performance cores on Apple Silicon and of all CPU threads
  elsewhere. A 1D transform runs its FFT in one piece per thread, so a thread on an efficiency
  core or on a core busy with other work holds it up (up to 2× slower with all 10 threads of an
  M1 Max); on a busy machine fewer threads than cores are faster.
- numpy's OpenBLAS threads keep spinning for a while after a numpy call and slow down a transform
  that follows; `OPENBLAS_NUM_THREADS=1` avoids this when timing.

## Comparison with FINUFFT

`bindings/python/tests/bench.py` compares `nfftjl` with FINUFFT's Python package on the problems of
the [Performance](@ref) page. The library runs the same machine code as the Julia package.
