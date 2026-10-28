# Python

!!! note "Experimental"
    The Python package `nfftjl` wraps `libnfftjl`, NFFT.jl compiled into a library with
    `juliac --trim` (see [C](c.md)). The wheel carries the library and its Julia runtime; no Julia
    installation is needed to use it. It covers the NFFT and its adjoint in 1–3 dimensions in
    double precision; the table on the [home page](index.md) lists what the Julia package offers
    beyond that. The interface may still change.

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

`nfftjl.Plan(k, N, reltol=1e-8, nthreads=None)`
: Plan for the nodes `k`, an array of shape `(J, D)` (one node per row; shape `(J,)` in 1D) with
  entries in ``[-1/2, 1/2)``, and the array size `N` (an integer or a tuple of `D` = 1, 2, 3
  integers). `reltol` is the requested relative error: window, window width and oversampling
  factor are chosen for it at the lowest predicted cost (see [Accuracy and Parameters](@ref)).
  `nthreads` defaults to `num_threads()`.

`p.trafo(f, out=None)`
: The NFFT of `f` (shape `N`, complex): an array of shape `(J,)`.

`p.adjoint(fhat, out=None)`
: The adjoint NFFT of `fhat` (shape `(J,)`): an array of shape `N`.

`p.close()`
: Releases the plan; garbage collection does the same.

`nfftjl.num_threads()`
: The default number of threads of a plan: the environment variable `NFFTJL_NUM_THREADS` if set,
  otherwise all CPU threads, which is also the upper limit of `nthreads`.

Along axis `d` of `f`, index `i` stands for the frequency ``l_d = i - N_d/2`` (``N_d/2`` rounded
down), in increasing order, as `numpy.fft.fftshift` orders them. Arrays in numpy's default (C)
order are passed to the library without a copy. With `out` the result goes into an existing
array.

The transform equals FINUFFT's type 2 with `x = 2π k`, `isign = -1` and the default mode ordering;
the adjoint equals type 1 with `isign = +1`.

## Threads

- One plan must not be executed by two threads at the same time: create one plan per thread.
  Separate plans run concurrently, each with its own `nthreads`.
- numpy's OpenBLAS threads keep spinning for a while after a numpy call and slow down a transform
  that follows; `OPENBLAS_NUM_THREADS=1` avoids this when timing.

## Comparison with FINUFFT

`bindings/python/tests/bench.py` compares `nfftjl` with FINUFFT's Python package on the problems of
the [Performance](@ref) page. The library runs the same machine code as the Julia package.
