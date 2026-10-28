# NFFT.jl

*Fast Fourier transforms at non-equidistant nodes: a Julia package for CPUs and GPUs, and a library
for Python and C compiled from the same code.*

!!! warning "Development version"
    This documentation describes NFFT.jl 1.0, which is under development on the main branch
    and not released yet. `Pkg.add("NFFT")` installs version 0.14, documented
    [here](https://JuliaMath.github.io/NFFT.jl/stable). See [Migrating from NFFT.jl 0.x](@ref).

NFFT.jl computes the non-equidistant fast Fourier transform (NFFT, also called NUFFT, or gridding in
MRI): a discrete Fourier transform where the nodes in one domain do not lie on a grid. It is
accurate to a tolerance you choose and runs in ``\mathcal{O}(|\bm{N}| \log |\bm{N}| + J)`` operations
for ``J`` nodes and an array of size ``\bm{N}``. The same plans run multi-threaded on the CPU and
on GPUs (CUDA, AMD, Apple).

## Quick start

**Julia**

```julia
using NFFT

k = rand(2, 10_000) .- 0.5           # 10 000 nodes in [-1/2, 1/2)²
f = randn(ComplexF64, 256, 256)      # equidistant data

p = plan_nfft(k, size(f))            # relative error 1e-8 by default
fHat = p * f                         # NFFT: equidistant → nodes
g = adjoint(p) * fHat                # adjoint NFFT: nodes → equidistant
```

**Python** (experimental, see [Python](python.md))

```python
{{include bindings/python/examples/quickstart.py}}
```

**C** (experimental, see [C](c.md))

```c
#include "nfftjl.h"

int64_t N[] = {256, 256};             /* column major: node j is k[2j], k[2j+1] */
int64_t p = nfftjl_plan_create(2, N, J, k, 1e-8, nfftjl_num_threads());
nfftjl_trafo(p, f, fhat);             /* NFFT: equidistant → nodes */
nfftjl_adjoint(p, fhat, g);           /* adjoint NFFT: nodes → equidistant */
nfftjl_plan_destroy(p);
```

## What is available in which language

The Julia package is the full library. Python and C use `libnfftjl`, a library compiled from the
same Julia code with `juliac --trim`: no Julia installation is needed to use it, and it is as fast
as the Julia package. Its interface is experimental and covers the core transform so far.

| | Julia | Python | C |
|:--|:--|:--|:--|
| NFFT and adjoint, `reltol` chooses the parameters | ✓ | ✓ | ✓ |
| dimensions | any | 1–3 | 1–3 |
| precision | Float32, Float64 | complex64, complex128 | float, double |
| multi-threaded | ✓ | ✓ | ✓ |
| transforms along dimensions (batches) | ✓ | ✓ | ✓ |
| RNFFT, NFCT, NFST, NNFFT | ✓ | – | – |
| GPUs | ✓ | – | – |
| stages, density compensation, Toeplitz | ✓ | – | – |
| status | 1.0 (in development) | experimental | experimental |

## What NFFT.jl offers

- **You ask for an accuracy, not for parameters.** `reltol` is all you need to set. The plan picks
  the window function, its width and the oversampling factor that reach this error at the lowest
  predicted cost, and `accuracy(p)` tells you what it expects to achieve.
- **CPU and GPU from one code base.** Every transform, including batch dimensions, runs on the GPU
  when the plan gets a GPU array type: `plan_nfft(k, N; storage=CuArray)`. The GPU kernels are
  written with KernelAbstractions.jl and serve CUDA, AMD and Apple GPUs alike; see [GPU](@ref).
- **Fast.** Tiled resampling with piecewise-polynomial window evaluation, an FFT that skips the
  zero padding, and multi-threading in every stage. See [Performance](@ref) for a comparison with
  FINUFFT and ducc.
- **Transforms along some dimensions of an array** (`dims`), with batch kernels that process many
  transforms with the same nodes at once, as needed for multi-coil MRI or OCT.
- **Real data at the nodes** (`plan_rnfft`, e.g. OCT), on a real grid.
- **Cosine and sine transforms** (NFCT, NFST) for real data, with the definitions of NFFT3 and
  about the cost of a complex NFFT.
- **Non-equidistant points in both domains** (NNFFT, `plan_nnfft`).
- **Single and double precision**, with node positions always computed in double precision.
- **The three stages of the algorithm as an interface**: resampling correction, FFT and
  resampling can be called one by one, for example to embed the NFFT into a Toeplitz approach or to
  profile it.
- **One interface for several implementations.** `AbstractNFFTs.jl` defines plans, options and
  operations; NFFT.jl and NonuniformFFTs.jl implement it. Code written against the interface runs
  with each of them.

## Installation

```julia
using Pkg
Pkg.add("NFFT")
```

This also installs `AbstractNFFTs.jl`, the interface package. Sampling density compensation and
Toeplitz kernels are in `NFFTTools.jl` (`Pkg.add("NFFTTools")`). For Python and C see
[Python](python.md) and [C](c.md).

## Guide

The documentation follows the [Diátaxis](https://diataxis.fr) structure:

**Getting started**, to learn NFFT.jl step by step:
- [Getting Started](@ref) in Julia: plans, transforms, precision, threads and changing nodes.
- [Python](python.md) and [C](c.md): installation, a first transform, the interface, threads.

**How-to guides** (Julia), for a task at hand:
- [Transforms along Dimensions](@ref): `dims` and many transforms with the same nodes.
- [Real Data](@ref): real values at the nodes with `plan_rnfft` (e.g. OCT).
- [Cosine and Sine Transforms](@ref): the NFCT and NFST.
- [Non-equidistant in Both Domains](@ref): the NNFFT.
- [GPU](@ref): all transforms on GPUs.
- [Density Compensation and Toeplitz Kernels](@ref): `NFFTTools.jl`.
- [Choosing Parameters](@ref): window, width, oversampling and FFT planning by hand.
- [Stages of the NFFT](@ref): calling the stages one by one.
- [Migrating from NFFT.jl 0.x](@ref): what changed in version 1.0.

**Explanation**, for understanding:
- [Mathematical Background](@ref): the transforms and the three stages of the algorithm.
- [Accuracy and Parameters](@ref): how `reltol` becomes window, width and oversampling.
- [How NFFT.jl Works](@ref): tiles, halo, pruned FFT, batches, real data, NNFFT and GPUs, with
  the ideas behind them.
- [Performance](@ref): NFFT.jl against FINUFFT, ducc and NonuniformFFTs.jl.

**Reference**, to look things up:
- [API](@ref): all exported functions and types.
- [Abstract Interface](@ref): `AbstractNFFTs.jl` and how to write an implementation.

## Citing

The source code is licensed under the MIT license. If you use NFFT.jl in your work, please cite

> T. Knopp, M. Boberg and M. Grosser, NFFT.jl: Generic and Fast Julia Implementation of the
> Nonequidistant Fast Fourier Transform, SIAM Journal on Scientific Computing 45(3), C179–C205, 2023
> ([arXiv:2208.00049](https://arxiv.org/abs/2208.00049)).

A BibTeX entry is in `NFFT.bib` in the root of the repository.

## Contact

Bugs and ideas go to the [issue tracker](https://github.com/JuliaMath/NFFT.jl/issues), questions to
the [discussions](https://github.com/JuliaMath/NFFT.jl/discussions).

## Contributors

- [Tobias Knopp](https://www.imt.uni-luebeck.de)
- [Niklas Hackelberg](https://github.com/nHackel)
- [Robert Dahl Jacobsen](https://github.com/robertdj)
- [Mirco Grosser](https://github.com/migrosser)
- [Jakob Assländer](https://med.nyu.edu/faculty/jakob-asslaender)
- [Mosè Giordano](https://github.com/giordano)
- [Jeff Fessler](https://github.com/JeffFessler)

The complete list is on [GitHub](https://github.com/JuliaMath/NFFT.jl/graphs/contributors).
