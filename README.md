# NFFT.jl

This package provides a Julia implementation of the Non-equidistant Fast Fourier Transform (NFFT).

[![](https://img.shields.io/badge/docs-stable%20(0.14)-blue.svg)](https://JuliaMath.github.io/NFFT.jl/stable)
[![](https://img.shields.io/badge/docs-dev%20(1.0)-blue.svg)](https://JuliaMath.github.io/NFFT.jl/dev)
[![Build status](https://github.com/JuliaMath/NFFT.jl/workflows/CI/badge.svg)](https://github.com/JuliaMath/NFFT.jl/actions)
[![codecov.io](http://codecov.io/github/JuliaMath/NFFT.jl/coverage.svg?branch=main)](http://codecov.io/github/JuliaMath/NFFT.jl?branch=main)

> [!IMPORTANT]
> **The main branch is the development version of NFFT.jl 1.0**, a new implementation with a
> new interface (AbstractNFFTs 1.0). It is not released yet and its interface may still change.
> Feedback and discussion: [issues](https://github.com/JuliaMath/NFFT.jl/issues).
>
> - `Pkg.add("NFFT")` installs the released version 0.14. Its documentation is
>   [stable](https://JuliaMath.github.io/NFFT.jl/stable), its code on the branch
>   [`release-0.14`](https://github.com/JuliaMath/NFFT.jl/tree/release-0.14).
> - The [dev documentation](https://JuliaMath.github.io/NFFT.jl/dev) describes the new version,
>   including a [migration guide](https://JuliaMath.github.io/NFFT.jl/dev/migration/) and
>   [benchmarks](https://JuliaMath.github.io/NFFT.jl/dev/performance/) against FINUFFT and ducc.
>
> To try the development version (AbstractNFFTs 1.0 is not registered yet, so it comes first):
> ```
> pkg> add https://github.com/JuliaMath/NFFT.jl:AbstractNFFTs#main
> pkg> add NFFT#main
> ```

## What is new in 1.0

- **One accuracy parameter**: `plan_nfft(k, N; reltol=1e-8)` chooses window, window width and
  oversampling factor at the lowest predicted cost; `accuracy(p)` reports what it expects.
- **Faster**: on par with or ahead of FINUFFT and ducc (up to 2× in 1D), and up to several times
  faster than 0.14 in 2D and 3D.
- **Transforms along dimensions** (`dims`) with batch kernels, e.g. for multi-coil MRI or OCT.
- **Real data**: `plan_rnfft` (real values at the nodes), fast NFCT and NFST.
- **The stages of the algorithm** (resampling correction, FFT, resampling) as an interface.
- **A benchmark suite** in `benchmark/` that compares with other libraries and with earlier
  revisions.

```julia
using NFFT

k = rand(2, 10_000) .- 0.5           # nodes in [-1/2, 1/2)²
f = randn(ComplexF64, 256, 256)
p = plan_nfft(k, size(f))            # relative error 1e-8 by default
fHat = p * f
g = adjoint(p) * fHat
```

## How to give credit
If you use this package please acknowledge it by citing:

T. Knopp, M. Boberg and M. Grosser, NFFT.jl: Generic and Fast Julia Implementation of the Nonequidistant
 Fast Fourier Transform, SIAM Journal on Scientific Computing, 45(3), C179-C205, 2023 [url](https://epubs.siam.org/doi/abs/10.1137/22M1510935) [*arXiv:2208.00049*](http://arxiv.org/abs/2208.00049)

A BibTeX file `NFFT.bib` can be found in the root folder of the Github repository.

## Community Standards

This project is part of the Julia community and follows the [Julia community standards](https://julialang.org/community/standards/).
