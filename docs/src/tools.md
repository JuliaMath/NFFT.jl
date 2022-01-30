# Tools

The packages `AbstractNFFT.jl` and `NFFT.jl` are on purpose kept small and focussed. Additional tooling that relates to the NFFT is offloaded into a package `NFFTTools.jl`. The later can also
be used with `NFFT3.jl` or `FINUFFT.jl`.

## Sampling Density

The first tool that `NFFTTools.jl` is the computation of the sampling density. To motivate this let
us have a look at the normalized variant of the DFT matrix ``\bm{F}``. ``\bm{F}`` is unitary, which
means that
```math
 \bm{F}^{\mathsf{H}} \bm{F} = \bm{I} 
```
where ``\bm{I}`` is the identity matrix. In other words, ``\bm{F}^{\mathsf{H}}`` is the inverse of ``\bm{F}^{\mathsf{H}}``.

Now lets switch to the NFFT. Have you already wondered why we don't call the adjoint the inverse? Well its because in general we have
```math
 \bm{A}^{\mathsf{H}} \bm{A} \neq \bm{I} 
```
In fact, for the NFFT the inverse is a much more complicated subject since the linear system ``\bm{A} \bm{f} = \hat{\bm{f}}`` can have one, no or many solutions since ``\bm{A}``can be under- or over-determined.

The good news is that in most cases, where ``M \approx N`` and no complete clustering of the sampling nodes, one can find a diagonal weighting matrix ``\bm{W} = \left( w_k \right)_{k=1}^{M}``such that
```math
 \bm{A}^{\mathsf{H}} \bm{W} \bm{A} \approx \bm{I} 
```
The weights ``w_k`` can be considered to we quadrature weights that account for the *area* covered by the node ``\bm{x}_k``. 

`NFFTTools.jl` provides the function `sdc` that takes an existing plan and calculated suitable density weights:
```julia
weights = sdc(p, iters = 10)
```
The function implements the method proposed in [Pipe & Menon, 1999. Mag Reson Med, 186, 179](https://doi.org/10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V).

## Toeplitz Kernel

The aforementioned matrix ``\bm{A}^{\mathsf{H}} \bm{W} \bm{A}`` arises when solving linear system of the form
```math
\bm{A}^{\mathsf{H}} \bm{f} = \hat{\bm{f}}
```
which can be done via the normal equation
```math
\bm{A}^{\mathsf{H}} \bm{W} \bm{A} \bm{f} = \bm{A}^{\mathsf{H}} \bm{W} \hat{\bm{f}}
```
The normal or Gram matrix ``\bm{A}^{\mathsf{H}} \bm{W} \bm{A}`` has a [Toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix) structure. For multidimensional NFFT is is a block Toeplitz matrix with Toeplitz blocks. A Toeplitz matrix (and its block variants) can be embedded into a [circulant matrix](https://en.wikipedia.org/wiki/Circulant_matrix) of twice the size in each dimension. Circulant matrices however are known to be diagonalizable by ordinary FFTs. This means we can multiply with ``\bm{A}^{\mathsf{H}} \bm{W} \bm{A}`` by just two FFTs of size ``2\bm{N}``, which is basically the same amount of NFFTs as are required for an NFFT-based calculation of matrix-vector products with ``\bm{A}^{\mathsf{H}} \bm{W} \bm{A}``. But the important difference is that no convolution step is required for the Toeplitz-based approach. This can often lead to large speedups, which are in particular important when using the Gram matrix in iterative solvers  (*paragraph needs references*).


With `NFFTTools.jl` on can calculate the kernel required in the Toeplitz with the function `calculateToeplitzKernel`. Multiplications with the Gram matrix can than be done using the function `calculateToeplitzKernel!`. The following outlines a complete example for the usage of both functions:

```julia
using NFFT, NFFTTools

Nx = 32;

trj = Float32.(rand(2, 1000) .- 0.5);
p = plan_nfft(trj, (2Nx,2Nx));

fftplan = plan_fft(zeros(ComplexF32, (2Nx,2Nx)); flags=NFFT.FFTW.MEASURE);

λ = Array{ComplexF32}(undef, 2Nx, 2Nx);

calculateToeplitzKernel!(λ, p, trj, fftplan);

x = randn(ComplexF32, Nx, Nx);
convolveToeplitzKernel!(x, λ);
```
