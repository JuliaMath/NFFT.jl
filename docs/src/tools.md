 # Tools

The packages `AbstractNFFT.jl` and `NFFT.jl` are on purpose kept small and focussed. Additional tooling that relates to the NFFT is offloaded into a package `NFFTTools.jl`. 

## Sampling Density

The first tool that `NFFTTools.jl` offers is the computation of the sampling density. To motivate this let us have a look at the normalized variant of the DFT matrix ``\bm{F}``. ``\bm{F}`` is unitary, which
means that
```math
 \bm{F}^{\mathsf{H}} \bm{F} = \bm{I} 
```
where ``\bm{I}`` is the identity matrix. In other words, ``\bm{F}^{\mathsf{H}}`` is the inverse of ``\bm{F}^{\mathsf{H}}``.

Now let's switch to the NFFT. Have you already wondered why we don't call the adjoint the inverse? Well it's because in general we have
```math
 \bm{A}^{\mathsf{H}} \bm{A} \neq \bm{I} 
```
In fact, the inverse of the NFFT is a much more complicated subject since the linear system ``\bm{A} \bm{f} = \hat{\bm{f}}`` can have one, no or many solutions because ``\bm{A}``can be under- or over-determined.

The good news is that in most cases, with ``J \approx N`` and no complete clustering of the sampling nodes, one can find a diagonal weighting matrix ``\bm{W} = \left( w_j \right)_{j=1}^{J}``such that
```math
 \bm{A}^{\mathsf{H}} \bm{W} \bm{A} \approx \bm{I} 
```
The weights ``w_j`` can be considered to we quadrature weights that account for the *area* covered by the node ``\bm{k}_j``. 

`NFFTTools.jl` provides the function `sdc` that takes an existing plan and calculated suitable density weights:
```julia
weights = sdc(p, iters = 10)
```
The function implements the method proposed in [Pipe & Menon, 1999. Mag Reson Med, 186, 179](https://doi.org/10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V).

## Toeplitz Kernel

The aforementioned matrix ``\bm{A}^{\mathsf{H}} \bm{W} \bm{A}`` arises when solving linear system of the form
```math
\bm{A} \bm{f} = \hat{\bm{f}}
```
which can be done via the normal equation
```math
\bm{A}^{\mathsf{H}} \bm{W} \bm{A} \bm{f} = \bm{A}^{\mathsf{H}} \bm{W} \hat{\bm{f}}
```
The normal or Gram matrix ``\bm{A}^{\mathsf{H}} \bm{W} \bm{A}`` has a [Toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix) structure. For multi-dimensional NFFT is is a block Toeplitz matrix with Toeplitz blocks. A Toeplitz matrix (and its block variants) can be embedded into a [circulant matrix](https://en.wikipedia.org/wiki/Circulant_matrix) of twice the size in each dimension. Circulant matrices are known to be diagonalizable by ordinary FFTs. This means we can multiply with ``\bm{A}^{\mathsf{H}} \bm{W} \bm{A}`` by just two FFTs of size ``2\bm{N}``, which is basically the same amount of FFTs as are required for an NFFT-based calculation of matrix-vector products with ``\bm{A}^{\mathsf{H}} \bm{W} \bm{A}``. But the important difference is that no convolution step is required for the Toeplitz-based approach. This can often lead to speedups, which are in particular important when using the Gram matrix in iterative solvers  (see [Fessler et al., IEEE Trans. Sig. Proc., 53, 9](https://doi.org/10.1109/TSP.2005.853152) for the mathematical background).


With `NFFTTools.jl` one can calculate the kernel required to exploit the Toeplitz structure with the function `calculateToeplitzKernel`. Multiplications with the Gram matrix can then be done using the function `convolveToeplitzKernel!`. The following outlines a complete example for the usage of both functions:

```julia
using NFFT, NFFTTools, FFTW

N = (32, 32)                            # signal size
Ñ = 2 .* N                              # oversampled signal size

k = Float32.(rand(2, 1000) .- 0.5)      # 2D sampling nodes
p = plan_nfft(k, Ñ)                     # 2D NFFT plan
fftplan = plan_fft(zeros(ComplexF32, Ñ));

λ = Array{ComplexF32}(undef, Ñ)         # pre-allocate Toeplitz kernel 
calculateToeplitzKernel!(λ, p, k,fftplan)       # calculate Toeplitz kernel 

y = randn(ComplexF32, Ñ)
convolveToeplitzKernel!(y, λ)           # multiply with Gram matrix

```
