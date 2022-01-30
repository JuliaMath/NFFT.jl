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
The weights ``w_k`` can be considered to we quadrature weights that account for the *area* covered by the node ``\bm{x}_k``

```julia

# create a 10x10 grid of unit spaced sampling points
N = 10
g = (0:(N-1)) ./ N .- 0.5  
x = vec(ones(N) * g')
y = vec(g * ones(N)')
nodes = cat(x',y', dims=1)

# approximate the density weights
p = plan_nfft(nodes, (N,N), m = 5, σ = 2.0); 
weights = sdc(p, iters = 10)

# test if they approximate the true weights (1/(N*N))
@test all( (≈).(vec(weights), 1/(N*N), rtol=1e-7) )
```

## Toeplitz Kernel

TODO
