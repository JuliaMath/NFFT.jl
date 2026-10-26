# # Cosine and Sine Transforms
#
# For real data with even or odd symmetry the non-equidistant fast cosine and sine transforms
# (NFCT, NFST) replace the complex NFFT. NFFT.jl uses the definitions of NFFT3: for a real array
# ``\bm{f}`` of size ``\bm{N}`` and nodes ``\bm{k}_j \in [0, 1/2]^D``,
# ```math
# \text{NFCT:}\quad \hat{f}_j = \sum_{\bm{n}=\bm{0}}^{\bm{N}-\bm{1}} f_{\bm{n}} \prod_{d=1}^D \cos(2\pi n_d k_{d,j}),
# \qquad
# \text{NFST:}\quad \hat{f}_j = \sum_{\bm{n}=\bm{1}}^{\bm{N}} f_{\bm{n}} \prod_{d=1}^D \sin(2\pi n_d k_{d,j}).
# ```
# The array index of mode ``n_d`` is ``n_d + 1`` for the NFCT and ``n_d`` for the NFST. Instead of
# an adjoint, the real transforms have a transpose.

using NFFT, LinearAlgebra, Random
Random.seed!(1)

k = rand(2, 1000) ./ 2            # nodes in [0, 1/2]²
f = randn(64, 48)

p = plan_nfct(k, size(f))
fHat = p * f
g = transpose(p) * fHat
size(fHat), size(g)

# The plans take the same options as the NFFT (`reltol`, `dims`, `eltype`, `nthreads`, `window`,
# `W`, `σ`) and are checked here against the direct evaluation:

relerr(a, b) = norm(a - b) / norm(b)
relerr(fHat, NDCTPlan(k, size(f)) * f), accuracy(p)

#-

q = plan_nfst(k, size(f); reltol=1e-12)
relerr(q * f, NDSTPlan(k, size(f)) * f), accuracy(q)

# ## How it works
#
# The NFCT is an NFFT of the even extension of ``\bm{f}`` (size ``2\bm{N}``), the NFST one of the
# odd extension. Its oversampled grid of ``2\tilde{\bm{N}}`` points is even (odd) about ``0`` and
# ``\tilde{\bm{N}}``, so only the points ``0, \dots, \tilde{N}_d`` are stored:
#
# 1. **Resampling correction**: divide by ``\hat\varphi`` as in the NFFT.
# 2. **FFT**: a DCT-I (DST-I) along each dimension, computed as a real FFT of the extended line.
# 3. **Resampling**: the halo of the grid is filled with mirror images (with a sign change for the
#    NFST), so the resampling loops are the ones of the NFFT, on a real grid.
#
# The cost is about that of a complex NFFT of size ``\bm{N}``:

tmin(f) = (f(); minimum(@elapsed(f()) for _ in 1:5))
kc = rand(2, 2^18) ./ 2; kf = rand(2, 2^18) .- 0.5
fr = randn(256, 256); fc = complex.(fr)
pc = plan_nfct(kc, (256, 256)); pf = plan_nfft(kf, (256, 256))
yr = zeros(2^18); yc = zeros(ComplexF64, 2^18)
(NFCT_ms = 1e3 * tmin(() -> mul!(yr, pc, fr)), NFFT_ms = 1e3 * tmin(() -> mul!(yc, pf, fc)))
