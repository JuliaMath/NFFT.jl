# # Choosing Parameters
#
# This guide shows how to override the automatic choice of window, width and oversampling, how to
# use a window of your own and how to let FFTW measure its plans. Usually `reltol` is all you need;
# how the plan turns it into parameters is explained in [Accuracy and Parameters](@ref).

using NFFT, LinearAlgebra, Random
Random.seed!(1)

N = (48, 48)
k = rand(2, 4000) .- 0.5
f = randn(ComplexF64, N)
exact = NDFTPlan(k, N) * f
relerr(p) = norm(p * f - exact) / norm(exact)

# ## Restrict the window family
#
# `window = KaiserBessel()` or `ExpSemicircle()` restricts the search to one family; the default
# `:auto` lets both compete:

pkb = plan_nfft(k, N; window=KaiserBessel(), reltol=1e-8)
pes = plan_nfft(k, N; window=ExpSemicircle(), reltol=1e-8)
NFFT.parameters(pkb), NFFT.parameters(pes)

# ## Set the width and the oversampling factor
#
# `W` and `σ` restrict the search as well; with both given, only the window shape comes from the
# table. ``W`` may be odd (see [Mathematical Background](@ref "Window, width and oversampling") for
# ``W = 2m``). A `σ` between the tabulated values (steps of 0.05) uses the shape of the nearest one,
# and `accuracy(p)` is then an estimate.

p = plan_nfft(k, N; W=7, σ=1.5)
NFFT.parameters(p), relerr(p)

# ## Use a window of your own
#
# Any other window can be used through [`CustomWindow`](@ref). The function receives the offset
# ``z`` in grid points and the width ``W``; it must be symmetric and vanish for ``|z| \ge W/2``.
# Without a table, `W` and `σ` must be given and `accuracy(p)` is unknown (`NaN`):

gauss = CustomWindow((z, W) -> abs(z) < W / 2 ? exp(-(2z / W)^2 * 9) : 0.0)
pg = plan_nfft(k, N; window=gauss, W=8, σ=2.0)
relerr(pg), accuracy(pg)

# ## Let FFTW measure its plans
#
# The FFT on the oversampled grid uses FFTW. `fft = FFTWBackend(flags=FFTW.MEASURE)` lets FFTW
# measure its plans, which takes longer when creating the NFFT plan:
#
# ```julia
# using FFTW
# p = plan_nfft(k, N; fft=FFTWBackend(flags=FFTW.MEASURE))
# ```
#
# The FFT itself is pruned to the lines that are not zero; see
# [How NFFT.jl Works](@ref "The FFT").
