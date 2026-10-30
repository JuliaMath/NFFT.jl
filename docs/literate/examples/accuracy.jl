# # Accuracy and Parameters
#
# The NFFT approximates the NDFT; its error depends on the window function ``\varphi``, the window
# width ``W`` (in grid points) and the oversampling factor ``\sigma`` (see
# [Mathematical Background](@ref)). NFFT.jl lets you state the error you need and chooses these
# three for you.
#
# ## From `reltol` to parameters
#
# `reltol` is the requested relative error in the ``\ell^2`` norm, for any nodes and data. For each
# combination of window family, ``W`` and ``\sigma`` NFFT.jl carries a table with the optimised
# shape parameter of the window and a bound on its aliasing error that also holds for nodes on grid
# points (Cartesian sampling, where the aliases of all nodes add up coherently). From all entries
# whose predicted error is below `reltol`, the plan takes the one with the lowest predicted run time. The cost model
# weighs the resampling (``J W^D``) against the FFT (``\sigma^D |\bm{N}| \log``), so the choice
# depends on the dimension and on the ratio of nodes to grid size. Plans on a GPU use a cost model
# of their own (see [Parameter choice on a GPU](@ref)).

using NFFT, LinearAlgebra, Random, CairoMakie
CairoMakie.activate!(type="png") #hide
Random.seed!(1)

N = (48, 48)
k = rand(2, 4000) .- 0.5
f = randn(ComplexF64, N)
exact = NDFTPlan(k, N) * f
relerr(p) = norm(p * f - exact) / norm(exact)

p = plan_nfft(k, N; reltol=1e-6)

# The plan shows what it chose; `NFFT.parameters` returns the same as a named tuple:

NFFT.parameters(p)

# Over a range of tolerances, the measured error stays below the requested one and below
# `accuracy(p)`, the bound the plan predicts:

tols = [10.0^-i for i in 2:13]
plans = [plan_nfft(k, N; reltol=t) for t in tols]
measured = relerr.(plans)
predicted = accuracy.(plans)

fig = Figure(size=(640, 380))
ax = Axis(fig[1, 1]; xscale=log10, yscale=log10, xreversed=true,
          xlabel="requested reltol", ylabel="relative error",
          xticks=LogTicks(-14:2:-2), yticks=LogTicks(-14:2:-2))
lines!(ax, tols, tols; color=:gray60, linestyle=:dash, label="reltol")
scatterlines!(ax, tols, predicted; label="accuracy(p)")
scatterlines!(ax, tols, measured; label="measured")
axislegend(ax; position=:rt)
fig

# The chosen widths and oversampling factors:

[(reltol=t, W=NFFT.parameters(q).W, σ=NFFT.parameters(q).σ) for (t, q) in zip(tols, plans)]

# Below about ``10^{-13}`` rounding errors dominate in double precision (more for large ``N``: node
# positions are accurate to ``\approx \varepsilon \max N``); in single precision the floor is about
# ``3 \cdot 10^{-6}``. A `reltol` below what any parameters reach throws an `ArgumentError` instead of
# returning a less accurate plan, unless `W` and `σ` are both given. Without `reltol` a plan uses
# [`default_reltol`](@ref) of its element type: ``10^{-8}``, in single precision ``1.2 \cdot 10^{-5}``.
#
# Element types wider than `Float64` (`Double64` from DoubleFloats.jl, `Float128` from Quadmath.jl)
# compute the window fit, the correction factors and the grid positions in their own precision and,
# with an FFT for them (`fft = FFTABackend()`), reach correspondingly smaller errors: `reltol = 1e-25`
# with `Double64` gives about ``10^{-26}``, `1e-28` with `Float128` about ``10^{-29}``. Beyond the table
# (``W > 16``) the plan uses the Kaiser–Bessel window with ``\beta = \pi W (1 - 1/(2\sigma))``, up to
# ``W = 48``; `BigFloat` (256 bits) reaches ``10^{-40}`` with ``W = 39``. The resampling kernels are
# unrolled in ``W``: with `Double64` and ``W \approx 30`` in 2D the first transform compiles for
# minutes. The FFT of FFTA.jl loses accuracy with the length (up to ``0.3 \, n \varepsilon``); the
# predicted error accounts for it.

# ## Window families
#
# Two families have parameter tables:
#
# - [`KaiserBessel`](@ref): the continuous Kaiser–Bessel window,
# - [`ExpSemicircle`](@ref): the "exponential of semicircle" window
#   ``\exp(\beta((1-(2z/W)^2)^{e_0} - 1))``.
#
# The default `window = :auto` lets both compete. How the error falls with the width, at ``\sigma = 2`` and ``\sigma = 1.25``:

Ws = 4:14
fig = Figure(size=(640, 380))
ax = Axis(fig[1, 1]; yscale=log10, xlabel="window width W", ylabel="relative error",
          xticks=Ws, yticks=LogTicks(-14:2:-2))
for (σ, style) in ((2.0, :solid), (1.25, :dash)), (win, name) in ((KaiserBessel(), "Kaiser–Bessel"),
                                                                   (ExpSemicircle(), "ES"))
  errs = [relerr(plan_nfft(k, N; window=win, W, σ)) for W in Ws]
  scatterlines!(ax, Ws, errs; linestyle=style, label="$name, σ = $σ")
end
axislegend(ax; position=:lb)
fig

# A smaller ``\sigma`` needs a wider window for the same error, but its FFT is cheaper; which is
# faster overall depends on ``D`` and on the number of nodes. This is the trade-off the automatic
# choice makes.

# To set window, width or oversampling yourself, see [Choosing Parameters](@ref); all plan options
# are listed in the [API](@ref "Options").
