# # Non-equidistant in Both Domains
#
# The NNFFT (type 3 in the terminology of FINUFFT) has arbitrary real points in both domains:
# values ``f_l`` at sources ``\bm{y}_l \in \mathbb{R}^D`` are transformed to targets
# ``\bm{k}_j \in \mathbb{R}^D``,
# ```math
# \hat{f}_j = \sum_{l=1}^{L} f_l \, \mathrm{e}^{-2\pi\mathrm{i}\,\bm{k}_j\cdot\bm{y}_l},
# \qquad
# f_l = \sum_{j=1}^{J} \hat{f}_j \, \mathrm{e}^{2\pi\mathrm{i}\,\bm{k}_j\cdot\bm{y}_l} \quad \text{(adjoint)}.
# ```
# Neither the sources nor the targets need to lie in a fixed interval. `plan_nnfft(k, y)` takes the
# targets `k` (``D \times J``) and the sources `y` (``D \times L``) and the same options as
# `plan_nfft` (`reltol`, `eltype`, `storage`, `nthreads`).
#
# ## How it works
#
# The sources are spread with a window onto an equidistant grid of spacing ``h``, an NFFT evaluates
# the Fourier series of that grid at the scaled targets ``h \bm{k}_j``, and a division by the
# Fourier transform of the window undoes the spreading (Elbel and Steidl 1998; Barnett et al.
# 2019). Both node sets are first centred, which is a phase factor before and after. The size of the
# inner NFFT is about ``4\sigma^2 Y_d K_d`` per dimension, where ``Y_d`` and ``K_d`` are the
# half-widths of the sources and targets: the cost grows with the space-frequency product of the
# data, not with the number of points.
#
# ## The Fourier transform of a function sampled at random points
#
# A Gaussian ``f(y) = \mathrm{e}^{-\pi y^2}`` has the Fourier transform
# ``\hat{f}(k) = \int_{-\infty}^{\infty} f(y) \mathrm{e}^{-2\pi\mathrm{i}ky} \,\mathrm{d}y
# = \mathrm{e}^{-\pi k^2}``. With samples at random points and trapezoidal weights, the integral
# becomes an NNFFT, evaluated here at Chebyshev points in ``[-3, 3]``:

using NFFT, LinearAlgebra, Random, CairoMakie
CairoMakie.activate!(type="png") #hide
Random.seed!(1)

y = sort(12 .* rand(4000) .- 6)                 # sources in [-6, 6]
w = [y[2] - y[1]; (y[3:end] .- y[1:end-2]) ./ 2; y[end] - y[end-1]]
k = 3 .* cospi.((2 .* (1:300) .- 1) ./ 600)    # targets in [-3, 3]

p = plan_nnfft(reshape(k, 1, :), reshape(y, 1, :); reltol=1e-10)
F = p * complex.(w .* exp.(-π .* y .^ 2))
maximum(abs, F .- exp.(-π .* k .^ 2))

# The remaining difference is the error of the trapezoidal rule on the random points, not that of the
# NNFFT. Against the direct sum:

pd = NNDFTPlan(reshape(k, 1, :), reshape(y, 1, :))
norm(F - pd * complex.(w .* exp.(-π .* y .^ 2))) / norm(F), accuracy(p)

#-

fig = Figure(size=(640, 300))
ax = Axis(fig[1, 1]; xlabel="k", ylabel="Re F(k)")
lines!(ax, -3:0.01:3, k -> exp(-π * k^2); color=:gray, label="exp(-πk²)")
scatter!(ax, k, real.(F); markersize=5, label="NNFFT")
axislegend(ax)
fig

# ## In several dimensions
#
# Sources and targets in 2D, off-centre on purpose, against the direct sum:

ys = 40 .* rand(2, 5000) .+ [100.0, -20.0]
ks = 2 .* rand(2, 3000) .- [0.3, 1.5]
f = randn(ComplexF64, 5000)
p2 = plan_nnfft(ks, ys; reltol=1e-8)
norm(p2 * f - NNDFTPlan(ks, ys) * f) / norm(p2 * f), accuracy(p2)

# The plan reports the size of the inner NFFT and the window of the spreading:

p2

# The adjoint is the transform from the targets back to the sources:

v = randn(ComplexF64, 3000)
norm(adjoint(p2) * v - adjoint(NNDFTPlan(ks, ys)) * v) / norm(adjoint(p2) * v)
