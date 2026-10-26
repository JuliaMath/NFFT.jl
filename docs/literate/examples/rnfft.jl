# # Real Data
#
# This guide shows how to transform real values at the nodes with `plan_rnfft`, the NFFT
# counterpart of FFTW's `rfft`/`brfft`. Real values at the nodes mean Hermitian coefficients, and
# the plan works with the half ``n_1 = 0, \dots, \lfloor N_1/2 \rfloor`` of them (definition in
# [Mathematical Background](@ref "Other transforms"), implementation in
# [How NFFT.jl Works](@ref "Real-valued transforms")).
#
# For an array of logical size ``\bm{N}`` the input of the plan has size
# ``(\lfloor N_1/2 \rfloor + 1, N_2, \dots)``; the other dimensions are indexed as in `plan_nfft`.
# The direct transform returns real values, the adjoint takes real values. The two are adjoint
# with respect to the real inner product, so they can be used in iterative solvers.
#
# ## Reconstruct OCT depth profiles
#
# In spectral-domain OCT, each A-scan is a real interferogram sampled at wavenumbers that are not
# equidistant; the depth profile is its Fourier transform. With `dims = 1` one plan reconstructs
# all A-scans at once. A synthetic B-scan with two reflecting layers:

using NFFT, LinearAlgebra, Random, CairoMakie
CairoMakie.activate!(type="png") #hide
Random.seed!(1)

J, Nz, L = 2048, 1024, 256                      # samples per A-scan, depth, A-scans
k = reshape(sort(0.45 .* sin.(range(-π / 2, π / 2; length=J))), 1, J)   # non-linear in k
depth(l) = (180 + round(Int, 20sin(2π * l / L)), 420 + round(Int, 30cos(2π * l / L)))
y = zeros(J, L)
for l in 1:L, (z, a) in zip(depth(l), (1.0, 0.6))
  y[:, l] .+= a .* cospi.(2z .* k[1, :])
end
y .+= 0.2 .* randn(J, L)

# The depth profiles are the adjoint applied to the interferograms, weighted with the spacing of
# the wavenumbers (a simple density compensation):

p = plan_rnfft(k, (2Nz, L); dims=1)
w = [k[2] - k[1]; (k[3:end] .- k[1:end-2]) ./ 2; k[end] - k[end-1]]
h = adjoint(p) * (w .* y)                       # depth profiles, n = 0 … Nz
size(h)

#-

dB = 20log10.(abs.(h) ./ maximum(abs, h))
fig = Figure(size=(640, 360))
ax = Axis(fig[1, 1]; xlabel="A-scan", ylabel="depth", yreversed=true)
hm = heatmap!(ax, 1:L, 0:Nz, dB'; colormap=:grays, colorrange=(-40, 0))
Colorbar(fig[1, 2], hm; label="dB")
fig

# The plan runs the 256 A-scans in parallel:

t = minimum(@elapsed(mul!(h, adjoint(p), w .* y)) for _ in 1:5)
(ms=1e3t, threads=Threads.nthreads())

# ## Check the result
#
# Against the direct evaluation of the definition:

N = (64, 48)
kr = rand(2, 500) .- 0.5
pr = plan_rnfft(kr, N)
hr = randn(ComplexF64, size_in(pr))
modes(I) = (I[1] - 1, I[2] - 1 - N[2] ÷ 2)
ref = [sum(real(hr[I] * cispi(-2 * sum(modes(I) .* Tuple(kr[:, j])))) for I in CartesianIndices(hr)) for j in 1:500]
norm(pr * hr - ref) / norm(ref), accuracy(pr)
