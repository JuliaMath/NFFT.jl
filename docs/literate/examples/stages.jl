# # Stages of the NFFT
#
# The NFFT consists of three stages, resampling correction, FFT and resampling (see
# [Mathematical Background](@ref)). The stage interface exposes them one by one on the
# oversampled array of size `oversampled_size(p)`. This is useful for methods that operate on the
# oversampled grid, such as Toeplitz embeddings, for teaching, and for profiling.

using NFFT, LinearAlgebra, Random, CairoMakie
CairoMakie.activate!(type="png") #hide
Random.seed!(1)

N = (256, 256)
k = rand(2, 100_000) .- 0.5
p = plan_nfft(k, N)
Ñ = oversampled_size(p)

# ## Direct NFFT

f = randn(ComplexF64, N)
g = zeros(ComplexF64, Ñ)
fHat = zeros(ComplexF64, size(k, 2))

resampling_correction!(p, f, g)   # D: divide by φ̂, zero-pad
oversampled_fft!(p, g)            # F
resampling!(p, g, fHat)           # B: evaluate at the nodes
fHat ≈ p * f

# ## Adjoint NFFT
#
# The adjoint runs the adjoint stages in reverse order:

back = zeros(ComplexF64, N)
resampling_adjoint!(p, fHat, g)           # Bᴴ: add node values to the grid
oversampled_bfft!(p, g)                   # Fᴴ
resampling_correction_adjoint!(p, g, back) # Dᴴ: crop, divide by φ̂
back ≈ adjoint(p) * fHat

# ## Where the time goes
#
# Timing the stages shows which part dominates. The FFT stages of this interface transform the
# full oversampled array, while `p * f` skips the zero padding, so the sum is a bit larger than a
# complete transform.

tmin(op) = (op(); minimum(@elapsed(op()) for _ in 1:5))
t = [tmin(() -> resampling_correction!(p, f, g)),
     tmin(() -> oversampled_fft!(p, g)),
     tmin(() -> resampling!(p, g, fHat)),
     tmin(() -> resampling_adjoint!(p, fHat, g)),
     tmin(() -> oversampled_bfft!(p, g)),
     tmin(() -> resampling_correction_adjoint!(p, g, back))]

fig = Figure(size=(640, 300))
ax = Axis(fig[1, 1]; xlabel="time / ms", yticks=(1:2, ["NFFT", "adjoint"]))
colors = Makie.wong_colors()
names = ["resampling correction", "FFT", "resampling"]
for (row, idx) in ((1, 1:3), (2, [6, 5, 4]))
  left = 0.0
  for (c, i) in enumerate(idx)
    barplot!(ax, [row], [1e3t[i]]; direction=:x, offset=left, fillto=left, color=colors[c])
    left += 1e3t[i]
  end
end
Legend(fig[2, 1], [PolyElement(color=colors[c]) for c in 1:3], names;
       orientation=:horizontal, framevisible=false)
fig

# The stage interface needs a plan without batch dimensions (`dims` covering all dimensions).
