# # Transforms along Dimensions
#
# This guide shows how to transform some dimensions of an array and keep the others, for example
# the coils of a multi-coil MRI acquisition, the A-scans of an OCT B-scan, time frames or channels.
# With `dims` a plan transforms along a contiguous range of dimensions; all other dimensions are
# batch dimensions and pass through unchanged. How the plan processes the batches is explained in
# [How NFFT.jl Works](@ref "Batch strategies").
#
# ## Transform one dimension of a matrix
#
# A 1D NFFT along the first dimension of a ``16 × 20`` matrix, at 11 nodes:

using NFFT, LinearAlgebra, Random
Random.seed!(1)

y = rand(11) .- 0.5
f = randn(ComplexF64, 16, 20)
p1 = plan_nfft(y, size(f); dims=1)
size(p1 * f)

# The nodes replace the transformed dimension. Along the second dimension:

p2 = plan_nfft(y, size(f); dims=2)
size(p2 * f)

# The result is one NFFT per column (or row):

q = plan_nfft(y, 16)
p1 * f ≈ reduce(hcat, [q * f[:, c] for c in 1:20])

# ## Transform all coils of an MRI acquisition
#
# Images of ``L`` receive coils are stored as an ``N_1 × N_2 × L`` array, and every coil is sampled
# at the same 2D trajectory. A plan with `dims = 1:2` transforms all coils in one call and returns a
# ``J × L`` array:

L = 8
N = (128, 128)
k = rand(2, 20_000) .- 0.5
coils = randn(ComplexF64, N..., L)
p = plan_nfft(k, size(coils); dims=1:2)
data = p * coils
size(data)

# The adjoint maps the ``J × L`` data back to the coil images:

size(adjoint(p) * data)

# One plan for all coils is faster than a loop over the coils with a plan for one image: depending
# on the number of threads it computes the window weights of a node once for all coils, or runs
# the coils in parallel:

single = plan_nfft(k, N)
out = similar(data)
loop!(out, single, coils) = for c in 1:L
  mul!(view(out, :, c), single, view(coils, :, :, c))
end
loop!(out, single, coils); mul!(data, p, coils)        # compile
t_loop = minimum(@elapsed(loop!(out, single, coils)) for _ in 1:5)
t_batch = minimum(@elapsed(mul!(data, p, coils)) for _ in 1:5)
(loop_ms=1e3t_loop, batch_ms=1e3t_batch, agree=out ≈ data)

# ## Transform many short signals
#
# For many small transforms, such as the 512 A-scans of an OCT B-scan, the plan runs whole slices
# in parallel, one per thread (with several threads, e.g. `julia -t auto`). Again a single plan
# with `dims` does it:

k1 = rand(1, 2048) .- 0.5
pm = plan_nfft(k1, (2048, 512); dims=1)
f1 = randn(ComplexF64, 2048, 512); y1 = pm * f1
single1 = plan_nfft(k1, 2048)
loop1!(y, p, f) = for c in axes(f, 2)
  mul!(view(y, :, c), p, view(f, :, c))
end
loop1!(y1, single1, f1)
(loop_ms=1e3minimum(@elapsed(loop1!(y1, single1, f1)) for _ in 1:5),
 dims_ms=1e3minimum(@elapsed(mul!(y1, pm, f1)) for _ in 1:5))

# For real-valued interferograms, as in OCT, see [Real Data](@ref).
