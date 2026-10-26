# # Transforms along Dimensions
#
# Many applications transform only some dimensions of an array and keep the others: the coils of
# a multi-coil MRI acquisition, the lateral position in OCT, time frames or channels. With `dims`
# the plan transforms along a contiguous range of dimensions; all other dimensions are batch
# dimensions and pass through unchanged.
#
# ## One dimension of a matrix
#
# A 1D NFFT along the first dimension of a ``16 × 20`` matrix, with 11 nodes:

using NFFT, LinearAlgebra, Random
Random.seed!(1)

y = rand(11) .- 0.5
f = randn(ComplexF64, 16, 20)
p1 = plan_nfft(y, size(f); dims=1)
size(p1 * f)

# The nodes replace the transformed dimension. Along the second dimension:

p2 = plan_nfft(y, size(f); dims=2)
size(p2 * f)

# The result equals one NFFT per column (or row):

q = plan_nfft(y, 16)
p1 * f ≈ reduce(hcat, [q * f[:, c] for c in 1:20])

# ## Several dimensions: multi-coil MRI
#
# Images of ``L`` receive coils are stored as an ``N_1 × N_2 × L`` array; each coil is sampled at
# the same 2D trajectory. A plan with `dims = 1:2` transforms all coils at once and returns a
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

# ## Why not a loop?
#
# The plan processes all slices together. For batch dimensions in front of `dims` it evaluates the
# window once per node and applies it to all slices with SIMD instructions; batch dimensions
# behind `dims` are first transposed into this layout. Compared with a loop over single plans:

single = plan_nfft(k, N)
out = similar(data)
loop!(out, single, coils) = for c in 1:L
  mul!(view(out, :, c), single, view(coils, :, :, c))
end
loop!(out, single, coils); mul!(data, p, coils)        # compile
t_loop = minimum(@elapsed(loop!(out, single, coils)) for _ in 1:5)
t_batch = minimum(@elapsed(mul!(data, p, coils)) for _ in 1:5)
(loop_ms=1e3t_loop, batch_ms=1e3t_batch, speedup=t_loop / t_batch)

# and the results agree:

out ≈ data

# Batch kernels are used for 1D and 2D transforms. For 3D transforms, and for batch dimensions on
# both sides of `dims`, the plan loops over the slices.
