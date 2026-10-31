# # Cosine and Sine Transforms
#
# This guide shows how to compute the non-equidistant fast cosine and sine transforms (NFCT, NFST)
# of real data. NFFT.jl uses the definitions of NFFT3 (see
# [Mathematical Background](@ref "Other transforms")): nodes lie in ``[0, 1/2]^D``, and the array
# index of mode ``n_d`` is ``n_d + 1`` for the NFCT (``n_d = 0, \dots, N_d - 1``) and ``n_d`` for the
# NFST (``n_d = 1, \dots, N_d``). Instead of an adjoint, the real transforms have a transpose.
#
# ## Transform and transpose

using NFFT, LinearAlgebra, Random
Random.seed!(1)

k = rand(2, 1000) ./ 2            # nodes in [0, 1/2]²
f = randn(64, 48)

p = plan_nfct(k, size(f))
fHat = p * f
g = transpose(p) * fHat
size(fHat), size(g)

# The plans take the same options as `plan_nfft` (`reltol`, `dims`, `eltype`, `storage`,
# `nthreads`, `window`, `W`, `σ`).
#
# ## Check the result
#
# Against the direct evaluation:

relerr(a, b) = norm(a - b) / norm(b)
relerr(fHat, NDCTPlan(k, size(f)) * f), accuracy(p)

#-

q = plan_nfst(k, size(f); reltol=1e-12)
relerr(q * f, NDSTPlan(k, size(f)) * f), accuracy(q)

# ## Many transforms at once
#
# With `dims`, one plan transforms all columns (or slices) at the same nodes:

F = randn(64, 10)
pc = plan_nfct(rand(1, 300) ./ 2, size(F); dims=1)
size(pc * F)

# How NFFT.jl computes the transforms, as NFFTs of the even or odd extension with a mirrored halo,
# is explained in [How NFFT.jl Works](@ref "Real-valued transforms"); they cost about as much as a
# complex NFFT of the same size (see [Performance](@ref)).
