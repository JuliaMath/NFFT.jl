# # Getting Started
#
# A plan holds everything that depends only on the nodes and the array size: the chosen window,
# the binned nodes and the FFT plans. Creating it costs about as much as a few transforms;
# afterwards every transform with these nodes is fast.
#
# ## A first transform
#
# Nodes are given as a ``d × J`` matrix with entries in ``[-1/2, 1/2)``; in one dimension a
# vector works as well. The second argument is the size of the equidistant array.

using NFFT, LinearAlgebra, Random
Random.seed!(1)

J, N = 2000, (64, 48)
k = rand(2, J) .- 0.5
p = plan_nfft(k, N)

# The plan behaves like the ``J × |\bm{N}|`` matrix of the NDFT:

size(p)

# Applying it to an array of size `N` gives the values at the nodes, the adjoint maps back:

f = randn(ComplexF64, N)
fHat = p * f
g = adjoint(p) * fHat
size(fHat), size(g)

# ## How accurate is it?
#
# Without further options the plan aims at a relative error of ``10^{-8}``. `accuracy(p)` is the
# error the plan predicts for itself. We compare with the direct evaluation (NDFT), which NFFT.jl
# offers as a plan of its own:

pd = NDFTPlan(k, N)
accuracy(p), norm(p * f - pd * f) / norm(pd * f)

# A different tolerance is a keyword away. The plan then chooses a narrower or wider window:

p4 = plan_nfft(k, N; reltol=1e-4)
p12 = plan_nfft(k, N; reltol=1e-12)
[norm(q * f - pd * f) / norm(pd * f) for q in (p4, p12)]

# ## Avoiding allocations
#
# `p * f` allocates its output. In loops, use `mul!` with preallocated arrays:

mul!(fHat, p, f)
mul!(g, adjoint(p), fHat);

# ## Single precision
#
# `eltype` selects the precision of the data; the default follows the nodes. In single precision
# the positions of the nodes on the grid are still computed in double precision, so large grids
# do not lose accuracy.

p32 = plan_nfft(k, N; eltype=Float32, reltol=1e-5)
fHat32 = p32 * ComplexF32.(f)
eltype(fHat32), norm(fHat32 - pd * f) / norm(pd * f)

# ## Threads
#
# All stages run multi-threaded with `Threads.nthreads()` threads (start Julia with `julia -t 8`
# or `-t auto`). `nthreads` limits the number for one plan:

p1 = plan_nfft(k, N; nthreads=1)
p1 * f ≈ p * f

# One plan must not be used by several tasks at the same time, since it owns its workspace.
# `copy(p)` gives a plan with its own workspace that shares the read-only parts:

plans = [copy(p) for _ in 1:2]
results = Vector{Vector{ComplexF64}}(undef, 2)
Threads.@threads for i in 1:2
  results[i] = plans[i] * f
end
results[1] == results[2]

# ## New nodes
#
# `nodes!` exchanges the nodes and keeps all other settings; the number of nodes may change.

nodes!(p, rand(2, 3000) .- 0.5)
size(p * f)

# ## Without a plan
#
# For a single transform, `nfft` and `nfft_adjoint` create the plan internally:

fHat = nfft(k, f)
g = nfft_adjoint(k, N, fHat)
size(g)

# ## Where to go next
#
# - Transform only some dimensions, e.g. many coils or A-scans: [Transforms along Dimensions](@ref).
# - Real data, cosine and sine transforms, points that are non-equidistant on both sides, GPUs: the
#   other how-to guides in the navigation.
# - What the plan computes and why it is fast: [Mathematical Background](@ref) and
#   [How NFFT.jl Works](@ref).
