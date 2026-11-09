# The resampling stage of an NFFT plan as a sparse matrix, for analysis and teaching. The
# transforms never use it: they evaluate the window weights on the fly.
module NFFTSparseArraysExt

using NFFT, SparseArrays
using NFFT: NFFTPlan, NFFTCore, full_core, support, halo

"""
    sparse(p::NFFTPlan) -> B

The resampling stage of the plan `p` (without batch dimensions, on the CPU) as a sparse
`J × prod(oversampled_size(p))` matrix of window weights, `W^D` entries per node: for an
oversampled array `g`, `resampling!(p, g, fHat)` computes `fHat = B * vec(g)` and
`resampling_adjoint!(p, fHat, g)` computes `vec(g) = B' * fHat`. Requires `using SparseArrays`.
"""
function SparseArrays.sparse(p::NFFTPlan{T,D}) where {T,D}
  c = full_core(p)
  c isa NFFTCore || throw(ArgumentError("sparse(p) needs a plan on the CPU"))
  b, W, Ñ = c.binned, c.W, c.Ñ
  h = halo(W)[1]
  n = W^D
  rows = Vector{Int}(undef, n * length(b.perm)); cols = similar(rows); vals = Vector{T}(undef, length(rows))
  LI = LinearIndices(Ñ)
  Threads.@threads for q in eachindex(b.xs)
    sw = map((x, m) -> support(c.kernel, x, m), b.xs[q], Ñ)
    for (t, I) in enumerate(CartesianIndices(ntuple(_ -> W, Val(D))))
      # padded index off + i, unpadded and periodic
      g = ntuple(d -> mod(first(sw[d]) + I[d] - h - 1, Ñ[d]) + 1, Val(D))
      e = (q - 1) * n + t
      rows[e] = b.perm[q]; cols[e] = LI[g...]
      vals[e] = prod(ntuple(d -> last(sw[d])[I[d]], Val(D)))
    end
  end
  sparse(rows, cols, vals, length(b.perm), prod(Ñ))
end

end
