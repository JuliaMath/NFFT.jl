# Grid geometry. The oversampled grid of size Ñ carries a halo of (below, above) = halo(W) points
# per dimension, so the resampling never wraps indices: the padded grid has Ñ + W points, node l
# (0-based, periodic) sits at padded index l + fld(W, 2) + 1.

halo(W) = (fld(W, 2), cld(W, 2))

# Grid position u ∈ [0, Ñ) of node x ∈ [−1/2, 1/2), always in Float64: in Float32, x·Ñ is off by
# ≈ eps(Float32)·Ñ grid points (1e-2 relative error at Ñ ≈ 4·10^5). Binning and support must use
# this same function; computed separately, a node just below 0 lands in the last tile but gets its
# support at the start of the grid.
@inline function gridpos(x, Ñ::Int)
  xd = Float64(x)
  u = (xd < 0 ? xd + 1 : xd) * Ñ
  u >= Ñ ? u - Ñ : u
end

# Offset into the padded grid (index of support point 0 minus 1) and the window weights of node x
# in one dimension. The first support node l0 lies in [1 − cld(W, 2), Ñ − fld(W, 2)].
@inline function support(K::SymKernel{W,H,NE,NO,T}, x, Ñ::Int) where {W,H,NE,NO,T}
  v = gridpos(x, Ñ) - W / 2
  f = floor(v)
  t = T(v - f)
  l0 = unsafe_trunc(Int, f) + 1
  l0 + fld(W, 2), weights(K, 2t - one(T))
end

# Kept modes −N÷2 … N−N÷2−1 sit in the two corners of the oversampled grid (1-based ranges).
keptranges(N, Ñ) = (1:N-N÷2, Ñ-N÷2+1:Ñ)

# Grid index (1-based, unpadded) of mode index i = 1…N (mode i − 1 − N÷2).
@inline gridindex(i, N, Ñ) = mod(i - 1 - N ÷ 2, Ñ) + 1
