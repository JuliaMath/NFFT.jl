# Grid geometry. The oversampled grid of size Ñ carries a halo of (below, above) = halo(W) points
# per dimension, so the resampling never wraps indices: the padded grid has at least Ñ + W points,
# node l (0-based, periodic) sits at padded index l + fld(W, 2) + 1.

halo(W) = (fld(W, 2), cld(W, 2))

# Size of the padded grid of an NFFT core: Ñ + W per dimension, plus 4 unused points at the end of
# each dimension but the last whose length Ñ + W is a multiple of 8. The FFT passes along the
# other dimensions run over lines whose stride is a product of these lengths; with many factors of
# two in it, FFTW's ESTIMATE plans lose up to 2.9× through cache set conflicts (4 extra points
# leave 2² in the stride, so complex Float32 lines stay 32-byte aligned). The extra points lie
# behind the upper halo, are zeroed with the grid and never read.
padded_size(Ñ::NTuple{D,Int}, W) where {D} = ntuple(d -> (n = Ñ[d] + W; d < D && n % 8 == 0 ? n + 4 : n), Val(D))

# ⌊x⌋ as Int for x ≥ 0 or integral (there truncation is the floor), without the range check of
# floor(Int, ·) for IEEE floats; other types take floor (DoubleFloats 1.12 truncates −3.0 to −2).
@inline floor_int(x::Base.IEEEFloat) = unsafe_trunc(Int, x)
@inline floor_int(x) = floor(Int, x)

# Grid position u ∈ [0, Ñ) of node x ∈ [−1/2, 1/2), at least in Float64: in Float32, x·Ñ is off by
# ≈ eps(Float32)·Ñ grid points (1e-2 relative error at Ñ ≈ 4·10^5). Binning and support must use
# this same function; computed separately, a node just below 0 lands in the last tile but gets its
# support at the start of the grid.
@inline function gridpos(x, Ñ::Int)
  xd = convert(wide(typeof(x)), x)
  u = (xd < 0 ? xd + 1 : xd) * Ñ
  u >= Ñ ? u - Ñ : u
end

# Offset into the padded grid (index of support point 0 minus 1) and the position s = 2t − 1 of
# node x in its window pieces (t the fractional grid offset), in one dimension. The first support
# node l0 lies in [1 − cld(W, 2), Ñ − fld(W, 2)].
@inline function support_offset(x, Ñ::Int, W, ::Type{T}) where {T}
  v = gridpos(convert(promote_type(typeof(x), T), x), Ñ) - W / 2
  f = floor(v)
  floor_int(f) + 1 + fld(W, 2), 2 * T(v - f) - one(T)
end

# Offset and window weights of node x in one dimension.
@inline function support(K::SymKernel{W,H,NE,NO,T}, x, Ñ::Int) where {W,H,NE,NO,T}
  off, s = support_offset(x, Ñ, W, T)
  off, weights(K, s)
end

# Kept modes −N÷2 … N−N÷2−1 sit in the two corners of the oversampled grid (1-based ranges).
keptranges(N, Ñ) = (1:N-N÷2, Ñ-N÷2+1:Ñ)

# Grid index (1-based, unpadded) of mode index i = 1…N (mode i − 1 − N÷2).
@inline gridindex(i, N, Ñ) = mod(i - 1 - N ÷ 2, Ñ) + 1
