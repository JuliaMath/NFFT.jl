# The window as W polynomial pieces of degree P in s = 2t − 1 ∈ [−1, 1], t the fractional grid
# offset of a node. The weight of support point i (0-based) is φ(1 + i − W/2 − t).

struct PolyKernel{W,P1,T}
  coeffs::NTuple{P1,SVector{W,T}}   # coeffs[k][i]: coefficient of s^(k-1) for support point i
end

# Least-squares fit at Chebyshev points; the monomial basis is well conditioned on [−1, 1].
PolyKernel(φ, W::Integer, P::Integer; T=Float64) = PolyKernel(φ, Val(W), Val(P), T)
function PolyKernel(φ, ::Val{W}, ::Val{P}, ::Type{T}) where {W,P,T}
  ns = 4P
  s = [cospi((2k - 1) / (2ns)) for k in 1:ns]
  V = [sk^k for sk in s, k in 0:P]
  B = [φ(1 + i - W / 2 - (sk + 1) / 2) for sk in s, i in 0:W-1]
  C = least_squares!(V, B)
  PolyKernel{W,P + 1,T}(ntuple(k -> SVector{W,T}(view(C, k, :)), Val(P + 1)))
end

# The least-squares solution X (n × m) of A X ≈ B for A (r × n, r ≥ n, full rank) by Householder QR;
# A and B are overwritten. In Julia rather than LAPACK, so that planning needs no BLAS library
# (the C library of NFFT.jl is shipped without one).
function least_squares!(A::Matrix{Float64}, B::Matrix{Float64})
  r, n = size(A)
  for j in 1:n
    α = sqrt(sum(abs2, view(A, j:r, j)))
    α = A[j, j] > 0 ? -α : α
    v = A[j:r, j]; v[1] -= α
    β = sum(abs2, v)
    for M in (A, B), c in (M === A ? (j:n) : axes(B, 2))
      t = 2 * sum(v[i] * M[j+i-1, c] for i in eachindex(v)) / β
      for i in eachindex(v)
        M[j+i-1, c] -= t * v[i]
      end
    end
  end
  X = B[1:n, :]
  for c in axes(X, 2), j in n:-1:1          # back substitution with R = triu(A[1:n, :])
    X[j, c] = (X[j, c] - sum(A[j, l] * X[l, c] for l in j+1:n; init=0.0)) / A[j, j]
  end
  X
end

# The window kernel of the resampling (polynomial degree W+3, accurate to the window itself) and the
# correction factors 1/φ̂ of the modes −N÷2 … N−N÷2−1 per dimension.
# With W as a Val the type of the kernel is inferred.
make_kernel(φ, W::Integer, ::Type{T}) where {T} = make_kernel(φ, Val(W), T)
make_kernel(φ, ::Val{W}, ::Type{T}) where {W,T} = SymKernel(PolyKernel(φ, Val(W), Val(W + 3), T))
# φ::F: a function that is only passed on is otherwise not specialised on.
inv_kernel_ft(φ::F, W, N, Ñ, ::Type{T}, nthreads) where {F,T} =
  ntuple(d -> T.(1 ./ kernel_ft(φ, W, N[d], Ñ[d]; nthreads)), length(N))

# The window is symmetric: piece W−1−i at s equals piece i at −s. With wᵢ(s) = Eᵢ(s²) + s·Oᵢ(s²)
# one Horner chain in s² over the first ⌈W/2⌉ pieces gives all weights (the even/odd evaluation of
# ducc); for odd W the middle piece maps onto itself and is kept once.
struct SymKernel{W,H,NE,NO,T}
  even::NTuple{NE,SVector{H,T}}
  odd::NTuple{NO,SVector{H,T}}
end
function SymKernel(K::PolyKernel{W,P1,T}) where {W,P1,T}
  H = cld(W, 2)
  even = ntuple(k -> SVector{H,T}(ntuple(i -> K.coeffs[2k-1][i], Val(H))), Val(cld(P1, 2)))
  odd = ntuple(k -> SVector{H,T}(ntuple(i -> K.coeffs[2k][i], Val(H))), Val(P1 ÷ 2))
  SymKernel{W,H,cld(P1, 2),P1 ÷ 2,T}(even, odd)
end

# The Horner chain is unrolled at compile time as a sequence of statements: with a runtime index
# into the coefficient tuple the compiler keeps the tuple in memory (up to 8× slower for W ≥ 10),
# and a nested fused broadcast falls back to a generic path for W ≥ 14.
@generated function weights(K::SymKernel{W,H,NE,NO,T}, s::T) where {W,H,NE,NO,T}
  esteps = [:(e = muladd.(e, z, ce[$k])) for k in NE-1:-1:1]
  osteps = [:(o = muladd.(o, z, co[$k])) for k in NO-1:-1:1]
  quote
    Base.@_inline_meta
    ce = K.even; co = K.odd
    z = s * s
    e = ce[$NE]
    o = co[$NO]
    $(esteps...)
    $(osteps...)
    so = s .* o
    lo = e .+ so
    hi = e .- so
    SVector{W,T}($([:(lo[$i]) for i in 1:H]...), $([:(hi[$i]) for i in (W - H):-1:1]...))
  end
end

# Nodes (increasing) and weights of the n-point Gauss–Legendre rule, by Newton's method on the
# Legendre polynomial P_n (no eigenvalue solver: planning needs no LAPACK).
function gausslegendre(n)
  x = zeros(n); w = zeros(n)
  for i in 1:cld(n, 2)
    z = cospi((i - 0.25) / (n + 0.5))
    dp = 0.0
    for _ in 1:100
      p0, p1 = 1.0, z                         # P_{k-1}, P_k
      for k in 2:n
        p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
      end
      n == 1 && (p0, p1 = 1.0, z)
      dp = n * (z * p1 - p0) / (z^2 - 1)
      dz = p1 / dp
      z -= dz
      abs(dz) <= 4eps() && break
    end
    p0, p1 = 1.0, z
    for k in 2:n
      p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
    end
    dp = n * (z * p1 - p0) / (z^2 - 1)
    x[i] = -z; x[n+1-i] = z
    w[i] = w[n+1-i] = 2 / ((1 - z^2) * dp^2)
  end
  x, w
end

# φ̂(ξ) = ∫ φ(z) cos(2πξz) dz for ξ = k/Ñ, k = −N÷2 … N−N÷2−1. φ̂ is even and smooth on [0, 1/2], so
# it is computed by Gauss–Legendre quadrature at nc Chebyshev points and interpolated (Clenshaw)
# at the N modes; quadrature per mode costs O(N·nq) (130 ms for N = 2^18). 32 points reach the
# rounding floor of the quadrature itself (≈ 1e-13 relative).
function kernel_ft(φ::F, W, N, Ñ; nq=2W + 20, nc=32, nthreads=Threads.nthreads()) where {F}
  ks = -N÷2:N-N÷2-1
  kmax = max(N ÷ 2, N - N ÷ 2 - 1)
  N <= 4nc && (quad = window_quadrature(φ, W; nq); return [quad(k / Ñ) for k in ks])
  ft = WindowFT(φ, W, kmax / Ñ; nq, nc)
  out = Vector{Float64}(undef, N)
  @maybe_threads nthreads for r in chunks(eachindex(ks), nthreads)
    for i in r
      out[i] = ft(abs(ks[i]) / Ñ)
    end
  end
  out
end

# ξ ↦ ∫ φ(z) cos(2πξz) dz by Gauss–Legendre quadrature with nq points
function window_quadrature(φ::F, W; nq=2W + 20) where {F}
  x, w = gausslegendre(nq)
  z = x .* (W / 2); wz = w .* (W / 2) .* φ.(z)
  ξ -> sum(wz[q] * cospi(2ξ * z[q]) for q in eachindex(z))
end

# φ̂ on [0, ξmax] as a Chebyshev interpolant of degree nc − 1 of the quadrature
struct WindowFT
  c::Vector{Float64}
  ξmax::Float64
end
function WindowFT(φ::F, W, ξmax; nq=2W + 20, nc=32) where {F}
  quad = window_quadrature(φ, W; nq)
  θ = [π * (j - 0.5) / nc for j in 1:nc]
  fv = [quad(ξmax * (cos(t) + 1) / 2) for t in θ]
  c = [2 / nc * sum(fv[j] * cos(m * θ[j]) for j in 1:nc) for m in 0:nc-1]
  c[1] /= 2
  WindowFT(c, ξmax)
end
function (ft::WindowFT)(ξ)
  c = ft.c
  y = 2ξ / ft.ξmax - 1
  b1 = 0.0; b2 = 0.0
  for m in length(c):-1:2
    b1, b2 = muladd(2y, b1, -b2) + c[m], b1
  end
  muladd(y, b1, -b2) + c[1]
end
