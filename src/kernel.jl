# The window as W polynomial pieces of degree P in s = 2t − 1 ∈ [−1, 1], t the fractional grid
# offset of a node. The weight of support point i (0-based) is φ(1 + i − W/2 − t).

struct PolyKernel{W,P1,T}
  coeffs::NTuple{P1,SVector{W,T}}   # coeffs[k][i]: coefficient of s^(k-1) for support point i
end

# Least-squares fit at Chebyshev points; the monomial basis is well conditioned on [−1, 1].
function PolyKernel(φ, W, P; T=Float64)
  ns = 4P
  s = [cospi((2k - 1) / (2ns)) for k in 1:ns]
  V = [sk^k for sk in s, k in 0:P]
  C = zeros(P + 1, W)
  for i in 0:W-1
    C[:, i+1] = V \ [φ(1 + i - W / 2 - (sk + 1) / 2) for sk in s]
  end
  PolyKernel{W,P + 1,T}(ntuple(k -> SVector{W,T}(C[k, :]), P + 1))
end

# The window is symmetric: piece W−1−i at s equals piece i at −s. With wᵢ(s) = Eᵢ(s²) + s·Oᵢ(s²)
# one Horner chain in s² over the first ⌈W/2⌉ pieces gives all weights (the even/odd evaluation of
# ducc); for odd W the middle piece maps onto itself and is kept once.
struct SymKernel{W,H,NE,NO,T}
  even::NTuple{NE,SVector{H,T}}
  odd::NTuple{NO,SVector{H,T}}
end
function SymKernel(K::PolyKernel{W,P1,T}) where {W,P1,T}
  H = cld(W, 2)
  even = Tuple(SVector{H,T}(K.coeffs[k][1:H]) for k in 1:2:P1)
  odd = Tuple(SVector{H,T}(K.coeffs[k][1:H]) for k in 2:2:P1)
  SymKernel{W,H,length(even),length(odd),T}(even, odd)
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

function gausslegendre(n)
  J = SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:n-1])
  x, V = eigen(J)
  x, 2 .* V[1, :] .^ 2
end

# φ̂(ξ) = ∫ φ(z) cos(2πξz) dz for ξ = k/Ñ, k = −N÷2 … N−N÷2−1. φ̂ is even and smooth on [0, 1/2], so
# it is computed by Gauss–Legendre quadrature at nc Chebyshev points and interpolated (Clenshaw)
# at the N modes; quadrature per mode costs O(N·nq) (130 ms for N = 2^18). 32 points reach the
# rounding floor of the quadrature itself (≈ 1e-13 relative).
function kernel_ft(φ, W, N, Ñ; nq=2W + 20, nc=32, nthreads=Threads.nthreads())
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
function window_quadrature(φ, W; nq=2W + 20)
  x, w = gausslegendre(nq)
  z = x .* (W / 2); wz = w .* (W / 2) .* φ.(z)
  ξ -> sum(wz[q] * cospi(2ξ * z[q]) for q in eachindex(z))
end

# φ̂ on [0, ξmax] as a Chebyshev interpolant of degree nc − 1 of the quadrature
struct WindowFT
  c::Vector{Float64}
  ξmax::Float64
end
function WindowFT(φ, W, ξmax; nq=2W + 20, nc=32)
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
