# Contract tests for any implementation of the AbstractNFFTs interface: they only use the generic
# behaviour of a plan (mul!, *, adjoint, size_in/size_out, nodes!, dims) and compare against an exact
# NDFT. `make_plan(k, N; reltol, dims)` builds the plan under test.
#
# Conventions checked: nodes k ∈ [-1/2, 1/2)^d, equidistant indices l = -N÷2 … N-N÷2-1 (in increasing
# order along every dimension), direct NFFT  fHat_j = Σ_l f_l exp(-2πi l·k_j),
# adjoint NFFT  f_l = Σ_j fHat_j exp(+2πi l·k_j).

using Test, LinearAlgebra, Random

# exp(2πi·s·l·k) with the phase l·k reduced modulo 1 in extended precision, so that the reference
# stays accurate for large l (the error of l·k in Float64 would otherwise be ≈ l·eps).
function cis2pi_exact(s, l::Integer, k::Float64)
  p = l * k
  e = fma(Float64(l), k, -p)
  cispi(2s * ((p - round(p)) + e))
end

freqs(N) = -(N ÷ 2):(N - N ÷ 2 - 1)

# Direct NDFT of f (size N) at nodes k (d × J), in Float64.
function ndft_ref(k::AbstractMatrix, f::AbstractArray)
  N = size(f); D = length(N)
  kd = Float64.(k)
  out = zeros(ComplexF64, size(k, 2))
  for j in axes(k, 2)
    s = zero(ComplexF64)
    for I in CartesianIndices(N)
      c = ComplexF64(f[I])
      for d in 1:D
        c *= cis2pi_exact(-1, freqs(N[d])[I[d]], kd[d, j])
      end
      s += c
    end
    out[j] = s
  end
  out
end

# Adjoint NDFT of fHat (length J) onto an array of size N, in Float64.
function ndft_adjoint_ref(k::AbstractMatrix, N::NTuple{D,Int}, fHat::AbstractVector) where {D}
  kd = Float64.(k)
  out = zeros(ComplexF64, N)
  for I in CartesianIndices(N)
    s = zero(ComplexF64)
    for j in axes(k, 2)
      c = ComplexF64(fHat[j])
      for d in 1:D
        c *= cis2pi_exact(1, freqs(N[d])[I[d]], kd[d, j])
      end
      s += c
    end
    out[I] = s
  end
  out
end

relerr(x, y) = norm(x - y) / norm(y)

# Relative error the contract allows for a requested `reltol`: a factor `slack` above it, but not
# below what the precision allows: rounding grows like log2 of the grid size (at least 100 eps(T)
# for tiny transforms), and node
# coordinates given in Float64 limit the phase to ≈ eps(Float64)·max(N) (this floor is the same for
# every implementation; data precision T does not change it if positions are reduced in Float64).
precision_floor(T, N) = 20 * eps(T) * max(log2(prod(N)), 5) + 2 * eps(Float64) * maximum(N)
allowed(reltol, T, N; slack=10) = max(slack * reltol, precision_floor(T, N))

# A plan for reltol, or `nothing` where the implementation rejects a reltol below the precision
# floor with an ArgumentError (rather than missing it); above the floor it must not reject.
function plan_or_reject(make_plan, k, N, tol)
  try
    make_plan(k, N; reltol=tol)
  catch e
    (e isa ArgumentError && tol < precision_floor(eltype(k), N)) || rethrow()
    nothing
  end
end

random_nodes(rng, T, D, J) = T.(rand(rng, D, J) .- 0.5)

# Nodes on and next to the places where implementations wrap or bin: the period boundary -1/2,
# zero (a node just below 0 lies at the end of the grid after wrapping), and around grid points.
function edge_nodes(T, D, N)
  v = T[-0.5, nextfloat(T(-0.5)), prevfloat(T(0.5)), zero(T), prevfloat(zero(T)), nextfloat(zero(T)),
        T(0.25), prevfloat(T(0.25)), T(-0.25), nextfloat(T(-0.25)), T(1) / 3, -T(1) / 7]
  # all combinations in up to 2 dims, the diagonal beyond
  if D == 1
    reshape(v, 1, :)
  elseif D == 2
    reduce(hcat, [T[a, b] for a in v for b in v])
  else
    reduce(hcat, [fill(a, D) for a in v])
  end
end

"""
    contract_tests(make_plan; T=Float64, tols, sizes, rng, slack, supports_dims=true)

Run the contract tests for plans built by `make_plan(k, N; reltol, dims)`. An implementation without
batch dimensions passes `supports_dims=false` to skip their test.
"""
function contract_tests(make_plan; T=Float64, tols=(T == Float64 ? (1e-4, 1e-8, 1e-12) : (1e-3, 1e-5)),
                        sizes=((64,), (31, 32), (12, 13, 10)), rng=Xoshiro(1234), slack=10, supports_dims=true)
  C = Complex{T}

  @testset "accuracy against NDFT ($(T), N = $N, reltol = $tol)" for N in sizes, tol in tols
    D = length(N); J = prod(N)
    k = random_nodes(rng, T, D, J)
    p = plan_or_reject(make_plan, k, N, tol)
    if p !== nothing
      f = randn(rng, C, N); fHat = randn(rng, C, J)
      @test relerr(p * f, ndft_ref(k, f)) <= allowed(tol, T, N; slack)
      @test relerr(adjoint(p) * fHat, ndft_adjoint_ref(k, N, fHat)) <= allowed(tol, T, N; slack)
    end
  end

  @testset "large 1D problem, sampled reference ($(T), reltol = $tol)" for tol in tols
    # Large grids expose coordinate precision: positions x·Ñ computed in T = Float32 are off by
    # ≈ eps(Float32)·Ñ grid points (1e-2 relative error at Ñ ≈ 4·10^5).
    N = (2^17,); J = 2^17; ns = 40
    k = random_nodes(rng, T, 1, J)
    p = plan_or_reject(make_plan, k, N, tol)
    if p !== nothing
      f = randn(rng, C, N); fHat = randn(rng, C, J)
      js = rand(rng, 1:J, ns); ls = rand(rng, 1:N[1], ns)
      g = p * f; h = adjoint(p) * fHat
      @test relerr(g[js], ndft_ref(k[:, js], f)) <= allowed(tol, T, N; slack)
      href = [sum(ComplexF64(fHat[j]) * cis2pi_exact(1, freqs(N[1])[l], Float64(k[1, j])) for j in 1:J) for l in ls]
      @test relerr(h[ls], href) <= allowed(tol, T, N; slack)
    end
  end

  @testset "adjointness ($(T), N = $N)" for N in sizes
    D = length(N); J = prod(N)
    k = random_nodes(rng, T, D, J)
    p = make_plan(k, N; reltol=first(tols))
    x = randn(rng, C, N); y = randn(rng, C, J)
    lhs = dot(p * x, y); rhs = dot(x, adjoint(p) * y)
    @test abs(lhs - rhs) <= 1000 * eps(T) * norm(p * x) * norm(y)
  end

  @testset "edge nodes ($(T), D = $(length(N)))" for N in sizes
    D = length(N)
    k = edge_nodes(T, D, N)
    tol = tols[min(2, length(tols))]
    p = make_plan(k, N; reltol=tol)
    f = randn(rng, C, N); fHat = randn(rng, C, size(k, 2))
    @test relerr(p * f, ndft_ref(k, f)) <= allowed(tol, T, N; slack)
    @test relerr(adjoint(p) * fHat, ndft_adjoint_ref(k, N, fHat)) <= allowed(tol, T, N; slack)
  end

  @testset "sizes, eltype, mul! ($(T), N = $N)" for N in sizes
    D = length(N); J = 37
    k = random_nodes(rng, T, D, J)
    p = make_plan(k, N; reltol=first(tols))
    @test size_in(p) == N
    @test size_out(p) == (J,)
    @test size(p) == (J, prod(N))
    @test eltype(p) == C
    f = randn(rng, C, N); fHat = randn(rng, C, J)
    g = similar(fHat); @test @inferred(mul!(g, p, f)) === g; @test g ≈ @inferred(p * f)
    h = similar(f); @test @inferred(mul!(h, adjoint(p), fHat)) === h; @test h ≈ @inferred(adjoint(p) * fHat)
  end

  @testset "nodes! ($(T), N = $N)" for N in sizes
    D = length(N); J = 50
    k1 = random_nodes(rng, T, D, J); k2 = random_nodes(rng, T, D, J)
    p = make_plan(k1, N; reltol=first(tols))
    nodes!(p, k2)
    q = make_plan(k2, N; reltol=first(tols))
    f = randn(rng, C, N); fHat = randn(rng, C, J)
    @test p * f ≈ q * f
    @test adjoint(p) * fHat ≈ adjoint(q) * fHat
  end

  @testset "dims: batch dimensions pass through ($(T))" begin
    # NFFT along dims 2:3 of a (3, 10, 11, 2) array: every slice is a 2D NFFT of its own
    N = (3, 10, 11, 2); dims = 2:3; J = 40
    tol = tols[min(2, length(tols))]
    k = random_nodes(rng, T, length(dims), J)
    if !supports_dims
      @test_skip "dims = $dims not supported"
    else
      p = make_plan(k, N; reltol=tol, dims)
      @test size_in(p) == N
      @test size_out(p) == (N[1], J, N[4])
      f = randn(rng, C, N); fHat = randn(rng, C, N[1], J, N[4])
      g = @inferred(p * f); h = @inferred(adjoint(p) * fHat)
      for a in 1:N[1], b in 1:N[4]
        @test relerr(g[a, :, b], ndft_ref(k, f[a, :, :, b])) <= allowed(tol, T, N[dims]; slack)
        @test relerr(h[a, :, :, b], ndft_adjoint_ref(k, N[dims], fHat[a, :, b])) <= allowed(tol, T, N[dims]; slack)
      end
    end
  end
end
