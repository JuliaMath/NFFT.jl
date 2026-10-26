# Benchmark problems: nodes, data and an exact reference on sampled outputs, generated
# deterministically from the task, so that every library (each in its own process) sees the same
# problem.
#
# Conventions of NFFT.jl for all libraries: nodes k ∈ [-1/2, 1/2)^D (NFCT/NFST: [0, 1/2]^D), modes
#   nfft:  l_d = -N_d÷2 … N_d-N_d÷2-1,          g_j = Σ_l f_l e^{-2πi l·k_j}, adjoint e^{+2πi l·k_j}
#   rnfft: l_1 = 0 … N_1÷2, others as nfft,     g_j = Re Σ_l h_l e^{-2πi l·k_j}, adjoint e^{+2πi l·k_j}
#   nfct:  l_d = 0 … N_d-1,                     g_j = Σ_l f_l Π_d cos(2π l_d k_dj), transposed alike
#   nfst:  l_d = 1 … N_d,                       g_j = Σ_l f_l Π_d sin(2π l_d k_dj), transposed alike
#   nnfft: prod(N) sources y_l uniform in Π_d [-N_d/2, N_d/2), g_j = Σ_l f_l e^{-2πi k_j·y_l}, adjoint
#          e^{+2πi k_j·y_l}; the same space-frequency extent as an NFFT of size N
# A batch of L transforms has the data as (input..., L) and (J, L); the reference covers slice 1.

using Random, LinearAlgebra, Serialization

struct Problem
  task::Dict{String,Any}
  T::Type
  N::Tuple
  k::Matrix                  # D × J, element type T
  src::Matrix                # nnfft: D × prod(N) sources, element type T
  x::Array                   # input of the direct transform (input..., L)
  y::Array                   # input of the adjoint (J, L)
  ref_nodes::Vector{Int}     # sampled nodes j
  ref_trafo::Vector{ComplexF64}
  ref_modes::Vector{CartesianIndex}
  ref_adjoint::Vector{ComplexF64}
end

const NSAMPLES = 64

function nodes(dist, D, J, rng)
  if dist == "uniform" || (dist == "radial" && D == 1)
    rand(rng, D, J) .- 0.5
  elseif dist == "radial"                    # spokes through the centre (MRI)
    ns = D == 2 ? round(Int, sqrt(J)) : round(Int, J^(1 / 3))^2
    nr = cld(J, ns)
    r = range(-0.5, 0.5; length=nr + 1)[1:nr]
    dirs = [normalize(randn(rng, D)) for _ in 1:ns]
    D == 2 && (dirs = [[cospi(s / ns), sinpi(s / ns)] for s in 0:ns-1])
    k = reduce(hcat, [ρ .* u for u in dirs for ρ in r])[:, 1:J]
    clamp.(k, -0.5, prevfloat(0.5))
  elseif dist == "clustered"                 # dense centre, sparse edge
    clamp.(0.15 .* randn(rng, D, J), -0.5, prevfloat(0.5))
  else
    error("unknown node distribution $dist")
  end
end

input_size(kind, N) = kind == "rnfft" ? (N[1] ÷ 2 + 1, Base.tail(N)...) : kind == "nnfft" ? (prod(N),) : N

# mode of array index i along dimension d
function mode(kind, d, i, Nd)
  kind == "nfct" ? i - 1 : kind == "nfst" ? i : (kind == "rnfft" && d == 1) ? i - 1 : i - 1 - Nd ÷ 2
end

# basis function of the direct transform; phases reduced modulo 1 in Float64 (|l| up to 2^22)
@inline function frac(l, x)
  p = l * x
  (p - round(p)) + fma(Float64(l), x, -p)
end
basis(kind, l, x) = kind == "nfct" ? complex(cospi(2frac(l, x))) :
                    kind == "nfst" ? complex(sinpi(2frac(l, x))) : cispi(-2frac(l, x))

function Problem(task::Dict{String,Any})
  T = task["T"] == "Float32" ? Float32 : Float64
  kind = task["transform"]
  N = Tuple(Int.(task["N"]))
  D = length(N)
  J = Int(task["J"]); L = Int(get(task, "L", 1))
  rng = Xoshiro(Int(get(task, "seed", 1234)) + 7919 * D + J)
  k = nodes(task["nodes"], D, J, rng)
  kind in ("nfct", "nfst") && (k = clamp.(abs.(k), 0.0, 0.5))
  k = T.(k)
  src = kind == "nnfft" ? T.((rand(rng, D, prod(N)) .- 0.5) .* collect(N)) : zeros(T, D, 0)
  Nin = input_size(kind, N)
  realin = kind in ("nfct", "nfst")
  realout = kind in ("rnfft", "nfct", "nfst")
  x = realin ? randn(rng, T, Nin..., L) : randn(rng, Complex{T}, Nin..., L)
  y = realout ? randn(rng, T, J, L) : randn(rng, Complex{T}, J, L)
  ref_nodes = rand(rng, 1:J, NSAMPLES)
  ref_modes = [CartesianIndex(ntuple(d -> rand(rng, 1:Nin[d]), length(Nin))) for _ in 1:NSAMPLES]
  # the exact sums take up to seconds (3D: 64·J·D phases); the problem is deterministic, so they
  # are cached on disk
  file = joinpath(REFCACHE, string(hash((kind, N, J, L, task["nodes"], task["T"], get(task, "seed", 1234), NSAMPLES)); base=16) * ".jls")
  ref_trafo, ref_adjoint = if isfile(file)
    deserialize(file)
  else
    kd = Float64.(k)
    x1 = ComplexF64.(selectdim(x, D + 1, 1)); y1 = ComplexF64.(y[:, 1])
    r = kind == "nnfft" ?
      ([ref_nnfft(kd[:, j:j], Float64.(src), x1, -1) for j in ref_nodes],
       [ref_nnfft(Float64.(src[:, I[1]:I[1]]), kd, y1, 1) for I in ref_modes]) :
      ([ref_direct(kind, kd[:, j], N, Nin, x1) for j in ref_nodes],
       [ref_adj(kind, kd, N, I, y1) for I in ref_modes])
    mkpath(REFCACHE); serialize(file, r)
    r
  end
  Problem(task, T, N, k, src, x, y, ref_nodes, ref_trafo, ref_modes, ref_adjoint)
end

const REFCACHE = joinpath(dirname(@__DIR__), ".cache", "references")

# direct transform at one node, contracted one dimension at a time
function ref_direct(kind, kj, N, Nin, x)
  r = x
  for d in eachindex(N)
    e = [basis(kind, mode(kind, d, i, N[d]), kj[d]) for i in 1:Nin[d]]
    r = reshape(transpose(e) * reshape(r, Nin[d], :), Nin[d+1:end])
  end
  kind == "rnfft" ? complex(real(r[])) : r[]
end

# NNFFT at one point a (D × 1) from the points b with values c: Σ_l c_l e^{sign 2πi a·b_l}
function ref_nnfft(a, b, c, sign)
  s = zero(ComplexF64)
  @inbounds for l in axes(b, 2)
    p = 0.0
    for d in axes(b, 1)
      p += a[d] * b[d, l]
    end
    s += c[l] * cispi(sign * 2 * (p - round(p)))
  end
  s
end

# adjoint (transposed) transform at one mode
function ref_adj(kind, k, N, I, y)
  s = zero(ComplexF64)
  @inbounds for j in axes(k, 2)
    c = y[j]
    for d in eachindex(N)
      b = basis(kind, mode(kind, d, I[d], N[d]), k[d, j])
      c *= kind in ("nfct", "nfst") ? b : conj(b)
    end
    s += c
  end
  s
end

relerr(a, b) = norm(a - b) / norm(b)
err_trafo(p::Problem, out) = relerr(ComplexF64[out[j, 1] for j in p.ref_nodes], p.ref_trafo)
function err_adjoint(p::Problem, out)
  D = length(p.N)
  relerr(ComplexF64[out[Tuple(I)..., 1] for I in p.ref_modes], p.ref_adjoint)
end
