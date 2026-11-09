# NFFT.jl on an Nvidia GPU (CUDA): plans with `storage = CuArray`, data on the device.
import NFFT, AbstractNFFTs, CUDA
using LinearAlgebra: mul!

supports(task) = task["transform"] != "nfft" ? "only the NFFT on CUDA" : nothing

function plan(prob, tol)
  D = length(prob.N); L = size(prob.y, 2)
  N = L == 1 ? prob.N : (prob.N..., L)
  haskey(prob.task, "sigma_req") ||
    return AbstractNFFTs.plan_nfft(NFFT.backend(), prob.k, N; reltol=tol, dims=1:D, storage=CUDA.CuArray)
  e = smallest_width(prob.task["sigma_req"], tol, prob.N, eltype(prob.k))
  AbstractNFFTs.plan_nfft(NFFT.backend(), prob.k, N; reltol=tol, dims=1:D, storage=CUDA.CuArray,
                          window=e.window, W=e.W, σ=e.σ)
end

# as in nfft.jl: the smallest W that reaches tol at the tabulated σ nearest to σ
function smallest_width(σ, tol, N, ::Type{T}) where {T}
  cands = NFFT.kernel_candidates(:auto, nothing, σ, T)
  ok = filter(e -> NFFT.predicted_error(e, Tuple(N), T) <= tol, cands)
  isempty(ok) && throw(Unsupported("no W reaches tol at σ = $σ"))
  argmin(e -> (e.W, NFFT.predicted_error(e, Tuple(N), T)), ok)
end
trafo!(p, out, x) = mul!(out, p, x)
adjoint!(p, out, y) = mul!(out, adjoint(p), y)
upload(x) = CUDA.CuArray(x)
tohost(x) = Array(x)
sync() = CUDA.synchronize()

function params(p)
  q = NFFT.parameters(p)
  Dict{String,Any}("W" => q.W, "sigma" => q.σ, "window" => string(nameof(typeof(q.window))))
end

libinfo() = Dict{String,Any}("version" => string(pkgversion(NFFT), ", CUDA ", pkgversion(CUDA)),
                             "library" => dirname(dirname(pathof(NFFT))))
