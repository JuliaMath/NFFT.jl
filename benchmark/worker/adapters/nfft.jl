# NFFT.jl 1.x (this repository or a git revision of it): all transforms, batches through `dims`.
import NFFT, AbstractNFFTs
using LinearAlgebra: mul!

const KIND_PLANS = Dict("nfft" => :plan_nfft, "rnfft" => :plan_rnfft, "nfct" => :plan_nfct, "nfst" => :plan_nfst,
                        "nnfft" => :plan_nnfft)

supports(task) = isdefined(AbstractNFFTs, KIND_PLANS[task["transform"]]) ? nothing :
                 "no $(task["transform"]) in this revision"

function plan(prob, tol)
  prob.task["transform"] == "nnfft" && return AbstractNFFTs.plan_nnfft(NFFT.backend(), prob.k, prob.src; reltol=tol)
  D = length(prob.N); L = size(prob.y, 2)
  N = L == 1 ? prob.N : (prob.N..., L)
  f = getfield(AbstractNFFTs, KIND_PLANS[prob.task["transform"]])
  haskey(prob.task, "sigma_req") || return f(NFFT.backend(), prob.k, N; reltol=tol, dims=1:D)
  e = smallest_width(prob.task["sigma_req"], tol, prob.N, eltype(prob.k))
  f(NFFT.backend(), prob.k, N; reltol=tol, dims=1:D, window=e.window, W=e.W, σ=e.σ)
end

# The table entry at the tabulated σ nearest to σ with the smallest W whose predicted error reaches tol
# (either window family; the more accurate one at equal W).
function smallest_width(σ, tol, N, ::Type{T}) where {T}
  isdefined(NFFT, :kernel_candidates) || throw(Unsupported("no kernel_candidates in this revision"))
  cands = NFFT.kernel_candidates(:auto, nothing, σ, T)
  ok = filter(e -> NFFT.predicted_error(e, Tuple(N), T) <= tol, cands)
  isempty(ok) && throw(Unsupported("no W reaches tol at σ = $σ"))
  argmin(e -> (e.W, NFFT.predicted_error(e, Tuple(N), T)), ok)
end
trafo!(p, out, x) = mul!(out, p, x)
adjoint!(p, out, y) = mul!(out, p isa AbstractNFFTs.AbstractRealFTPlan ? transpose(p) : adjoint(p), y)

function params(p)
  # NNFFT: the window of the spreading and the size of the inner NFFT
  isdefined(NFFT, :NNFFTPlan) && p isa NFFT.NNFFTPlan &&
    return Dict{String,Any}("W" => p.outer.W, "sigma" => p.outer.σ, "window" => "n = " * join(p.n, "×"))
  isdefined(NFFT, :parameters) || return Dict{String,Any}()
  q = NFFT.parameters(p)
  Dict{String,Any}("W" => q.W, "sigma" => q.σ, "window" => string(nameof(typeof(q.window))))
end

libinfo() = Dict{String,Any}("version" => string(pkgversion(NFFT)), "library" => dirname(dirname(pathof(NFFT))))
