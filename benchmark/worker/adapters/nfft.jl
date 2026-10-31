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
  haskey(prob.task, "rule") && return rule_plan(f, prob, N, tol)
  haskey(prob.task, "sigma_req") || return f(NFFT.backend(), prob.k, N; reltol=tol, dims=1:D)
  e = smallest_width(prob.task["sigma_req"], tol, prob.N, eltype(prob.k))
  f(NFFT.backend(), prob.k, N; reltol=tol, dims=1:D, window=e.window, W=e.W, σ=e.σ)
end

# The parameters another library would choose (src/rules.jl): its window as a CustomWindow, its W, and
# its grid, exactly through the option gridsize. Revisions without gridsize get the grid through σ (a
# grid with factors 7 or 11, as ducc's, then rounds up to the next one from {2, 3, 5}, and only one σ
# for all dimensions). One plan per direction where the rule differs between trafo (type 2) and
# adjoint (type 1).
struct RulePlan{P,Q}
  trafo::P
  adjoint::Q
  choice::Any
  N::Vector{Int}
end

function rule_plan(f, prob, N, tol)
  c = Rules.choose(prob.task["rule"], eltype(prob.k), prob.N, size(prob.k, 2), tol; threads=Threads.nthreads())
  D = length(prob.N)
  window(ci) = NFFT.CustomWindow(Rules.window(ci))
  function mk(ci)
    has_gridsize() && return f(NFFT.backend(), prob.k, N; reltol=tol, dims=1:D, window=window(ci), W=ci.W,
                               σ=minimum(ci.grid ./ collect(prob.N)), gridsize=Tuple(ci.grid))
    allequal(ci.grid ./ collect(prob.N)) || throw(Unsupported("the grid of the rule is not one σ for all dimensions"))
    f(NFFT.backend(), prob.k, N; reltol=tol, dims=1:D, window=window(ci), W=ci.W, σ=(ci.grid[1] - 0.5) / prob.N[1])
  end
  p = mk(c.trafo)
  RulePlan(p, c.trafo == c.adjoint ? p : mk(c.adjoint), c, collect(prob.N))
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
trafo!(p::RulePlan, out, x) = trafo!(p.trafo, out, x)
adjoint!(p::RulePlan, out, y) = adjoint!(p.adjoint, out, y)

has_gridsize() = isdefined(NFFT, :NFFT_OPTIONS) && :gridsize in NFFT.NFFT_OPTIONS
fft_flag(task) = "ESTIMATE"

# The grid the plan built.
function grid_of(p, N)
  hasproperty(p, :gridsize) && p.gridsize !== nothing && return collect(p.gridsize)
  q = NFFT.parameters(p)
  collect(applicable(NFFT.grid_size, q.σ, Tuple(N), q.W) ? NFFT.grid_size(q.σ, Tuple(N), q.W) :
          NFFT.grid_size(q.σ, Tuple(N)))
end
function params(p::RulePlan)
  N = p.N; q, qa = NFFT.parameters(p.trafo), NFFT.parameters(p.adjoint)
  g, ga = grid_of(p.trafo, N), grid_of(p.adjoint, N)
  Dict{String,Any}("W" => q.W, "sigma" => g[1] / N[1], "window" => p.choice.trafo.window, "grid" => g,
                   "grid_adj" => ga, "rep_W" => q.W, "rep_W_adj" => qa.W, "rep_grid" => g, "rep_grid_adj" => ga)
end

function params(p)
  # NNFFT: the window of the spreading and the size of the inner NFFT
  isdefined(NFFT, :NNFFTPlan) && p isa NFFT.NNFFTPlan &&
    return Dict{String,Any}("W" => p.outer.W, "sigma" => p.outer.σ, "window" => "n = " * join(p.n, "×"))
  isdefined(NFFT, :parameters) || return Dict{String,Any}()
  q = NFFT.parameters(p)
  r = Dict{String,Any}("W" => q.W, "sigma" => q.σ, "window" => string(nameof(typeof(q.window))))
  p isa NFFT.NFFTPlan && hasproperty(p, :dims) && isdefined(NFFT, :grid_size) && (r["grid"] = grid_of(p, p.N[p.dims]))
  r
end

libinfo() = Dict{String,Any}("version" => string(pkgversion(NFFT)), "library" => dirname(dirname(pathof(NFFT))))
