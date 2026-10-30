# The C++ port of NFFT.jl's CPU NFFT (workspace `port-cpp/`, outside this repository), with the
# parameters of the NFFT.jl plan the NFFT adapter builds for the task: NFFT.jl's own choice, or a
# rule's window, W and grid (one plan per direction where they differ). NFFT, D = 1–3,
# Float64/Float32, no batches. The library is `ENV["NFFTPORT_LIB"]` or the build in `port-cpp/`
# (`ENV["NFFTPORT_DIR"]`); the environment is NFFT.jl's.
const PORT = get(ENV, "NFFTPORT_DIR", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "port-cpp")))
include(joinpath(PORT, "julia", "PortCpp.jl"))

# the NFFT adapter, for its plans
module JL
using ..Rules
import ..Unsupported
include(joinpath(@__DIR__, "nfft.jl"))
end
import .JL: NFFT

supports(task) = task["transform"] != "nfft" ? "the port has only the NFFT" :
                 get(task, "L", 1) != 1 ? "the port has no batches" :
                 length(task["N"]) > 3 ? "the port has D ≤ 3" : nothing

struct PortPlan{P}
  jl::P      # the NFFT.jl plan (or RulePlan) whose parameters the port takes
  trafo::PortCpp.Plan
  adjoint::PortCpp.Plan
end

function plan(prob, tol)
  p = JL.plan(prob, tol)
  p isa JL.RulePlan || return (q = PortCpp.Plan(p); PortPlan(p, q, q))
  q = PortCpp.Plan(p.trafo)
  PortPlan(p, q, p.adjoint === p.trafo ? q : PortCpp.Plan(p.adjoint))
end
release!(p::PortPlan) = (PortCpp.destroy!(p.trafo); PortCpp.destroy!(p.adjoint); nothing)

trafo!(p::PortPlan, out, x) = PortCpp.trafo!(out, p.trafo, x)
adjoint!(p::PortPlan, out, y) = PortCpp.adjoint!(out, p.adjoint, y)

params(p::PortPlan) = JL.params(p.jl)
fft_flag(task) = "ESTIMATE"

libinfo() = Dict{String,Any}("version" => "$(PortCpp.build_info()); julia $(VERSION) LLVM $(Base.libllvm_version)",
                             "library" => PortCpp.LIB)
