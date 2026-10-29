# NFFT.jl on an Apple GPU (Metal, Float32): plans with `storage = MtlArray`, data on the device.
import NFFT, AbstractNFFTs, Metal
using LinearAlgebra: mul!

supports(task) = task["transform"] != "nfft" ? "only the NFFT on Metal" :
                 task["T"] != "Float32" ? "Metal has no Float64" : nothing

function plan(prob, tol)
  D = length(prob.N); L = size(prob.y, 2)
  N = L == 1 ? prob.N : (prob.N..., L)
  AbstractNFFTs.plan_nfft(NFFT.backend(), prob.k, N; reltol=tol, dims=1:D, storage=Metal.MtlArray)
end
trafo!(p, out, x) = mul!(out, p, x)
adjoint!(p, out, y) = mul!(out, adjoint(p), y)
upload(x) = Metal.MtlArray(x)
tohost(x) = Array(x)
sync() = Metal.synchronize()

function params(p)
  q = NFFT.parameters(p)
  Dict{String,Any}("W" => q.W, "sigma" => q.σ, "window" => string(nameof(typeof(q.window))))
end

libinfo() = Dict{String,Any}("version" => string(pkgversion(NFFT), ", Metal ", pkgversion(Metal)),
                             "library" => dirname(dirname(pathof(NFFT))))
