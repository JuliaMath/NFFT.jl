# NFFT.jl 0.14, the last release before 1.0: NFFT only.
import NFFT, FFTW
using LinearAlgebra: mul!

supports(task) = task["transform"] == "nfft" ? nothing : "NFFT.jl 0.14 has no fast $(task["transform"])"

function plan(prob, tol)
  D = length(prob.N); L = size(prob.y, 2)
  N = L == 1 ? prob.N : (prob.N..., L)
  NFFT.plan_nfft(prob.k, N; reltol=tol, dims=1:D, fftflags=FFTW.ESTIMATE)
end
trafo!(p, out, x) = mul!(out, p, x)
adjoint!(p, out, y) = mul!(out, adjoint(p), y)
params(p) = Dict{String,Any}("W" => 2p.params.m, "sigma" => p.params.σ)

libinfo() = Dict{String,Any}("version" => string(pkgversion(NFFT)), "library" => dirname(dirname(pathof(NFFT))))
