"""
    weights = sdc(p::AbstractNFFTPlan; iters = 20)

Sampling density compensation weights for the nodes of the NFFT plan `p`, computed with
`iters` iterations of the method of Pipe and Menon (Magn Reson Med 41(1):179–186, 1999,
doi:10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V). The iteration uses the
resampling stage of the plan, i.e. the window function on the oversampled grid, as its kernel.
The weights therefore depend on the window of the plan rather than on its accuracy: narrow
windows give better weights than wide ones. With NFFT.jl, `plan_nfft(k, N; W = 4, σ = 2.0)`
works well (MRI test data: 0.11 relative deviation of a gridding reconstruction from the
scanner's, 0.21 with the window chosen for `reltol = 1e-3`).

The weights are real, positive and scaled such that ``A^{\\mathsf{H}} \\operatorname{diag}(w) A
\\bm{1} ≈ \\bm{1}``. For nodes on the equidistant grid of an array of size `N` they are
≈ `1/prod(N)`.

`p` must be a plan without batch dimensions that implements the stage interface of
AbstractNFFTs (`oversampled_size`, `resampling!`, `resampling_adjoint!`).
"""
function sdc(p::AbstractNFFTPlan{T,D,1}; iters::Int = 20) where {T<:Real,D}
  # in the storage of the plan (e.g. on the GPU), like its nodes
  weights = similar(nodes(p), T, only(size_out(p)))
  return sdc!(weights, p; iters)
end

"""
    sdc!(weights, p::AbstractNFFTPlan; iters = 20, workg, workf, workv)

In-place version of [`sdc`](@ref). The work arrays (complex, of sizes `oversampled_size(p)`,
`size_out(p)` and `size_in(p)`) can be passed to avoid allocations.
"""
function sdc!(weights::AbstractVector{T}, p::AbstractNFFTPlan{T,D,1}; iters::Int = 20,
              workg::AbstractArray{Complex{T}} = similar(weights, Complex{T}, oversampled_size(p)),
              workf::AbstractVector{Complex{T}} = similar(weights, Complex{T}, only(size_out(p))),
              workv::AbstractArray{Complex{T}} = similar(weights, Complex{T}, size_in(p))) where {T<:Real,D}
  fill!(weights, one(T))
  scale = one(T)
  for i in 1:iters
    # w ← w / (B Bᴴ w), with B Bᴴ normalised by its value in the first iteration
    workf .= weights
    resampling_adjoint!(p, workf, workg)
    i == 1 && (scale = maximum(real, workg))
    resampling!(p, workg, workf)
    any(x -> real(x) <= 0, workf) && throw(ErrorException("non-positive weights in iteration $i"))
    weights .*= scale ./ real.(workf)
  end

  # scale w by the c that minimises ‖1 - c v‖₂ with v = Aᴴ diag(w) A 1
  fill!(workv, one(T))
  mul!(workf, p, workv)
  workf .*= weights
  mul!(workv, adjoint(p), workf)
  weights .*= real(sum(workv)) / sum(abs2, workv)
  return weights
end
