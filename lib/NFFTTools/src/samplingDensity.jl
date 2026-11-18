"""
    weights = sdc(p::AbstractNFFTPlan; method = :pipemenon, iters)

Sampling density compensation weights for the nodes of the NFFT plan `p`, scaled such that
``A^{\\mathsf{H}} \\operatorname{diag}(w) A \\bm{1} ≈ \\bm{1}``. For nodes on the equidistant grid
of an array of size `N` they are ≈ `1/prod(N)`.

`method = :pipemenon` (default): `iters` (default 20) iterations of the method of Pipe and Menon
(Magn Reson Med 41(1):179–186, 1999,
doi:10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V). The iteration uses the
resampling stage of the plan, i.e. the window function on the oversampled grid, as its kernel.
The weights therefore depend on the window of the plan rather than on its accuracy: narrow
windows give better weights than wide ones. With NFFT.jl, `plan_nfft(k, N; W = 4, σ = 2.0)`
works well (MRI test data: 0.11 relative deviation of a gridding reconstruction from the
scanner's, 0.21 with the window chosen for `reltol = 1e-3`). The weights are positive. `p` must
implement the stage interface of AbstractNFFTs (`oversampled_size`, `resampling!`,
`resampling_adjoint!`).

`method = :frobenius`: the weights that minimise
``\\|A^{\\mathsf{H}} \\operatorname{diag}(w) A - I\\|_F`` (Sedarat and Nishimura, IEEE Trans Med
Imaging 19(4):306–317, 2000), the best a diagonal weighting can do. With at least as many nodes
as the doubled array has modes (`prod(2 .* N)`), spread without clusters, they reconstruct every
array exactly. They solve ``S w = |I_N| \\bm{1}`` with ``S_{jh} = |(A A^{\\mathsf{H}})_{jh}|^2``:
the entries of ``A A^{\\mathsf{H}}`` are the Dirichlet kernel at ``x_j - x_h``, their squares the
Fejér kernel, a trigonometric polynomial on the doubled array with coefficients
``c_l = \\prod_d (N_d - |l_d|)``. Hence ``S = A_{2N} \\operatorname{diag}(c) A_{2N}^{\\mathsf{H}}``
costs one NFFT and one adjoint NFFT of size `2 .* N`, and `iters` (default 50) steps of conjugate
gradients minimise the Frobenius error. The weights can be negative. The plan of size `2 .* N` is
made with the active backend.

`p` must be a plan without batch dimensions.
"""
function sdc(p::AbstractNFFTPlan{T,D,1}; method::Symbol = :pipemenon,
             iters::Int = method === :frobenius ? 50 : 20) where {T<:Real,D}
  # in the storage of the plan (e.g. on the GPU), like its nodes
  weights = similar(nodes(p), T, only(size_out(p)))
  method === :pipemenon && return sdc!(weights, p; iters)
  method === :frobenius && return frobenius_weights!(weights, p; iters)
  throw(ArgumentError("method must be :pipemenon or :frobenius, got :$method"))
end

# Conjugate gradients for S w = |I_N| 1, S = A_{2N} diag(c) A_{2N}ᴴ (see `sdc`). The objective
# wᵀSw − 2|I_N| Σw + |I_N| is the squared Frobenius error ‖Aᴴ diag(w) A − I‖²_F.
function frobenius_weights!(w::AbstractVector{T}, p::AbstractNFFTPlan{T,D,1}; iters::Int) where {T<:Real,D}
  N = size_in(p)
  p2 = plan_nfft(nodes(p), 2 .* N; reltol = max(1e-8, 10 * eps(T)), eltype = T)
  # the doubled array holds l = −N_d … N_d − 1; c vanishes at l = −N_d
  c = similar(w, T, 2 .* N)
  copyto!(c, [prod(map((i, n) -> max(n - abs(i - 1 - n), 0), Tuple(I), N)) for I in CartesianIndices(2 .* N)])
  g = similar(w, Complex{T}, 2 .* N)
  v = similar(w, Complex{T}, length(w))
  function S!(out, x)
    v .= x
    mul!(g, adjoint(p2), v)
    g .*= c
    mul!(v, p2, g)
    out .= real.(v)
  end
  fill!(w, zero(T))
  r = fill!(similar(w), T(prod(N)))
  d = copy(r); Sd = similar(w)
  rr = sum(abs2, r)
  for _ in 1:iters
    S!(Sd, d)
    α = rr / sum(d .* Sd)
    w .+= α .* d
    r .-= α .* Sd
    rn = sum(abs2, r)
    d .= r .+ (rn / rr) .* d
    rr = rn
  end
  return w
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
