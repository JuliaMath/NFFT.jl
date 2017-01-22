export sdc

function sdc{D,T}(p::NFFTPlan{D,0,T}; iters=20)
  # Weights for sample density compensation.
  # Uses method of Pipe & Menon, 1999. Mag Reson Med, 186, 179.
  weights = ones(Complex{T}, p.M)
  weights_tmp = similar(weights)
  # Pre-weighting to correct non-uniform sample density
  for i in 1:iters
    p.tmpVec[:] = 0.0
    convolve_adjoint!(p, weights, p.tmpVec)
    weights_tmp[:] = 0.0
    convolve!(p, p.tmpVec, weights_tmp)
    for j in 1:length(weights)
      weights[j] = weights[j] / (abs(weights_tmp[j]) + eps(T))
    end
  end
  # Post weights to correct image scaling
  # This finds c, where ||u - c*v||_2^2 = 0 and then uses
  # c to scale all weights by a scalar factor.
  u = ones(Complex{T}, p.N)
  f = nfft(p, u)
  f = f .* weights # apply weights from above
  v = nfft_adjoint(p, f)
  c = v[:] \ u[:]  # least squares diff
  abs(weights * c[1]) # [1] needed b/c 'c' is a 1x1 Array
end
