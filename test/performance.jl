using NFFT

### performance test ###
function nfft_performance()
  let m = 3, sigma = 2.0
    @info "NFFT Performance Test 1D"
    let N = 2^19, M = N, x = rand(M) .- 0.5, fHat = rand(M)*1im
      @info "* initialization"
      @time p = NFFTPlan(x,N,m,sigma)

      @info "* adjoint"
      @time fApprox = nfft_adjoint(p, fHat, true)

      @info "* trafo"
      @time nfft(p, fApprox, true)
    end

    @info "NFFT Performance Test 2D"
    let N = 1024, M = N*N, x2 = rand(2,M) .- 0.5, fHat = rand(M)*1im
      @info "* initialization"
      @time p = NFFTPlan(x2,(N,N),m,sigma)

      @info "* adjoint"
      @time fApprox = nfft_adjoint(p, fHat, true)

      @info "* trafo"
      @time nfft(p, fApprox, true)
    end

    @info "NFFT Performance Test 3D"
    let N = 32, M = N*N*N, x3 = rand(3,M) .- 0.5, fHat = rand(M)*1im
      @info "* initialization"
      @time p = NFFTPlan(x3,(N,N,N),m,sigma)

      @info "* adjoint"
      @time fApprox = nfft_adjoint(p, fHat, true)

      @info "* trafo"
      @time nfft(p, fApprox, true)
    end
  end
  return nothing
end

nfft_performance()
