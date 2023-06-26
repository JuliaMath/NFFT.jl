@testset "Chainrules" begin

eps = 1e-4

for N in [17,33,67]
  for T in [Float32, Float64]
    M = rand(N÷2:2*N)
    # signal
    fHat = rand(complex(T), N) # random signal
    f = rand(complex(T),M) # random Fourier space signal
    # Fourier matrix
    k = rand(real(T),M).-real(T)(0.5)
    x = range(-N÷2, step=1, length=N)
    nfftMat = [exp(-real(T)(2π)*im*k[i]*x[j]) for i=1:M, j=1:N]
    # NFFT-plan
    p = plan_nfft(k,(N,))
    p_adj = adjoint(p)
    
    # test gradients of NFFT
    y, g = Zygote.withgradient(v->sum(abs.(nfftMat*v).^2), fHat)
    y2, g2 = Zygote.withgradient(v->sum(abs.(nfft(k,v)).^2), fHat)
    y3, g3 = Zygote.withgradient(v->sum(abs.(p*v).^2), fHat)
    @test (y-y2)/y < eps
    @test (y-y3)/y < eps
    @test norm(g[1]-g2[1])/norm(g[1]) < eps
    @test norm(g[1]-g3[1])/norm(g[1]) < eps

    # test gradients of adjoint NFFT
    y, g = Zygote.withgradient(v->sum(abs.(adjoint(nfftMat)*v).^2), f)
    y2, g2 = Zygote.withgradient(v->sum(abs.(nfft_adjoint(k,N,v)).^2), f)
    y3, g3 = Zygote.withgradient(v->sum(abs.(p_adj*v).^2), f)
    @test (y-y2)/y < eps
    @test (y-y3)/y < eps
    @test norm(g[1]-g2[1])/norm(g[1]) < eps
    @test norm(g[1]-g3[1])/norm(g[1]) < eps
  end

end


end