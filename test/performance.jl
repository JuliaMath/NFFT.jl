### performance test ###

function nfft_performance()

    m = 4
    sigma = 2.0

    # 1D

    N = 2^19
    M = N

    x = rand(M) .- 0.5
    fHat = rand(M)*1im

    println("NFFT Performance Test 1D")

    tic()
    p = NFFTPlan(x,N,m,sigma)
    println("initialization")
    toc()

    tic()
    fApprox = nfft_adjoint(p,fHat)
    println("adjoint")
    toc()

    tic()
    fHat2 = nfft(p, fApprox);
    println("trafo")
    toc()

    N = 1024
    M = N*N

    x2 = rand(2,M) .- 0.5
    fHat = rand(M)*1im

    println("NFFT Performance Test 2D")

    tic()
    p = NFFTPlan(x2,(N,N),m,sigma)
    println("initialization")
    toc()

    tic()
    fApprox = nfft_adjoint(p,fHat)
    println("adjoint")
    toc()

    tic()
    fHat2 = nfft(p, fApprox);
    println("trafo")
    toc()

    N = 32
    M = N*N*N

    x3 = rand(3,M) .- 0.5
    fHat = rand(M)*1im

    println("NFFT Performance Test 3D")

    tic()
    p = NFFTPlan(x3,(N,N,N),m,sigma)
    println("initialization")
    toc()

    tic()
    fApprox = nfft_adjoint(p,fHat)
    println("adjoint")
    toc()

    tic()
    fHat2 = nfft(p, fApprox);
    println("trafo")
    toc()

end
