@testset "Constructors" begin

  @test_throws ArgumentError NFFTPlan(zeros(1,4), (2,2))
  @test_throws ArgumentError NFFTPlan(zeros(2,4), (2,3))

  p = NFFTPlan(zeros(2,4), (2,2))
  pCopy = copy(p)

  for n in fieldnames(typeof(p))
    println(n)
    if n ∉ [:tmpVec, :forwardFFT, :backwardFFT, :blocks, :nodesInBlock]
      @test getfield(p,n) == getfield(pCopy,n)
    end
  end

  pAdjCopy = copy(adjoint(p)) # just ensure that copy does not error

  @show p
  @show adjoint(p)

  ## test range error


  k =[-0.6  0.9; 0.5  -0.5]
  @test_throws ArgumentError NFFTPlan(k, (2,2))
  k =[-0.3  0.3; 0.3  NaN]
  @test_throws Exception NFFTPlan(k, (2,2))
  # The previous test throws an ArgumentError in the single-threaded case
  # and an TaskFailedException in the multi-threaded case. Needs some rethrow

  ## test nodes!(p, tr)
  Nx = 32
  trj1 = rand(2, 1000) .- 0.5
  trj2 = rand(2, 1000) .- 0.5

  p1 = NFFT.NFFTPlan(trj1, (Nx, Nx))
  p2 = NFFT.NFFTPlan(trj2, (Nx, Nx))
  NFFT.nodes!(p2, trj1)


  for n in fieldnames(typeof(p1))
    println(n)
    if n ∉ [:tmpVec, :forwardFFT, :backwardFFT, :blocks, :nodesInBlock, :params]
      @test getfield(p1,n) == getfield(p2,n)
    end
  end

  for n in fieldnames(typeof(p1.params))
    println(n)
    @test getfield(p1.params,n) == getfield(p2.params,n)
  end

  for n in fieldnames(typeof(p1.forwardFFT))
      println(n)
      if n ∉ [:pinv, :plan]
          @test getfield(p1.forwardFFT,n) == getfield(p2.forwardFFT,n)
          @test getfield(p1.backwardFFT,n) == getfield(p2.backwardFFT,n)
      end
  end

end