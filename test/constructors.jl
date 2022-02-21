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

@show p

## test range error


x = [-0.6  0.9; 0.5  -0.5]
@test_throws ArgumentError NFFTPlan(x, (2,2))
x = [-0.3  0.3; 0.3  NaN]
@test_throws Exception NFFTPlan(x, (2,2))
# The previous test throws an ArgumentError in the single-threaded case
# and an TaskFailedException in the multi-threaded case. Needs some rethrow

## test nodes!(p, tr)
Nx = 32
trj1 = rand(2, 1000) .- 0.5
trj2 = rand(2, 1000) .- 0.5

p1 = NFFT.NFFTPlan(trj1, (Nx, Nx))
p2 = NFFT.NFFTPlan(trj2, (Nx, Nx))
NFFT.nodes!(p2, trj1)

@test p1.N == p2.N
@test p1.NOut == p2.NOut
@test p1.M == p2.M
@test all(p1.x .== p2.x)
@test p1.n == p2.n
@test p1.dims == p2.dims
for n in fieldnames(typeof(p1.params))
  @test getfield(p1.params,n) == getfield(p2.params,n)
end
@test p1.windowLinInterp == p2.windowLinInterp
@test p1.windowPolyInterp == p2.windowPolyInterp
@test p1.windowHatInvLUT == p2.windowHatInvLUT
@test p1.B == p2.B

for n in fieldnames(typeof(p1.forwardFFT))
    println(n)
    if n != :pinv
        @test getfield(p1.forwardFFT,n) == getfield(p1.forwardFFT,n)
        @test getfield(p1.backwardFFT,n) == getfield(p1.backwardFFT,n)
    end
end

end