@testset "Constructors" begin

@test_throws ArgumentError NFFTPlan(zeros(1,4), (2,2))
@test_throws ArgumentError NFFTPlan(zeros(2,4), (2,3))

p = NFFTPlan(zeros(2,4), (2,2))
pCopy = copy(p)

for n in fieldnames(typeof(p))
  println(n)
  if n ∉ [:tmpVec, :forwardFFT, :backwardFFT]
    @test getfield(p,n) == getfield(pCopy,n)
  end
end

@show p

end