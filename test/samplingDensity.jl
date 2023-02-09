@testset "Sampling Density" begin

# create a 10x10 grid of unit spaced sampling points
N = 10
g = (0:(N-1)) ./ N .- 0.5  
x = vec(ones(N) * g')
y = vec(g * ones(N)')
nodes = cat(x',y', dims=1)

for pre in [NFFT.LINEAR, NFFT.FULL, NFFT.TENSOR, NFFT.POLYNOMIAL]
  # approximate the density weights
  p = plan_nfft(nodes, (N,N), m = 5, σ = 2.0, precompute=pre); 
  weights = sdc(p, iters = 10)

  @test all( (≈).(vec(weights), 1/(N*N), rtol=1e-7) )
end

end
