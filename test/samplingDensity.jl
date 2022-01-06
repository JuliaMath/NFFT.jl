@testset "Accuracy" begin

# create a 10x10 grid of unit spaced sampling points
N = 10
g = (0:(N-1)) ./ N .- 0.5  
x = vec(ones(N) * g')
y = vec(g * ones(N)')
nodes = cat(x',y', dims=1)

# approximate the density weights
p = NFFT.plan_nfft(nodes, (N,N), m = 5, σ = 2.0); 
weights = NFFT.sdc(p, iters = 10)

# test if they approximate the true weights (1/(N*N))
@test all( (≈).(vec(weights), 1/(N*N), rtol=1e-7) )

end