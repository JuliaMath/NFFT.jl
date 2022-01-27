using Plots

M = 10;
D = 2;

x = rand(D,M) .- 0.5;
N = ntuple(d-> 32 , D)

padding = ntuple(d-> 2 , D)
blockSize = ntuple(d-> (d==1) ? 16 : 4 , D)
blockSizePadded = ntuple(d-> blockSize[d] + 2*padding[d] , D)

numBlocks =  ntuple(d-> ceil(Int, N[d]/blockSize[d]) , D)
totalBlocks = prod(numBlocks)



nodesInBlock = [ Int[] for l in CartesianIndices(numBlocks) ]

patches = [ zeros(ComplexF64, blockSizePadded) for l in CartesianIndices(numBlocks) ]

for k=1:size(x,2)
  idx = ntuple(d->floor(Int, ((x[d,k]+0.5)*(N[d]-1))÷blockSize[d])+1, D)

  push!(nodesInBlock[idx...], k)
end


scatter(x[1,:], x[2,:])

