# Method ambiguities, type piracy, stale dependencies and compat entries of the three packages.
using Aqua

@testset "Aqua" begin
  # The resampling methods take one weight vector SVector{W,T} per dimension, NTuple{D,…}: W and T
  # are unbound only for D = 0, which no plan has.
  Aqua.test_all(NFFT; unbound_args=false)
  # ScopedValues is loaded on Julia 1.10 only; later versions have it in Base.
  Aqua.test_all(AbstractNFFTs; stale_deps=(ignore=[:ScopedValues],))
  Aqua.test_all(NFFTTools)
end
