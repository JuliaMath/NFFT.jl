# Method ambiguities, type piracy, stale dependencies and compat entries of the three packages.
using Aqua

@testset "Aqua" begin
  # The resampling methods take one weight vector SVector{W,T} per dimension, NTuple{D,…}: W and T
  # are unbound only for D = 0, which no plan has.
  # LinearAlgebra on Julia 1.10 has `mul!` and `*` for an AbstractMatrix times a triangular or
  # diagonal matrix, which are ambiguous with those of adjoint plans (an Adjoint is an
  # AbstractMatrix); later versions have none of these.
  # CI adds NFFTTools to the project (Julia 1.10 has no [sources]), so that it appears as a
  # dependency that NFFT does not load.
  Aqua.test_all(NFFT; unbound_args=false, ambiguities=VERSION >= v"1.11",
                stale_deps=(ignore=[:NFFTTools],))
  # ScopedValues is loaded on Julia 1.10 only; later versions have it in Base.
  Aqua.test_all(AbstractNFFTs; ambiguities=VERSION >= v"1.11", stale_deps=(ignore=[:ScopedValues],))
  Aqua.test_all(NFFTTools)
end
