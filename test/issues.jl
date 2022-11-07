@testset "Issues Encountered during Development" begin

  # https://github.com/JuliaMath/NFFT.jl/issues/106
  # The issue was an out of bounce in the linear interpolation code.
  T = Float32
  Nz = 120
  img_shape_os = (2Nz,)
  λ = Array{Complex{T}}(undef, img_shape_os)
  
  trj = zeros(T, 1, 2)
  nfftplan = plan_nfft(trj, img_shape_os; precompute = LINEAR, blocking = false, fftflags = FFTW.ESTIMATE)
  
  trj[1,:] .= 0.008333333 # throws error
  nfftplan = plan_nfft(trj, img_shape_os; precompute = LINEAR, blocking = false, fftflags = FFTW.ESTIMATE)
  mul!(λ, adjoint(nfftplan), ones(Complex{T}, size(trj,2)))
  

end