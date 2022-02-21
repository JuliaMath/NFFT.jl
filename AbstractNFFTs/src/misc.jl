# Precompute flags for the B matrix
@enum PrecomputeFlags begin
  FULL = 1
  TENSOR = 2
  LINEAR = 3
  POLYNOMIAL = 4
end

# Timing functions that allow for timing parts of an NFFT

mutable struct TimingStats
  pre::Float64
  conv::Float64
  fft::Float64
  apod::Float64
  conv_adjoint::Float64
  fft_adjoint::Float64
  apod_adjoint::Float64
end

TimingStats() = TimingStats(0.0,0.0,0.0,0.0,0.0,0.0,0.0)

function Base.println(t::TimingStats)
  print("Timing: ")
  @printf "pre = %.4f s apod = %.4f / %.4f s fft = %.4f / %.4f s conv = %.4f / %.4f s\n" t.pre t.apod t.apod_adjoint t.fft t.fft_adjoint t.conv t.conv_adjoint

  total = t.apod + t.fft + t.conv 
  totalAdj = t.apod_adjoint + t.fft_adjoint + t.conv_adjoint
  @printf "                       apod = %.4f / %.4f %% fft = %.4f / %.4f %% conv = %.4f / %.4f %%\n" 100*t.apod/total 100*t.apod_adjoint/totalAdj 100*t.fft/total 100*t.fft_adjoint/totalAdj 100*t.conv/total 100*t.conv_adjoint/totalAdj
  
end

function reltolToParams(reltol) 
  w = ceil(Int, log(10,1/reltol)) + 1 
  m = (w)÷2
  return m, 2.0
end

function paramsToReltol(m::Int, σ)
  w = 2*m 
  return 10.0^(-(w-1))
end


"""
  accuracyParams(; [m, σ, reltol]) -> m, σ, reltol

Calculate accuracy parameters m, σ, reltol based on either
* reltol
or
* m, σ

TODO: Right now, the oversampling parameter is not taken into account, i.e. σ=2.0 is assumed
"""
function accuracyParams(; kargs...)

  if haskey(kargs, :reltol)
    reltol = kargs[:reltol]
    m, σ = reltolToParams(reltol)
  elseif haskey(kargs, :m) && haskey(kargs, :σ)
    m = kargs[:m]
    σ = kargs[:σ]
    reltol = paramsToReltol(m, σ)
  else
    reltol = 1e-9
    m, σ = reltolToParams(reltol)
  end

  return m, σ, reltol
end
