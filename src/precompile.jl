using SnoopPrecompile 

@precompile_setup begin
    @precompile_all_calls begin

      J, N = 8, 16
      k = range(-0.4, stop=0.4, length=J)  # nodes at which the NFFT is evaluated
      f = randn(ComplexF64, J)             # data to be transformed
      p = plan_nfft(k, N, reltol=1e-9)     # create plan
      fHat = adjoint(p) * f                # calculate adjoint NFFT
      y = p * fHat                         # calculate forward NFFT

    end
end
