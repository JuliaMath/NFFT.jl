module AbstractNFFTsChainRulesCoreExt

using AbstractNFFTs
import ChainRulesCore

###############
# mul-interface
###############
function ChainRulesCore.frule((_, Δx, _), ::typeof(*), plan::AbstractFTPlan, x::AbstractArray)
  y = plan*x
  Δy = plan*Δx
  return y, Δy
end
function ChainRulesCore.rrule(::typeof(*), plan::AbstractFTPlan, x::AbstractArray)
  y = plan*x
  project_x = ChainRulesCore.ProjectTo(x)
  function mul_pullback(ȳ)
      x̄ = project_x( adjoint(plan)*ChainRulesCore.unthunk(ȳ) )
      return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), x̄
  end
  return y, mul_pullback
end

##################
# NFFT, NFCT, NFST
##################
for (op,trans) in zip([:nfft, :nfct, :nfst], [:adjoint, :transpose, :transpose])

  func_trans = Symbol("$(op)_$(trans)")
  pbfunc = Symbol("$(op)_pullback")
  pbfunc_trans = Symbol("$(op)_$(trans)_pullback")
  @eval begin
  
    # direct trafo
    function ChainRulesCore.frule((_, Δx, _), ::typeof($(op)), k::AbstractArray, x::AbstractArray)
      y = $(op)(k,x)
      Δy = $(op)(k,Δx)
      return y, Δy
    end
    function ChainRulesCore.rrule(::typeof($(op)), k::AbstractArray, x::AbstractArray)
      y = $(op)(k,x)
      project_x = ChainRulesCore.ProjectTo(x)
      function $(pbfunc)(ȳ)
        x̄ = project_x($(func_trans)(k, size(x), ChainRulesCore.unthunk(ȳ)))
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), x̄
      end
      return y, nfft_pullback
    end

    # adjoint trafo
    function ChainRulesCore.frule((_, Δx, _), ::typeof($(func_trans)), k::AbstractMatrix, N, x::AbstractArray)
      y = $(func_trans)(k,N,x)
      Δy = $(func_trans)(k,N,Δx)
      return y, Δy
    end
    function ChainRulesCore.rrule(::typeof($(func_trans)), k::AbstractArray, N, x::AbstractArray)
      y = $(func_trans)(k,N,x)
      project_x = ChainRulesCore.ProjectTo(x)
      function $(pbfunc_trans)(ȳ)
        x̄ = project_x($(op)(k, ChainRulesCore.unthunk(ȳ)))
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), x̄
      end
      return y, $(pbfunc_trans)
    end

  end

end




end # module