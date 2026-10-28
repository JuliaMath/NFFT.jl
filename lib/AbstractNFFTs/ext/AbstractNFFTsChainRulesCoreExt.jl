module AbstractNFFTsChainRulesCoreExt

using AbstractNFFTs, LinearAlgebra
import ChainRulesCore

const NT = ChainRulesCore.NoTangent

###############
# mul-interface
###############
# The pullback of a plan is its adjoint, for the real transforms (NFCT, NFST) its transpose; the
# pullback of an adjoint or transposed plan is the plan itself.
const WrappedPlan = Union{Adjoint{<:Any,<:AbstractFTPlan},Transpose{<:Any,<:AbstractFTPlan}}
pullback_plan(plan::AbstractRealFTPlan) = transpose(plan)
pullback_plan(plan::AbstractFTPlan) = adjoint(plan)
pullback_plan(plan::WrappedPlan) = parent(plan)

# Adjoint and Transpose are AbstractMatrix: the second signature is more specific than the
# matrix-times-vector rule of ChainRules.
for X in (AbstractArray, AbstractVecOrMat{<:Union{Real,Complex}})
  @eval begin
    function ChainRulesCore.frule((_, _, Δx), ::typeof(*), plan::Union{AbstractFTPlan,WrappedPlan}, x::$X)
      return plan * x, plan * Δx
    end
    function ChainRulesCore.rrule(::typeof(*), plan::Union{AbstractFTPlan,WrappedPlan}, x::$X)
      y = plan * x
      project_x = ChainRulesCore.ProjectTo(x)
      mul_pullback(ȳ) = (NT(), NT(), project_x(pullback_plan(plan) * ChainRulesCore.unthunk(ȳ)))
      return y, mul_pullback
    end
  end
end

##################
# NFFT, NFCT, NFST
##################
for (op, trans) in zip([:nfft, :nfct, :nfst], [:adjoint, :transpose, :transpose])

  func_trans = Symbol("$(op)_$(trans)")
  pbfunc = Symbol("$(op)_pullback")
  pbfunc_trans = Symbol("$(op)_$(trans)_pullback")
  @eval begin

    # direct transform
    function ChainRulesCore.frule((_, _, Δx), ::typeof($(op)), k::AbstractArray, x::AbstractArray; kwargs...)
      return $(op)(k, x; kwargs...), $(op)(k, Δx; kwargs...)
    end
    function ChainRulesCore.rrule(::typeof($(op)), k::AbstractArray, x::AbstractArray; kwargs...)
      y = $(op)(k, x; kwargs...)
      project_x = ChainRulesCore.ProjectTo(x)
      $(pbfunc)(ȳ) = (NT(), NT(), project_x($(func_trans)(k, size(x), ChainRulesCore.unthunk(ȳ); kwargs...)))
      return y, $(pbfunc)
    end

    # adjoint / transposed transform
    function ChainRulesCore.frule((_, _, _, Δx), ::typeof($(func_trans)), k::AbstractArray, N, x::AbstractArray; kwargs...)
      return $(func_trans)(k, N, x; kwargs...), $(func_trans)(k, N, Δx; kwargs...)
    end
    function ChainRulesCore.rrule(::typeof($(func_trans)), k::AbstractArray, N, x::AbstractArray; kwargs...)
      y = $(func_trans)(k, N, x; kwargs...)
      project_x = ChainRulesCore.ProjectTo(x)
      $(pbfunc_trans)(ȳ) = (NT(), NT(), NT(), project_x($(op)(k, ChainRulesCore.unthunk(ȳ); kwargs...)))
      return y, $(pbfunc_trans)
    end

  end
end

end # module
