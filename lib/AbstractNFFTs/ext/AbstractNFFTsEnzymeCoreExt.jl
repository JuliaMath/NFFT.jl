module AbstractNFFTsEnzymeCoreExt

using AbstractNFFTs, LinearAlgebra
using AbstractNFFTs: WrappedPlan, pullback_plan
using EnzymeCore
using EnzymeCore: EnzymeRules

# A plan is a linear map that does not depend on the data: Enzyme treats plans as constants and
# differentiates `mul!(y, p, x)` by applying p to the tangent (forward mode) and the pullback plan
# to the cotangent (reverse mode). Without these rules Enzyme would differentiate the threaded
# kernels and the FFT library calls of an implementation, which it cannot do correctly.
const PlanLike = Union{AbstractFTPlan,WrappedPlan}

EnzymeRules.inactive_type(::Type{<:AbstractFTPlan}) = true
for plan in (:plan_nfft, :plan_nfct, :plan_nfst, :plan_rnfft, :plan_nnfft)
  @eval EnzymeRules.inactive(::typeof(AbstractNFFTs.$plan), args...; kwargs...) = nothing
end

shadows(config, x) = EnzymeRules.width(config) == 1 ? (x.dval,) : x.dval

# cotangent z added to the shadow dx; a real input takes the real part
accumulate!(dx::AbstractArray{<:Real}, z) = (dx .+= real.(z); dx)
accumulate!(dx::AbstractArray, z) = (dx .+= z; dx)

function EnzymeRules.forward(config::EnzymeRules.FwdConfig, ::Const{typeof(mul!)}, ::Type{RT},
                             y::Annotation, p::Const{<:PlanLike}, x::Annotation) where {RT}
  mul!(y.val, p.val, x.val)
  if !(y isa Const)
    for (i, dy) in enumerate(shadows(config, y))
      x isa Const ? fill!(dy, 0) : mul!(dy, p.val, shadows(config, x)[i])
    end
  end
  if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
    EnzymeRules.width(config) == 1 ? Duplicated(y.val, y.dval) : BatchDuplicated(y.val, y.dval)
  elseif EnzymeRules.needs_shadow(config)
    y.dval
  elseif EnzymeRules.needs_primal(config)
    y.val
  else
    nothing
  end
end

# The derivative does not depend on x (the map is linear), so nothing is taped.
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(mul!)}, ::Type{RT},
                                      y::Annotation, p::Const{<:PlanLike}, x::Annotation) where {RT}
  mul!(y.val, p.val, x.val)
  primal = EnzymeRules.needs_primal(config) ? y.val : nothing
  shadow = EnzymeRules.needs_shadow(config) ? y.dval : nothing
  EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig, ::Const{typeof(mul!)}, ::Type{RT}, tape,
                             y::Annotation, p::Const{<:PlanLike}, x::Annotation) where {RT}
  if !(y isa Const)
    q = pullback_plan(p.val)
    for (i, dy) in enumerate(shadows(config, y))
      x isa Const || accumulate!(shadows(config, x)[i], q * dy)
      fill!(dy, 0)
    end
  end
  (nothing, nothing, nothing)
end

end # module
