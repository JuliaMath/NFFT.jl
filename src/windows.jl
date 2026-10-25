"""
    AbstractWindow

Window function of the NFFT. A window is evaluated in grid units, `φ(z)` for `z ∈ [-W/2, W/2]`
(W the window width, W = 2m in the notation of Knopp et al. 2023); it is symmetric and zero
outside. The resampling evaluates a piecewise polynomial fitted to φ, so the cost of φ itself
does not matter.

Families with a parameter table (used by the automatic parameter choice): [`KaiserBessel`](@ref),
[`ExpSemicircle`](@ref). Any other window can be used with explicit `W` and `σ` through
[`CustomWindow`](@ref).
"""
abstract type AbstractWindow end

"""
    KaiserBessel()

Continuous Kaiser–Bessel window (Potts & Tasche 2021): the sinh form
sinh(β√(1−(2z/W)²))/√(1−(2z/W)²) minus its edge value β, so that it vanishes at |z| = W/2. The shape
parameter β is taken from the kernel table (optimised per W and σ).
"""
struct KaiserBessel <: AbstractWindow end

"""
    ExpSemicircle()

Exponential of semicircle family, φ(z) = exp(β((1−(2z/W)²)^e₀ − 1)). e₀ = 1/2 is the ES kernel of
FINUFFT (Barnett et al. 2019); a free exponent e₀ is the family used by ducc (Arras et al. 2021).
β and e₀ are taken from the kernel table.
"""
struct ExpSemicircle <: AbstractWindow end

"""
    CustomWindow(φ)

User-defined window: `φ(z, W)` must be symmetric in z and zero for |z| ≥ W/2. Requires explicit
`W` and `σ` in the plan.
"""
struct CustomWindow{F} <: AbstractWindow
  φ::F
end

# Continuous Kaiser–Bessel window (Potts & Tasche 2021): the sinh form minus its value β at the
# edge, so that φ(±W/2) = 0. The plain sinh form has φ(±W/2) ≈ 2β·e^{−β}; a node exactly on a grid
# point then gets one edge point inside the support but not its mirror (W = 6 in Float32: errors of
# 1.3e-4 at reltol 1e-5 in 3D).
function kb_window(z, W, β)
  t = 1 - (2z / W)^2
  t > 0 || return zero(z)
  s = sqrt(t)
  ((s < 1e-8 ? β : sinh(β * s) / s) - β) / (sinh(β) - β)
end

es_window(z, W, β, e0) = (t = 1 - (2z / W)^2; t > 0 ? exp(β * (t^e0 - 1)) : zero(z))

table_name(::KaiserBessel) = "kb"
table_name(::ExpSemicircle) = "es"
table_window(name::AbstractString) = name == "kb" ? KaiserBessel() : ExpSemicircle()

# The function z ↦ φ(z) for a table entry or a custom window.
window_function(::KaiserBessel, W, β, e0) = z -> kb_window(z, W, β)
window_function(::ExpSemicircle, W, β, e0) = z -> es_window(z, W, β, e0)
window_function(w::CustomWindow, W, β, e0) = z -> w.φ(z, W)
