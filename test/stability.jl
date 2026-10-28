# Type stability and allocations of the transforms of every plan type. A plan holds its engine (the
# core) in an untyped field: one dynamic dispatch per transform. Behind it everything a transform
# reads has a concrete type, checked field by field (`concrete_core`), and the functions behind
# the barrier are inferred. A transform allocates a constant amount (task scheduling, views), never
# memory of the size of the data: the bounds below are far below the size of the arrays of the
# large problem.

# A concrete field type; arrays and tuples with concrete element types; a union of such types.
concrete_type(T) = T isa Union ? all(concrete_type, Base.uniontypes(T)) :
  isconcretetype(T) && (T <: AbstractArray ? concrete_type(eltype(T)) :
                        T <: Tuple ? all(concrete_type, fieldtypes(T)) : true)

# All fields of the engine `x` and of the NFFT.jl structs inside it have concrete types (plans are
# barriers of their own; `fft`, the backend, is only read when a core is copied). Returns the
# offending fields.
function concrete_core(x, path="core", bad=String[])
  if x isa AbstractArray
    isempty(x) || parentmodule(typeof(first(x))) === NFFT && concrete_core(first(x), "$path[1]", bad)
    return bad
  end
  T = typeof(x)
  (parentmodule(T) === NFFT && !(x isa AbstractNFFTs.AbstractFTPlan) && isstructtype(T)) || return bad
  for (name, FT) in zip(fieldnames(T), fieldtypes(T))
    name === :fft && continue
    concrete_type(FT) || push!(bad, "$path.$name::$FT")
    concrete_core(getfield(x, name), "$path.$name", bad)
  end
  bad
end

# The functions behind the barrier, called with the core `c` (of concrete type at the call)
inner_trafo!(p::NFFT.FastPlan, s, c, fHat, f) = NFFT.transform!(s, p, c, fHat, f)
inner_adjoint!(p::NFFT.FastPlan, s, c, f, fHat) = NFFT.transform_adjoint!(s, p, c, f, fHat)
inner_trafo!(p::NNFFTPlan, s, c, fHat, f) = NFFT.nnfft!(vec(fHat), c, vec(f))
inner_adjoint!(p::NNFFTPlan, s, c, f, fHat) = NFFT.nnfft_adjoint!(vec(f), c, vec(fHat))

# bytes allocated by mul!, the fewest of three calls after a warm-up (in a function: no boxing of globals)
function mul_allocs(y, p, x)
  mul!(y, p, x)
  minimum(@allocated(mul!(y, p, x)) for _ in 1:3)
end

function stability_plans(J, n, nthreads)
  rng = Xoshiro(5)
  k1 = rand(rng, 1, J) .- 0.5; k2 = rand(rng, 2, J) .- 0.5; k3 = rand(rng, 3, J) .- 0.5
  kp = abs.(k2) ./ 2
  C = ComplexF64
  # (name, plan, input type, output type)
  [("NFFT 1D", plan_nfft(k1, (n^2,); nthreads), C, C),
   ("NFFT 2D", plan_nfft(k2, (n, n); nthreads), C, C),
   ("NFFT 3D", plan_nfft(k3, (n ÷ 4, n ÷ 4, n ÷ 4); nthreads), C, C),
   ("NFFT Float32", plan_nfft(Float32.(k2), (n, n); nthreads), ComplexF32, ComplexF32),
   ("NFFT batch first", plan_nfft(k2, (4, n, n); dims=2:3, nthreads), C, C),
   ("NFFT batch last", plan_nfft(k2, (n, n, 4); dims=1:2, nthreads), C, C),
   ("NFFT batch middle", plan_nfft(k1, (3, n, 5); dims=2, nthreads), C, C),
   ("NFFT many slices", plan_nfft(k1[:, 1:50], (16, 64); dims=1, nthreads), C, C),
   ("NFFT 3D batch", plan_nfft(k3, (n ÷ 4, n ÷ 4, n ÷ 4, 3); dims=1:3, nthreads), C, C),
   ("RNFFT", plan_rnfft(k2, (n, n); nthreads), C, Float64),
   ("RNFFT batch", plan_rnfft(k1, (n, 6); dims=1, nthreads), C, Float64),
   ("NFCT", plan_nfct(kp, (n, n); nthreads), Float64, Float64),
   ("NFST", plan_nfst(kp, (n, n); nthreads), Float64, Float64),
   ("NFCT batch", plan_nfct(kp[1:1, :], (n, 6); dims=1, nthreads), Float64, Float64),
   ("NNFFT", plan_nnfft(k2, rand(rng, 2, J ÷ 2) .- 0.5; nthreads), C, C),
   ("NDFT", NDFTPlan(k2[:, 1:50], (8, 8)), C, C)]
end

@testset "Type stability and allocations" begin
  # the plan type is inferred when `dims` is left out or a single dimension
  k1 = rand(1, 20) .- 0.5; k2 = rand(2, 20) .- 0.5
  @test @inferred(plan_nfft(k2, (8, 8))) isa NFFTPlan{Float64,2,1}
  @test @inferred(plan_nfft(Float32.(k2), (8, 8); reltol=1e-4)) isa NFFTPlan{Float32,2,1}
  @test @inferred(plan_nfft(k1, (8, 3); dims=1)) isa NFFTPlan{Float64,2,2}
  @test @inferred(plan_nfft(vec(k1), (8,))) isa NFFTPlan{Float64,1,1}
  @test @inferred(plan_rnfft(k2, (8, 8))) isa RNFFTPlan{Float64,2,1}
  @test @inferred(plan_nfct(abs.(k2) ./ 2, (8, 8))) isa NFCTPlan{Float64,2,1}
  @test @inferred(plan_nfst(abs.(k1) ./ 2, (3, 8); dims=2)) isa NFSTPlan{Float64,2,2}

  for (name, p, Tin, Tout) in stability_plans(500, 16, Threads.nthreads())
    @testset "$name" begin
      back = Tin <: Real ? transpose(p) : adjoint(p)
      f = randn(Tin, size_in(p)); fHat = randn(Tout, size_out(p))
      @test @inferred(p * f) isa Array{Tout}
      @test @inferred(back * fHat) isa Array{Tin}
      @test @inferred(mul!(fHat, p, f)) === fHat
      @test @inferred(mul!(f, back, fHat)) === f
      # behind the barrier
      if p isa Union{NFFT.FastPlan,NNFFTPlan}
        # in the plan only the barrier and configuration that no transform reads are abstract
        abstract_fields = [n for (n, T) in zip(fieldnames(typeof(p)), fieldtypes(typeof(p))) if !concrete_type(T)]
        @test abstract_fields ⊆ [:k, :y, :ysrc, :fft, :core]
        @test isempty(concrete_core(p.core))
        s = p isa NFFT.FastPlan ? Val(p.strategy) : nothing
        @inferred inner_trafo!(p, s, p.core, fHat, f)
        @inferred inner_adjoint!(p, s, p.core, f, fHat)
      end
      # views of the data
      fv = view(randn(Tin, size_in(p) .+ 1), axes(f)...)
      @test @inferred(p * fv) ≈ p * collect(fv)
    end
  end

  # single-threaded: a few KB per call, the same for a small and a large problem; with threads a
  # few KB per task
  for (J, n) in ((500, 16), (20_000, 128)), (nt, bound) in ((1, 2^15), (Threads.nthreads(), 2^15 * (Threads.nthreads() + 1)))
    @testset "allocations, J = $J, nthreads = $nt: $name" for (name, p, Tin, Tout) in stability_plans(J, n, nt)
      back = Tin <: Real ? transpose(p) : adjoint(p)
      f = randn(Tin, size_in(p)); fHat = randn(Tout, size_out(p))
      # the batch plans with threaded cores per slice (:loop) schedule tasks for every slice
      L = p isa NFFT.FastPlan && p.strategy === :loop ? prod(size_in(p)) ÷ prod(p.N[p.dims]) : 1
      @test mul_allocs(fHat, p, f) <= L * bound
      @test mul_allocs(f, back, fHat) <= L * bound
    end
  end
end
