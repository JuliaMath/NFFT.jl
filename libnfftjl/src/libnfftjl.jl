# C library of NFFT.jl, compiled with `juliac --trim`: NFFT, RNFFT, NFCT and NFST and their
# adjoints in 1–3 dimensions, with batch dimensions, double and single precision, parameters chosen from NFFT.jl's kernel table. Interface in
# include/nfftjl.h.
#
# Trimming needs every call to be resolved at compile time. The plans of NFFT.jl choose the window
# width W at run time, and the type of the core depends on it; this file is the boundary that turns
# the run-time (transform, T, D, W, batch strategy) into one compiled branch per combination and
# keeps the cores in registries of concrete type.

using NFFT
using NFFT: NFFTCore, SymKernel, KaiserBessel, FFTWBackend, kb_window, es_window, grid_size,
            choose_kernels, make_kernel, trafo_core!, adjoint_core!

# NFFT.jl calls no BLAS or LAPACK routine (planning has its own least squares and quadrature): the
# library starts without loading OpenBLAS, and build.sh leaves it out of the bundle.
using LinearAlgebra
@eval LinearAlgebra.OpenBLAS_jll __init__() = nothing
@eval LinearAlgebra __init__() = nothing

# FFTW.jl destroys plans queued by finalizers through a call on its abstract plan type, which
# trimming cannot resolve; this version calls the C functions directly.
using NFFT: FFTW
@eval FFTW function destroy_deferred()
  lock(deferred_destroy_lock)
  try
    if !isempty(deferred_destroy_plans) && trylock(fftwlock)
      try
        for p in deferred_destroy_plans
          ptr = getfield(p, :plan)::PlanPtr
          if p isa FFTWPlan{<:fftwDouble}
            ccall((:fftw_destroy_plan, libfftw3), Cvoid, (PlanPtr,), ptr)
          else
            ccall((:fftwf_destroy_plan, libfftw3f), Cvoid, (PlanPtr,), ptr)
          end
        end
        empty!(deferred_destroy_plans)
      finally
        unlock(fftwlock)
      end
    end
  finally
    unlock(deferred_destroy_lock)
  end
end

# FFTW.jl loads libfftw3 on first use through a method that looks the library up by name and calls
# an untyped callback; this version is resolved at compile time. `load_fftw` makes it part of the
# library.
@eval FFTW function dlopen(lib::FakeLazyLibrary)
  h = @atomic :monotonic lib.h
  h != C_NULL && return h
  @lock fftwlock begin
    h = @atomic :monotonic lib.h
    h != C_NULL && return h
    h = dlopen(lib.reallibrary === :libfftw3_no_init ? libfftw3_no_init : libfftw3f_no_init)
    fftw_init_threads()
    @atomic :release lib.h = h
  end
  h
end
# FFTW's threads are Julia tasks started by this callback; `@sync` waits on an untyped channel.
@eval FFTW function spawnloop(f::Ptr{Cvoid}, fdata::Ptr{Cvoid}, elsize::Csize_t, num::Cint, callback_data::Ptr{Cvoid})
  ts = Vector{Task}(undef, num)
  for i = 0:num-1
    ts[i+1] = Threads.@spawn ccall(f, Ptr{Cvoid}, (Ptr{Cvoid},), fdata + elsize*i)
  end
  foreach(wait, ts)
end
# The check of an out-of-place FFTW plan builds its error message with `string` of a run-time
# choice, which trimming cannot resolve; this version has fixed messages.
@eval FFTW function assert_applicable(p::FFTWPlan{T,K,inplace}, X::StridedArray{T}, Y::StridedArray) where {T,K,inplace}
  assert_applicable(p, X)
  size(Y) == p.osz || throw(ArgumentError("FFTW plan applied to wrong-size output"))
  strides(Y) == p.ostride || throw(ArgumentError("FFTW plan applied to wrong-strides output"))
  (alignment_of(Y) == p.oalign || p.flags & UNALIGNED != 0) ||
    throw(ArgumentError("FFTW plan applied to output with wrong memory alignment"))
  inplace == (pointer(X) == pointer(Y)) || throw(ArgumentError("FFTW plan applied to in-place or out-of-place data it was not made for"))
  nothing
end
load_fftw() = (NFFT.FFTW.dlopen(NFFT.FFTW.libfftw3); NFFT.FFTW.dlopen(NFFT.FFTW.libfftw3f); nothing)

# The variants of the library, one compiled branch each: transform × element type × dimension ×
# window width × batch strategy (as in NFFT.jl: no batch; slices one at a time, on one threaded core
# or in parallel on single-threaded cores; for the NFFT the batch kernels for batches before or
# after the transformed dimensions). A handle is
# index << 32 | x << 28 | s << 24 | t << 16 | D << 8 | W, with x, s and t the positions in
# TRANSFORMS, STRATEGIES and TS. The parameter choice never takes W > 8 in single precision.
const TRANSFORMS = (:nfft, :rnfft, :nfct, :nfst)
const TS = (Float64, Float32)
const DS = 1:3
widths(::Type{Float64}) = 4:16
widths(::Type{Float32}) = 4:8
const STRATEGIES = (:single, :slices, :lead, :trail)

# Element types of the input and the output of a transform.
eltypes(X, T) = X === :nfft ? (Complex{T}, Complex{T}) : X === :rnfft ? (Complex{T}, T) : (T, T)

# The window of a table entry as one callable type for both families (Kaiser–Bessel and
# exponential of semicircle): the planning code is compiled once instead of once per family.
struct TableWindow
  kb::Bool
  W::Int
  β::Float64
  e0::Float64
end
(w::TableWindow)(z) = w.kb ? kb_window(z, w.W, w.β) : es_window(z, w.W, w.β, w.e0)

plan_type(X) = X === :nfft ? NFFTPlan : X === :rnfft ? RNFFTPlan : X === :nfct ? NFCTPlan : NFSTPlan
kernel_type(T, W) = SymKernel{W,cld(W, 2),cld(W + 4, 2),(W + 4) ÷ 2,T}

# The core of one transform (no batch), on the grid of NFFT.jl for oversampling σ.
function make_core(::Val{X}, ::Type{T}, k, N::NTuple{D,Int}, σ, W, φ::F, kernel, nthreads) where {X,T,D,F}
  fft = FFTWBackend()
  if X === :nfft || X === :rnfft
    Ñ = grid_size(σ, N, W)
    X === :nfft ? NFFTCore(k, N, Ñ, W, φ, T, fft, nothing, nthreads; kernel) :
                  NFFT.RNFFTCore(k, N, Ñ, W, φ, T, fft, nothing, nthreads; kernel)
  else
    S = X === :nfct ? :cos : :sin
    NFFT.RealCore{S}(k, N, grid_size(σ, NFFT.virtual_size(S, N), W), W, φ, T, fft, nothing, nthreads; kernel)
  end
end

# The batch kernels of the NFFT with chunks of 4 (batches of at least 4).
make_batch(::Type{T}, k, N::NTuple{D,Int}, σ, W, φ::F, kernel, L, nthreads) where {T,D,F} =
  NFFT.BatchCore(k, N, grid_size(σ, N, W), W, φ, T, FFTWBackend(), L, nothing, nthreads; kernel, chunk=Val(4))

# The core types, inferred from the argument types (and checked to be concrete) instead of written
# out.
function inferred(f, argtypes...)
  R = Core.Compiler.return_type(f, Tuple{argtypes...})
  isconcretetype(R) || error("not concrete: $R for $f$(argtypes)")
  R
end
argtypes(T, D, W) = (Matrix{Float64}, NTuple{D,Int}, Float64, Int, TableWindow, kernel_type(T, W))
core_type(X, T, D, W) = inferred(make_core, Val{X}, Type{T}, argtypes(T, D, W)..., Int)
batch_type(T, D, W) = inferred(make_batch, Type{T}, argtypes(T, D, W)..., Int, Int)

engine_core(X, S, T, D, W) =
  W ∉ widths(T) ? Nothing :
  S === :single ? core_type(X, T, D, W) :
  S === :slices ? Vector{core_type(X, T, D, W)} :
  X !== :nfft || D > 2 ? Nothing :                         # batch kernels: NFFT in 1D and 2D
  S === :lead ? batch_type(T, D, W) :
  NFFT.TrailCore{batch_type(T, D, W),Array{Complex{T},D + 1},Matrix{Complex{T}}}

# A plan of the library: its core, the batch sizes before (L1) and after (L3) the transformed
# dimensions, the number of nodes, and for :slices one slice buffer per core. Data are
# (L1, Nin..., L3) and (L1, J, L3), column major; Nin = N, for the RNFFT N[1] ÷ 2 + 1 in dimension 1.
struct Engine{X,S,C,F,G}
  core::C
  L1::Int
  L3::Int
  J::Int
  fbufs::Vector{F}
  gbufs::Vector{G}
end
engine_type(X, S, C, T, D) = Engine{X,S,C,Array{eltypes(X, T)[1],D},Vector{eltypes(X, T)[2]}}

# One registry of plans per variant; Nothing[] for combinations that do not exist.
for X in TRANSFORMS, S in STRATEGIES, T in TS, D in DS, W in widths(Float64)
  C = engine_core(X, S, T, D, W)
  E = C === Nothing ? Nothing : Union{Nothing,engine_type(X, S, C, T, D)}
  @eval const $(Symbol(:plans_, X, :_, S, :_, T, :_, D, :_, W)) = $E[]
end
@generated registry(::Val{X}, ::Val{S}, ::Type{T}, ::Val{D}, ::Val{W}) where {X,S,T,D,W} =
  Symbol(:plans_, X, :_, S, :_, T, :_, D, :_, W)

index(x, xs) = findfirst(==(x), xs)
name(::Val{S}) where {S} = S
handle(X, S, T, D, W, i) = Int64(i) << 32 | Int64(index(X, TRANSFORMS)) << 28 | Int64(index(S, STRATEGIES)) << 24 |
                           Int64(index(T, TS)) << 16 | Int64(D) << 8 | Int64(W)
unhandle(h) = (Int((h >> 28) & 0xf), Int((h >> 24) & 0xf), Int((h >> 16) & 0xff), Int((h >> 8) & 0xff),
               Int(h & 0xff), Int(h >> 32))

# f(Val(TRANSFORMS[x])), f(Val(STRATEGIES[s])), f(TS[t]), f(Val(D)), f(Val(W)) for run-time values,
# one branch per value (window widths per element type).
for (name, values, arg, err, T) in ((:with_transform, 1:length(TRANSFORMS), x -> :(Val($(QuoteNode(TRANSFORMS[x])))),
                                     "unsupported transform", nothing),
                                    (:with_strategy, 1:length(STRATEGIES), s -> :(Val($(QuoteNode(STRATEGIES[s])))),
                                     "unsupported strategy", nothing),
                                    (:with_type, 1:length(TS), t -> :($(TS[t])), "unsupported element type", nothing),
                                    (:with_dim, DS, D -> :(Val($D)), "unsupported dimension", nothing),
                                    ((:with_width, widths(T), W -> :(Val($W)), "unsupported window width", T) for T in TS)...)
  body = :(throw(ArgumentError($err)))
  for v in reverse(values)
    body = :(x == $v ? f($(arg(v))) : $body)
  end
  if T === nothing
    @eval $name(f, x) = $body
  else
    @eval $name(f, x, ::Type{$T}) = $body
  end
end

# Plans can be created and destroyed from any thread; a plan must not run on two threads at once.
# Planning runs one at a time: FFTW's planner thread count is global state.
const REGISTRY_LOCK = ReentrantLock()
const PLAN_LOCK = ReentrantLock()

# The batch strategy of NFFT.jl (`make_engine`), with the batch kernels of the NFFT for batches of at
# least 4: (strategy, slices in parallel).
function strategy(X, D, N, J, L1, L3, nthreads)
  L = L1 * L3
  L == 1 && return 1, false
  NFFT.use_slices(L, J + prod(N), nthreads) && return 2, true
  X === :nfft && D <= 2 && L3 == 1 && L1 >= 4 && return 3, false
  X === :nfft && D <= 2 && L1 == 1 && L3 >= 4 && return 4, false
  2, false
end

function make_engine(::Val{X}, ::Val{S}, ::Type{T}, k, N::NTuple{D,Int}, σ, W, φ::F, kernel, nthreads, L1, L3, J,
                     parallel) where {X,S,T,D,F}
  core = if S === :single
    make_core(Val(X), T, k, N, σ, W, φ, kernel, nthreads)
  elseif S === :slices
    c = make_core(Val(X), T, k, N, σ, W, φ, kernel, nthreads)
    parallel ? NFFT.single_cores(c, L1 * L3) : [c]
  elseif X !== :nfft || D > 2
    throw(ArgumentError("no batch kernels for this transform"))
  elseif S === :lead
    make_batch(T, k, N, σ, W, φ, kernel, L1, nthreads)
  else
    NFFT.TrailCore(make_batch(T, k, N, σ, W, φ, kernel, L3, nthreads),
                   Array{Complex{T}}(undef, L3, N...), Matrix{Complex{T}}(undef, L3, J))
  end
  Tin, Tout = eltypes(X, T)
  nbuf = S === :slices ? length(core) : 0
  Engine{X,S,typeof(core),Array{Tin,D},Vector{Tout}}(core, L1, L3, J,
    [Array{Tin}(undef, NFFT.input_size(plan_type(X), N, 1:D)) for _ in 1:nbuf], [Vector{Tout}(undef, J) for _ in 1:nbuf])
end

function create(::Val{X}, ::Type{T}, ::Val{D}, Np::Ptr{Int64}, J::Int, kp::Ptr{Float64}, reltol::Float64,
                nthreads::Int, L1::Int, L3::Int) where {X,T,D}
  load_fftw()
  (L1 >= 1 && L3 >= 1) || throw(ArgumentError("batch sizes must be positive"))
  N = ntuple(d -> Int(unsafe_load(Np, d)), Val(D))
  k = copy(unsafe_wrap(Array, kp, (D, J)))
  NFFT.check_nodes(plan_type(X), k)
  e = first(choose_kernels(reltol, NFFT.error_size(plan_type(X), N), J, T; Ncost=N)[1])
  W = e.W
  st, parallel = strategy(X, D, N, J, L1, L3, nthreads)
  φ = TableWindow(e.window isa KaiserBessel, W, e.β, e.e0)
  with_width(W, T) do w
    kernel = make_kernel(φ, w, T)
    with_strategy(st) do s
      eng = make_engine(Val(X), s, T, k, N, e.σ, W, φ, kernel, nthreads, L1, L3, J, parallel)
      reg = registry(Val(X), s, T, Val(D), w)
      @lock REGISTRY_LOCK begin
        push!(reg, eng)
        handle(X, name(s), T, D, W, length(reg))
      end
    end
  end
end

# f(registry, index) for the registry of plan h, which must be of transform X and element type T
# (-1 otherwise). The dispatch runs over the strategy, the dimension and the window width only:
# over all transforms and element types it would compile every transform for the data pointers of
# every other.
function with_registry(f, h, ::Val{X}, ::Type{T}) where {X,T}
  x, s, t, D, W, i = unhandle(h)
  (x == index(X, TRANSFORMS) && t == index(T, TS)) || return Cint(-1)
  with_strategy(s) do sv
    with_dim(D) do d
      with_width(W, T) do w
        f(registry(Val(X), sv, T, d, w), i)
      end
    end
  end
end

# f(engine) for plan h of transform X and element type T; -1 for an unknown handle or another
# transform or element type
lookup(f, ::Val{X}, ::Type{T}, h) where {X,T} = with_registry(h, Val(X), T) do reg, i
  e = @lock REGISTRY_LOCK (1 <= i <= length(reg) ? reg[i] : nothing)
  e === nothing ? Cint(-1) : f(e)
end

# size of the input of one transform
insize(c::NFFT.BatchCore) = c.N
insize(t::NFFT.TrailCore) = t.core.N
insize(c::Vector) = NFFT.slice_size(c[1])
insize(c) = NFFT.slice_size(c)

# The transforms on the data of the plan, (L1, Nin..., L3) and (L1, J, L3). The batch kernels (:lead,
# :trail) read only the full size and the transformed dimensions of their plan. :slices runs a
# core on one slice at a time as an Array, so that the core is compiled for Arrays only (for
# views of the slices it would be compiled a second time): with L1 = 1 the slices are contiguous
# and are wrapped in place, otherwise they are copied through the buffers of the task.
function run_trafo(e::Engine{X,S}, fp, gp) where {X,S}
  N = insize(e.core)
  if S === :single
    trafo_core!(unsafe_wrap(Array, gp, e.J), e.core, unsafe_wrap(Array, fp, N))
  elseif S === :lead || S === :trail
    NFFT.transform!(Val(S), (N=(e.L1, N..., e.L3), dims=2:length(N)+1), e.core,
                    unsafe_wrap(Array, gp, (e.L1, e.J, e.L3)), unsafe_wrap(Array, fp, (e.L1, N..., e.L3)))
  else
    each_slice(e) do c, t, a, b
      slice!(c, e, t, a, b, fp, gp, true)
    end
  end
  Cint(0)
end
function run_adjoint(e::Engine{X,S}, gp, fp) where {X,S}
  N = insize(e.core)
  if S === :single
    adjoint_core!(unsafe_wrap(Array, fp, N), e.core, unsafe_wrap(Array, gp, e.J))
  elseif S === :lead || S === :trail
    NFFT.transform_adjoint!(Val(S), (N=(e.L1, N..., e.L3), dims=2:length(N)+1), e.core,
                            unsafe_wrap(Array, fp, (e.L1, N..., e.L3)), unsafe_wrap(Array, gp, (e.L1, e.J, e.L3)))
  else
    each_slice(e) do c, t, a, b
      slice!(c, e, t, a, b, fp, gp, false)
    end
  end
  Cint(0)
end

# f(core, task, a, b) for all slices: one after the other on one threaded core, or in parallel, one
# single-threaded core per task taking the next slice when done.
function each_slice(f, e::Engine)
  if length(e.core) == 1
    for b in 1:e.L3, a in 1:e.L1
      f(e.core[1], 1, a, b)
    end
  else
    next = Threads.Atomic{Int}(1)
    NFFT.tasks(length(e.core)) do t
      while (s = Threads.atomic_add!(next, 1)) <= e.L1 * e.L3
        f(e.core[t], t, (s - 1) % e.L1 + 1, (s - 1) ÷ e.L1 + 1)
      end
    end
  end
end

# The transform (forward) or the adjoint of slice (a, b) on core c with the buffers of task t.
function slice!(c, e::Engine, t, a, b, fp::Ptr{A}, gp::Ptr{B}, forward::Bool) where {A,B}
  N = insize(c); M = prod(N); J = e.J; L1 = e.L1
  if L1 == 1
    fs = unsafe_wrap(Array, fp + sizeof(A) * M * (b - 1), N)
    gs = unsafe_wrap(Array, gp + sizeof(B) * J * (b - 1), J)
    forward ? trafo_core!(gs, c, fs) : adjoint_core!(fs, c, gs)
    return
  end
  f = unsafe_wrap(Array, fp, L1 * M * e.L3); g = unsafe_wrap(Array, gp, L1 * J * e.L3)
  fs = e.fbufs[t]; gs = e.gbufs[t]
  fo = a + L1 * M * (b - 1); go = a + L1 * J * (b - 1)
  if forward
    for m in 1:M; fs[m] = f[fo + L1 * (m - 1)]; end
    trafo_core!(gs, c, fs)
    for j in 1:J; g[go + L1 * (j - 1)] = gs[j]; end
  else
    for j in 1:J; gs[j] = g[go + L1 * (j - 1)]; end
    adjoint_core!(fs, c, gs)
    for m in 1:M; f[fo + L1 * (m - 1)] = fs[m]; end
  end
end

function plan_create(::Val{X}, ::Type{T}, D, N, J, k, reltol, nthreads, L1=1, L3=1) where {X,T}
  try
    @lock PLAN_LOCK with_dim(Int(D)) do d
      create(Val(X), T, d, N, Int(J), k, Float64(reltol),
             nthreads > 0 ? min(Int(nthreads), Threads.nthreads()) : default_threads(), Int(L1), Int(L3))
    end
  catch e
    Core.println(Core.stderr, e)
    Int64(-1)
  end
end

function trafo(::Val{X}, ::Type{T}, h, f, fhat) where {X,T}
  try
    lookup(e -> run_trafo(e, f, fhat), Val(X), T, h)
  catch
    Cint(-1)
  end
end

function adjoint(::Val{X}, ::Type{T}, h, fhat, f) where {X,T}
  try
    lookup(e -> run_adjoint(e, fhat, f), Val(X), T, h)
  catch
    Cint(-1)
  end
end

function plan_destroy(h)
  try
    x, _, t, _, _, _ = unhandle(h)
    with_transform(x) do xv
      with_type(t) do T
        with_registry(h, xv, T) do reg, i
          @lock REGISTRY_LOCK begin
            (1 <= i <= length(reg) && reg[i] !== nothing) || return Cint(-1)
            reg[i] = nothing
            Cint(0)
          end
        end
      end
    end
  catch
    Cint(-1)
  end
end

# The C functions: nfftjl_… in double precision, nfftjlf_… in single precision (nodes and reltol stay
# double); nfftjl_trafo etc. for the NFFT, nfftjl_rnfft_trafo etc. for the other transforms. A
# plan of any transform is released by nfftjl_plan_destroy (or nfftjlf_plan_destroy).
for (T, prefix) in ((Float64, "nfftjl_"), (Float32, "nfftjlf_")), X in TRANSFORMS
  P = X === :nfft ? prefix : string(prefix, X, "_")
  Tin, Tout = eltypes(X, T)
  @eval begin
    Base.@ccallable $(Symbol(P, :plan_create))(D::Cint, N::Ptr{Int64}, J::Int64, k::Ptr{Float64},
      reltol::Cdouble, nthreads::Cint)::Int64 = plan_create(Val($(QuoteNode(X))), $T, D, N, J, k, reltol, nthreads)
    Base.@ccallable $(Symbol(P, :plan_create_many))(D::Cint, N::Ptr{Int64}, J::Int64, k::Ptr{Float64},
      L1::Int64, L3::Int64, reltol::Cdouble, nthreads::Cint)::Int64 =
      plan_create(Val($(QuoteNode(X))), $T, D, N, J, k, reltol, nthreads, L1, L3)
    Base.@ccallable $(Symbol(P, :trafo))(h::Int64, f::Ptr{$Tin}, fhat::Ptr{$Tout})::Cint =
      trafo(Val($(QuoteNode(X))), $T, h, f, fhat)
    Base.@ccallable $(Symbol(P, :adjoint))(h::Int64, fhat::Ptr{$Tout}, f::Ptr{$Tin})::Cint =
      adjoint(Val($(QuoteNode(X))), $T, h, fhat, f)
  end
end
Base.@ccallable nfftjl_plan_destroy(h::Int64)::Cint = plan_destroy(h)
Base.@ccallable nfftjlf_plan_destroy(h::Int64)::Cint = plan_destroy(h)

# The library starts one Julia thread per CPU thread; plans use `nthreads` of them, by default
# NFFTJL_NUM_THREADS if set, otherwise all.
function default_threads()
  n = tryparse(Int, get(ENV, "NFFTJL_NUM_THREADS", ""))
  n === nothing ? Threads.nthreads() : clamp(n, 1, Threads.nthreads())
end
Base.@ccallable nfftjl_num_threads()::Cint = Cint(default_threads())
