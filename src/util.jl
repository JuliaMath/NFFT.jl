# A threaded loop over an indexable collection, or a plain loop when `n == 1`: scheduling the tasks
# costs ≈ 10 µs, as much as a whole small transform on a single-threaded core. The threaded loop
# runs on `threaded_foreach`, not `Threads.@threads`: leaving a `@threads` region takes the libuv
# lock, which deadlocks against a garbage collection when foreign threads (a C or Python caller of
# the compiled library) run transforms concurrently. A schedule argument (`:dynamic`) is accepted
# and ignored, the elements are always handed out dynamically.
macro maybe_threads(n, args...)
  loop = args[end]
  (loop isa Expr && loop.head === :for) || error("@maybe_threads expects a for loop")
  var, itr = loop.args[1].args
  esc(quote
    if $n == 1
      $loop
    else
      $(GlobalRef(@__MODULE__, :threaded_foreach))($itr) do $var
        $(loop.args[2])
      end
    end
  end)
end

# f(x) for all x in xs, on min(length(xs), nthreads()) tasks that take the next element when done.
function threaded_foreach(f, xs)
  next = Threads.Atomic{Int}(1)
  tasks(min(length(xs), Threads.nthreads())) do _
    while (i = Threads.atomic_add!(next, 1)) <= length(xs)
      f(xs[i])
    end
  end
  nothing
end

# Split a range into at most k contiguous chunks of nearly equal length.
chunks(r, k) = isempty(r) ? typeof(r[1:0])[] :
  [r[i:min(i + cld(length(r), k) - 1, end)] for i in 1:cld(length(r), k):length(r)]

# Smallest even n′ ≥ n whose prime factors are 2, 3, 5 and 7 (FFT-friendly grid sizes).
function smooth_size(n)
  m = n + isodd(n)
  while !issmooth(m)
    m += 2
  end
  m
end
function issmooth(n)
  for p in (2, 3, 5, 7)
    while n % p == 0
      n ÷= p
    end
  end
  n == 1
end

# Private tile buffers of the resampling tasks, one per task (at most 4·nthreads tasks, one for a
# single-threaded core), allocated with the plan so that transforms do not allocate them.
task_buffers(::Type{C}, sz, nthreads) where {C} = [zeros(C, sz) for _ in 1:(nthreads == 1 ? 1 : 4nthreads)]

# Run f(1), …, f(n) as tasks; inline for n = 1, so that single-threaded cores called from a
# parallel loop over slices spawn nothing.
function tasks(f, n)
  if n == 1
    f(1)
  else
    ts = Vector{Task}(undef, n)
    for it in 1:n
      ts[it] = Threads.@spawn f(it)
    end
    foreach(wait, ts)
  end
  nothing
end

# The part r of the last dimension (selectdim with a constant dimension, without its error path).
lastdim_view(A::AbstractArray{T,N}, r) where {T,N} = view(A, ntuple(_ -> Colon(), Val(N - 1))..., r)

function threaded_copy!(dst, src, nthreads)
  D = ndims(dst)
  @maybe_threads nthreads for r in chunks(axes(dst, D), nthreads)
    copyto!(lastdim_view(dst, r), lastdim_view(src, r))
  end
  dst
end

function threaded_fill!(a, v, nthreads)
  @maybe_threads nthreads for s in chunks(eachindex(a), 4nthreads)
    @inbounds for i in s
      a[i] = v
    end
  end
  a
end

# f(core, a, b) for all slices (a, b) ∈ 1:L1 × 1:L3, each task with its own core, slices taken
# dynamically
function slices_parallel(f, cores, L1, L3)
  next = Threads.Atomic{Int}(1)
  CI = CartesianIndices((L1, L3))
  tasks(length(cores)) do it
    c = cores[it]
    while (s = Threads.atomic_add!(next, 1)) <= L1 * L3
      f(c, CI[s][1], CI[s][2])
    end
  end
end

# Batch slices run in parallel, one single-threaded core per task, when there are many of them or
# when they are small; otherwise one after the other on a threaded core (or in the batch kernels).
# 8 threads, M1 Max: OCT-like 1D (N = 2048, J = 2048, 512 slices) 2–4× faster than the batch
# kernels; 2D 256², J = 2·10⁵, 8 slices: adjoint 1.6× slower.
use_slices(L, work, nthreads) = nthreads > 1 && (L >= 4nthreads || (L >= nthreads && work <= 2^15))

single_cores(c, L) = [copy_core(c, 1) for _ in 1:min(c.nthreads, L)]
