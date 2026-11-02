# C

!!! note "Experimental"
    `libnfftjl` is NFFT.jl compiled into a shared library with `juliac --trim`. It carries its
    Julia runtime; programs that use it need no Julia installation. It covers the NFFT, RNFFT,
    NFCT, NFST and NNFFT and their adjoints in 1–3 dimensions, with batches, in single and double
    precision (functions `nfftjl_rnfft_…`, `nfftjl_nfct_…`, `nfftjl_nfst_…`, `nfftjl_nnfft_…`, see
    the header below); the table on the [home page](index.md) lists what the Julia package offers
    beyond that. The interface may still change.

## Building

Building needs Julia ≥ 1.12:

```
git clone https://github.com/JuliaMath/NFFT.jl && cd NFFT.jl
libnfftjl/build.sh
```

The result is `libnfftjl/build/`: `lib/libnfftjl.so` (`.dylib` on macOS) next to the Julia runtime
it needs, and the artifacts (FFTW) in `share/`. Keep the directory together. A program compiles
against the header `libnfftjl/include/nfftjl.h`:

```
cc -O2 prog.c -Ilibnfftjl/include -Llibnfftjl/build/lib -lnfftjl -lm -Wl,-rpath,$PWD/libnfftjl/build/lib
```

The library starts one Julia thread per CPU thread; a plan uses `nthreads` of them, by default
the value of the environment variable `NFFTJL_NUM_THREADS` or one per performance core. It is built with Julia's signal
handling, which the garbage collector needs to stop threads, and for one CPU target (see
`libnfftjl/README.md`).

## A first transform

The NFFT evaluates ``\hat f_j = \sum_{\bm l} f_{\bm l} \, e^{-2\pi i \bm l \cdot \bm k_j}`` at nodes
``\bm k_j \in [-1/2, 1/2)^D`` for the equidistant array ``f`` (see
[Mathematical Background](@ref)); the adjoint is the sum with the opposite sign. This program
computes both and checks the first values against the direct sum:

```c
{{include libnfftjl/examples/getting_started.c}}
```

## Interface

```c
{{include libnfftjl/include/nfftjl.h}}
```

`nfftjl_plan_create(D, N, J, k, reltol, nthreads)`
: A plan for `D` = 1, 2, 3 dimensions, the array size `N[0..D-1]` and `J` nodes `k`
  (``D \times J``, column major: node `j` is `k[D*j]`, …, `k[D*j + D-1]`, entries in
  ``[-1/2, 1/2)``). `reltol` is the requested relative error: window, window width and
  oversampling factor are chosen for it at the lowest predicted cost (see
  [Accuracy and Parameters](@ref)). As in Julia, a `reltol` that no parameters reach is an error
  rather than a less accurate plan. The library is compiled for window widths ``W = 4, \dots, 16``
  (single precision: ``4, \dots, 8``), which covers the tolerances that the planner accepts in each
  precision; the NNFFT chooses both of its windows within this range. `nthreads = 0` uses
  `nfftjl_num_threads()`. Returns a handle, or −1 on error (`nfftjl_last_error` gives the message).
  The nodes are copied.

`nfftjl_trafo(p, f, fhat)`, `nfftjl_adjoint(p, fhat, f)`
: The NFFT of `f` (``N_0 \times \dots \times N_{D-1}`` complex values, column major: element
  ``(i_0, \dots, i_{D-1})`` at ``i_0 + N_0 (i_1 + N_1 (i_2 \dots))``) into `fhat` (`J` values), and
  the adjoint. Index ``i_d`` stands for the frequency ``l_d = i_d - \lfloor N_d/2 \rfloor``. Return
  0, or −1 for an unknown handle.

`nfftjl_plan_create_many(D, N, J, k, L1, L3, reltol, nthreads)`
: A plan for many transforms with the same nodes: the data are ``L_1 \times N_0 \times \dots \times
  N_{D-1} \times L_3`` and ``L_1 \times J \times L_3`` (column major), transformed along the
  dimensions in between. `nfftjl_plan_create` is the case ``L_1 = L_3 = 1``.

`nfftjl_plan_destroy(p)`
: Releases the plan.

`nfftjlf_plan_create`, `nfftjlf_plan_create_many`, `nfftjlf_trafo`, `nfftjlf_adjoint`, `nfftjlf_plan_destroy`
: The same in single precision: data in `float complex`; the nodes and `reltol` stay `double`, and
  grid positions are computed from the nodes in double precision. A plan belongs to one precision:
  the functions of the other return −1.

`nfftjl_nnfft_plan_create(D, J, k, N, y, reltol, nthreads)`, `nfftjl_nnfft_plan_create_many(D, J, k, N, y, L1, L3, reltol, nthreads)`
: The NNFFT ``\hat f_j = \sum_l f_l e^{-2\pi i \bm k_j \cdot \bm y_l}`` from `N` sources `y` to `J`
  targets `k` (both ``D \times`` count, column major, arbitrary real coordinates); data of
  ``L_1 \times N \times L_3`` and ``L_1 \times J \times L_3`` complex values. The cost grows with
  the product of the extents of `k` and `y`, not with `J` and `N` (see
  [Non-equidistant in Both Domains](@ref)). `nfftjl_nnfft_trafo`, `nfftjl_nnfft_adjoint` and the
  `nfftjlf_nnfft_…` functions as for the NFFT.

`nfftjl_num_threads()`
: The default number of threads of a plan: `NFFTJL_NUM_THREADS` if set, otherwise the number of
  performance cores (Apple Silicon) or of all CPU threads. The library has one thread per CPU
  thread, the upper limit of `nthreads`.

Row-major data of size ``N_0 \times \dots \times N_{D-1}`` is column-major data of the reversed
size: pass the sizes and the coordinates of each node in reversed order (the Python package does
this).

## Batches

One plan for many transforms with the same nodes is faster than a loop over single transforms:
the nodes are binned once, and the batch kernels of NFFT.jl process several transforms per pass
over the nodes.

```c
{{include libnfftjl/examples/batch.c}}
```

## Threads

- The Julia runtime starts on the first call into the library. Make that call from one thread
  before other threads call (`nfftjl_num_threads()` will do).
- Plans can be created and destroyed from any thread. One plan must not be executed by two
  threads at the same time.
- Separate plans can be executed by several threads at the same time, each with its own
  `nthreads`.
- The default number of threads of a plan (`nthreads = 0`) is `NFFTJL_NUM_THREADS` if set,
  otherwise the number of performance cores on Apple Silicon and of all CPU threads elsewhere. A
  1D transform runs its FFT in one piece per thread, so a thread on an efficiency core or on a
  core busy with other work holds it up (up to 2× slower with all 10 threads of an M1 Max); on a
  busy machine fewer threads than cores are faster.

## MATLAB and Octave

`bindings/matlab/` wraps the library for MATLAB (R2018a or later) and Octave: a MEX gateway
`nfftjl_mex.c` and a handle class `nfftjl.Plan` for all transforms of the library, in single and
double precision. Arrays are column major as in C, so data and nodes are passed as they are: nodes
`k` as a ``D \times J`` matrix in ``[-1/2, 1/2)`` (NFCT, NFST: ``[0, 1/2]``), coefficients of
size `N`.

```matlab
build_nfftjl                                  % once: compiles nfftjl_mex against libnfftjl/build/lib
p = nfftjl.Plan(rand(2, 1e4) - 0.5, [256 256], 'reltol', 1e-8);
fhat = p.trafo(randn(256) + 1i * randn(256));
f = p.adjoint(fhat);
q = nfftjl.Plan(k, [64 64], 'kind', 'nfct', 'single', true);   % 'rnfft', 'nfct', 'nfst', 'nnfft'
```

The options are `'kind'`, `'reltol'` (default ``10^{-8}``), `'single'` and `'nthreads'`; the
NNFFT takes the sources instead of the size, `nfftjl.Plan(k, y, 'kind', 'nnfft')`. Inputs are
converted to the plan's precision. `test_nfftjl.m` compares all transforms with the direct sums
(`octave --eval "run('bindings/matlab/test_nfftjl.m')"`). Tested with Octave 11.3 on macOS.

