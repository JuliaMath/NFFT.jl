# C

!!! note "Experimental"
    `libnfftjl` is NFFT.jl compiled into a shared library with `juliac --trim`. It carries its
    Julia runtime; programs that use it need no Julia installation. It covers the NFFT and its
    adjoint in 1–3 dimensions in single and double precision; the table on the [home page](index.md) lists
    what the Julia package offers beyond that. The interface may still change.

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
the value of the environment variable `NFFTJL_NUM_THREADS` or all. It is built with Julia's signal
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
  [Accuracy and Parameters](@ref)). `nthreads = 0` uses `nfftjl_num_threads()`. Returns a handle,
  or −1 on error. The nodes are copied.

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

`nfftjl_num_threads()`
: The default number of threads of a plan: `NFFTJL_NUM_THREADS` if set, otherwise all CPU
  threads, which is also the upper limit of `nthreads`.

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
