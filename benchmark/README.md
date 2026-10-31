# Benchmarks of NFFT.jl

Compares NFFT.jl with FINUFFT, ducc, NonuniformFFTs.jl and NFFT.jl 0.14, and NFFT.jl with any of
its own git revisions. The results in `results/` are the data of the
[performance page](https://JuliaMath.github.io/NFFT.jl/dev/performance/) of the documentation.

```
julia benchmark/bench.jl list                       # suites and libraries
julia benchmark/bench.jl run fast                   # NFFT.jl alone, ≈ 15 s
julia benchmark/bench.jl run quick                  # NFFT vs FINUFFT vs ducc, ≈ 1 min
julia benchmark/bench.jl run quick --libs NFFT --dims 2 --tols 1e-8
julia benchmark/bench.jl ab HEAD~1                  # working tree against a revision (suite fast), ≈ 1 min
julia benchmark/bench.jl compare out/a.csv out/b.csv
julia benchmark/bench.jl run all --save             # all suites of the documentation, ≈ 15 min
julia benchmark/bench.jl run lite --save            # the figures with every other point, 2 rounds, ≈ 10 min
julia benchmark/bench.jl report                     # figures and explorer data for the docs
julia benchmark/bench.jl check                      # worker's FFTW threads against a fresh process
```

Run from the repository root with any Julia (the orchestrator uses the standard library only).
Options: `--libs`, `--dims`, `--tols`, `--T Float32`, `--threads 1,8`, `--budget` (seconds of
repeated calls per measurement, default 0.1), `--rounds` (`ab`; `run`: repeat all tasks, each round
in a new random order of cases and of the tasks within a case, with the round and its load average
in the CSV), `--save`.

## How it works

Each library runs in a worker process with its own environment (`envs/<lib>/Project.toml`,
instantiated on first use): competitors never become dependencies of NFFT.jl, libraries with
conflicting dependencies (NonuniformFFTs.jl needs AbstractNFFTs 0.9, NFFT.jl 0.14) can be compared,
and each library gets exactly the requested number of threads. `NFFT@<rev>` is NFFT.jl at a git
revision, checked out as a worktree in `.cache/`.

- `src/suites.jl`: the suites, each a list of tasks (library, transform, size, nodes, batch,
  precision, threads, tolerance).
- `worker/problems.jl`: nodes, data and an exact reference (NDFT on 64 sampled outputs), generated
  deterministically from the task, so every process sees the same problem.
- `worker/worker.jl`: plan (3 times, minimum), direct and adjoint transform (minimum and median of
  repeated calls within the budget), relative error against the reference, allocations, the
  parameters the library chose.
- `worker/fftw.jl`: forgets FFTW's wisdom before every plan, in every loaded FFTW, after
  initialising FFTW's threads (a planner created before `fftw_init_threads` lacks the threaded
  Cooley-Tukey solvers). `bench.jl check` times threaded FFTW and FINUFFT plans after this call
  against a fresh process; run it after changing the worker.
- `worker/adapters/<lib>.jl`: maps the convention of NFFT.jl (nodes in [-1/2, 1/2), modes in
  increasing order, e^{-2πi l·k} for the direct transform) onto each library. A wrong mapping shows
  up as an error of order one.
- `src/Benchmarks.jl`: orchestration, CSV files, `compare` and `ab`.
- `src/report.jl`: figures (`docs/src/assets/benchmarks/*.svg`) and `data.js` for the explorer on
  the performance page; the CPU figures for the reference machine (`report [machine]`, default the
  current one) and with the machine as suffix for the others. `src/report_platforms.jl`: the
  figures across platforms (overview of the ratios, CUDA and Metal, NFFT.jl on four devices,
  delivered error).

Results go to `out/<time>-<suite>.csv` (not versioned); `--save` also writes
`results/<machine>/<suite>.csv`, the data of the documentation. The CSV header records machine,
Julia version, NFFT.jl commit and command; each row the version and library file of the library.

Results that were not written by `--save` on their machine, with their origin:

| File | Origin |
|---|---|
| `apple-m1-max/metal.csv` | `out/2026-10-29_081023-metal.csv` (`run metal`, NFFT.jl 676d13e with changes committed as 7105dea) |
| `apple-m1-max/contract.csv` | `out/2026-10-31_023233-contract.csv` (`run contract`, NFFT.jl 00b3c45) |
| `amd-epyc-7763/*.csv` | `run lite` (2 rounds) on a shared AMD EPYC 7763 server, 8 threads, 2026-10-31, NFFT.jl 4e0df8a; FINUFFT 2.5.1 from its JLL, ducc0 0.42 built with `-march=native`. The run overlapped with GPU runs in the same clone; its rows were sorted back by suite and library. Header lines added by hand. Provisional. |
| `rtx-a6000/cuda.csv` | `run cuda` and `run cuda --sizes big` on GPU 5 of the same server, 2026-10-31: the rows of NFFT CUDA from a run with the unrolled interpolation kernel and inlined window weights (on top of 4e0df8a, before 3588e7a), the rows of NonuniformFFTs CUDA from the first runs (two rounds at the default sizes, one at the big sizes). Header lines added by hand. Provisional. |
| `rtx-a6000/cufinufft*.csv` | `python benchmark/cufinufft_bench.py [big]` on the same GPU, cuFINUFFT 2.5.1 with CuPy 14.2; header lines added by hand. |

Every library uses its defaults apart from the tolerance and the number of threads; FFTW plans use
`ESTIMATE`. The libraries interpret the tolerance differently, so compare time against the
measured error (`pareto`), not against the tolerance. In `pareto` NFFT.jl also runs half a decade
below each tolerance, in `sizes` and `density` the others run at the tolerance and NFFT.jl on half
decades around it; the report compares at equal error by
interpolating NFFT.jl's time at the error each library delivers. NonuniformFFTs.jl has no tolerance
parameter; its half support comes from the tolerance as in NFFT.jl 0.14. ducc has no batched
transform; batches loop over the slices.

## Apple GPUs

`run metal` compares NFFT.jl on Metal (`storage = MtlArray`) with
[mlx-nufft](https://github.com/martinlachaine/mlx-nufft) (Python, MLX), both in Float32 with the
data on the GPU. mlx-nufft runs through PythonCall in a Python of your choice:

```
python3 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-nufft
MLX_NUFFT_PYTHON=$PWD/.venv-mlx/bin/python julia benchmark/bench.jl run metal
```

Without `MLX_NUFFT_PYTHON` its tasks are skipped. GPU timings include a synchronisation after
every call.

## Nvidia GPUs

`run cuda` compares NFFT.jl on CUDA (`storage = CuArray`) with NonuniformFFTs.jl on CUDA, data on
the GPU, Float64 and Float32, tolerance sweep, J = |N|; `--sizes big` for 2^22, 2048², 160³ instead
of 2^18, 512², 64³. cuFINUFFT runs the same problems from Python (`pip install cufinufft cupy-cuda12x`)
and writes its own CSV:

```
CUDA_VISIBLE_DEVICES=5 julia benchmark/bench.jl run cuda --sizes big
CUDA_VISIBLE_DEVICES=5 python benchmark/cufinufft_bench.py big > cufinufft-big.csv
CUDA_VISIBLE_DEVICES=5 julia benchmark/bench.jl run batch --libs "NFFT CUDA" --T Float64,Float32
CUDA_VISIBLE_DEVICES=5 python benchmark/cufinufft_bench.py batch > cufinufft-batch.csv
```

The cuFINUFFT reference uses the nodes rounded to the precision the library receives them in.

## Native builds of the competitors

By default FINUFFT and ducc come from their release binaries (JLL packages). To benchmark builds
with native CPU flags, point the JLL packages to them in `envs/finufft/LocalPreferences.toml` and
`envs/ducc/LocalPreferences.toml` (not versioned):

```toml
[finufft_jll]
libfinufft_path = "/path/to/finufft/build/src/libfinufft.dylib"
```

```toml
[ducc0_jll]
libducc_julia_path = "/path/to/libducc_julia.dylib"
```

The library that was loaded is recorded in each row (`lib_library`).

"FINUFFT 2.5" (`envs/finufft25`) is a second FINUFFT for the suite `decomposition`: a build of
v2.5.1 next to the current one, with its path in `envs/finufft25/LocalPreferences.toml`. Its
CMake links the shared library with the OpenMP flags as one argument, which clang rejects; link
the objects by hand:

```
cmake -S finufft-2.5.1 -B build -DCMAKE_BUILD_TYPE=Release -DFINUFFT_ARCH_FLAGS=-mcpu=native -DFINUFFT_STATIC_LINKING=OFF
cmake --build build                    # fails at the last link step
cd build/src && c++ -dynamiclib -o libfinufft.dylib -install_name @rpath/libfinufft.dylib \
  CMakeFiles/finufft.dir/*.o common/CMakeFiles/finufft_common.dir/*.o \
  -L/opt/homebrew/opt/fftw/lib -lfftw3 -lfftw3f -lfftw3_omp -lfftw3f_omp -L/opt/homebrew/opt/libomp/lib -lomp
```

## Suite `decomposition`

Splits a speed ratio NFFT.jl : X into a parameter factor T_jl(rule X)/T_jl(own) and an
implementation factor T_X/T_jl(rule X). `src/rules.jl` gives, for FINUFFT 2.5.1 and 2.6-dev,
cuFINUFFT 2.5.1, ducc 0.42, NonuniformFFTs.jl 0.9.8, NFFT3 and mlx-nufft 0.3.3, the window, W and
grid the library would choose (each rule cites its source); NFFT.jl runs with them through a
`CustomWindow` and the option `gridsize` (the grid exactly, also with factors 7 and 11 as ducc's). A
task with `rule` records the rule's parameters (`rule_*`), what the library reports (`rep_*`: the
debug output of FINUFFT, ducc's verbose report, the parameters of NonuniformFFTs.jl, or the grid
NFFT.jl built) and their differences (`rule_dev`). A revision of NFFT.jl without `gridsize` rounds
such a grid up to the next one from {2, 3, 5} (`grid_rounded`). `tol_used` is the tolerance a library
got after its adapter raised or clamped it. NFFT.jl with its own choice runs on half decades of
tolerance, so that its time can be interpolated at the error each original delivers. FFT planning
counts as implementation: every library plans with its own default flag and threading (FINUFFT
ESTIMATE, NonuniformFFTs.jl MEASURE, NFFT.jl ESTIMATE with its own threaded passes, also under a
rule). Suite `fftflags` runs the originals on FFTW with both flags on 8 of the cases (`fft_flag`);
every plan starts without FFTW wisdom, so that a measured plan does not speed up later ones in the
same worker.

Every row of every suite has `grid` and `grid_adj`, the oversampled grid Ñ per dimension of the trafo
and the adjoint: from the plan (NFFT.jl, NonuniformFFTs.jl) or from what the library reports (FINUFFT,
ducc; one more plan outside the timing).

```
julia benchmark/bench.jl run decomposition --rounds 5     # 711 tasks per round
julia benchmark/bench.jl run fftflags                      # 96 tasks
```

## Suite `contract`

Delivered error against the requested tolerance (`err_trafo`, `err_adj` against `tol`) for every CPU
library: uniform random nodes and nodes on a regular lattice (`cartesian`, the first J points of an
n^D lattice from −1/2 with spacing 1/n), the tolerance sweep of `pareto`, Float64 and Float32.
