# NFFT Benchmarks

This folder contains the accuracy and performance scripts for the NFFT. 
To run the scripts additional Julia packages, e.g. for plotting are required.
This folder contains a `Project.toml` can be used by running Julia from within
the folder with
```
julia --project=.
```
When you do this the first time you need to to the package mode run `instantiate`.

## Accuracy

The accuracy script is run by
```
julia> include("accuracy.jl")
```
It creates an error plot for various ``m`` and ``\sigma`` that is stored in `../docs/src/assets`. The results can be also read from the CSV being generated.

## Performance

The performance script is run by
```
julia> include("performance.jl")
```
It will spawn new julia processes with different numbers of threads. Again the results are stored in a CSV file and plotted into a pdf that is stored in `../docs/src/assets`.

## Other

The benchmark suite takes a long time and is not useful for profiling. Therefore the script `performance_simple.jl` contains a subset of the suite, which allows to invoke the profiler:

```
using ProfileView
include("performance_simple.jl")

# run once
nfft_performance_simple(N=1024, M=1024^2, m=4, threading=false, pre=NFFT.LINEAR, ctor=NFFTPlan)

# call the profiler
@profview nfft_performance_simple(N=1024, M=1024^2, m=4, threading=false, pre=NFFT.LINEAR, ctor=NFFTPlan)

# instead run with FINUFFT
nfft_performance_simple(N=1024, M=1024^2, m=4, threading=false, pre=NFFT.LINEAR, ctor=FINUFFTPlan)
```
