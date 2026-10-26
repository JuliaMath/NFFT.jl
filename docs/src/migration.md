# Migrating from NFFT.jl 0.x

NFFT.jl 1.0 is a new implementation with a cleaner interface. Most code that only uses
`plan_nfft`, `*`, `mul!`, `adjoint` and `reltol` runs unchanged. The changes:

| NFFT.jl 0.x | NFFT.jl 1.0 |
|:--|:--|
| `m`, `σ` | `W = 2m` (window width; odd widths allowed), `σ` |
| `window = :kaiser_bessel`, … | `window = KaiserBessel()`, `ExpSemicircle()`, `CustomWindow(φ)`; default `:auto` |
| `precompute = POLYNOMIAL / TENSOR / LINEAR / FULL` | removed; the window is always evaluated as a piecewise polynomial |
| `blocking`, `sortNodes`, `blockSize` | removed (always tiled); `tilesize` as an expert option |
| `storeDeconvolutionIdx`, `LUTSize` | removed |
| `fftflags = FFTW.MEASURE` | `fft = FFTWBackend(flags=FFTW.MEASURE)` |
| `timing` keyword, `TimingStats` | removed; time the [stages](@ref "Stages of the NFFT") instead |
| positional array type `plan_nfft(Array, k, N)` | `storage = Array` |
| `NFFT._use_threads[] = false` | `nthreads = 1` |
| `convolve!`, `convolve_transpose!` | `resampling!`, `resampling_adjoint!` |
| `deconvolve!`, `deconvolve_transpose!` | `resampling_correction!`, `resampling_correction_adjoint!` |
| `p.Ñ` | `oversampled_size(p)` |

Behaviour that changed:

- **Unknown options are errors.** In 0.x, misspelled or unsupported keywords could be ignored silently.
- **`reltol` is met.** The plan chooses window, width and oversampling from tables of measured
  errors; `accuracy(p)` reports the predicted error.
- **Nodes are checked.** Nodes outside ``[-1/2, 1/2]`` or not finite throw an `ArgumentError`.
- **`dims`** accepts contiguous ranges of dimensions, not only a single dimension, and uses batch
  kernels.
- **GPU plans** are not yet available in 1.0.
