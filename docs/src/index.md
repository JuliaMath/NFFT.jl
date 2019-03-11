# MPIReco.jl

*Julia package for the reconstruction of magnetic particle imaging (MPI) data*

## Introduction

This project provides functions for the reconstruction of MPI data. The project
is implemented in the programming language Julia and contains algorithms for

* [Basic Reconstruction](@ref) using a system matrix based approach
* [Multi-Patch Reconstruction](@ref) for data that has been acquired
  using a focus field sequence
* [Multi-Contrast Reconstruction](@ref)
* [Matrix-Compression Techniques](@ref)

Key features are

* Frequency filtering for memory efficient reconstruction. Only frequencies used
  during reconstructions are loaded into memory.
* Different solvers provided by the package [RegularizedLeastSquares.jl](https://github.com/tknopp/RegularizedLeastSquares.jl)
* High-level until low-level reconstruction providing maximum flexibility for
  the user
* Spectral leakage correction (implemented in
  [MPIFiles.jl](https://github.com/MagneticParticleImaging/MPIFiles.jl))

## Installation

Start julia and open the package mode by entering `]`. Then enter
```julia
add MPIReco
```
This will install the packages `MPIReco.jl` and all its dependencies. In particular
this will install the core dependencies [MPIFiles](https://github.com/MagneticParticleImaging/MPIFiles.jl.git) and [RegularizedLeastSquares](https://github.com/tknopp/RegularizedLeastSquares.jl.git).

## License / Terms of Usage

The source code of this project is licensed under the MIT license. This implies that
you are free to use, share, and adapt it. However, please give appropriate credit
by citing the project.

## Contact

If you have problems using the software, find mistakes, or have general questions please use
the [issue tracker](https://github.com/MagneticParticleImaging/MPIReco.jl/issues) to contact us.

## Contributors

* [Tobias Knopp](https://www.tuhh.de/ibi/people/tobias-knopp-head-of-institute.html)
* [Martin Möddel](https://www.tuhh.de/ibi/people/martin-moeddel.html)
* [Patryk Szwargulski](https://www.tuhh.de/ibi/people/patryk-szwargulski.html)
