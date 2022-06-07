
# Mathematical Background

## NDFT

We next define the non-equidistant discrete Fourier transform (NDFT) that corresponds to the ordinary DFT. Let ``\bm{N} \in (2\mathbb{N})^d`` with ``d \in \mathbb{N}`` be the dimension of the ``d``-dimensional Fourier coefficients ``\hat{f}_{\bm{k}}, k \in I_{\bm{N}}``. It is defined on the index set
```math
I_{\bm{N}} := \left\lbrace \pmb{k} \in \mathbb{Z}^d: -\frac{N_i}{2} \leq \bm{k}_i \leq \frac{N_i}{2}-1, i=1,2,\ldots,d \right\rbrace
```
and thus represents the same data that would be considered for an ordinary DFT. The NDFT is  defined as
```math
  	f(\bm{x}_j) := \sum_{ \bm{k} \in I_{\bm{N}}} \hat{f}_{\bm{k}} \, \mathrm{e}^{-2\pi\mathrm{i}\,\bm{k}\cdot\bm{x}}
```
where ``\bm{x}_j \in \mathbb{T}^d, j=1,\dots, M`` with ``M \in \mathbb{N}`` are the nonequidistant sampling nodes and ``\mathbb{T} := [1/2,1/2)`` is the torus and ``f`` is the ``d``-dimensional trigonometric polynomial associated with the Fourier coefficients ``\hat{f}_{\bm{k}}``.

The direct NDFT operator has an associated adjoint that can be formulated as
```math
	\hat{g}_{\bm{k}} = \sum_{j = 1}^{M} f(\bm{x}_j) \, \mathrm{e}^{2 \pi \mathrm{i} \, \bm{k} \cdot \bm{x}_j}, \bm{k} \in I_{\bm{N}}.
```
We note that in general the adjoint NDFT is not the inverse.

!!! note
    The indices in the index set ``I_{\bm{N}}`` are centered around zero, which is the usual definition of the NFFT. In contrast the indices for the DFT usually run from ``1,\dots,N_d``. This means an `fftshift` needs to be applied to change from one representation to the other.

!!! note
    In the literature the NFFT has different names. Often it is called NUFFT, and in the MRI context gridding. The NFFT is sometimes divided into three types of which type 1 corresponds to the adjoint NFFT, type 2 corresponds to the direct NFFT and type 3 corresponds to the NNFFT that we discuss later. Further information on this alternative formulation can be found [here](https://finufft.readthedocs.io/en/latest/math.html). 

## Matrix-Vector Notation

The NDFT can be written as
```math
 \bm{f} = \bm{A} \hat{\bm{f}}
```
where
```math
\begin{aligned}
 \bm{f} &:= \left( f(\bm{x}_j) \right)_{j=1}^{M} \in \mathbb{C}^\mathbf{N} \\
 \hat{\bm{f}} &:= \left( \hat{f}_{\bm{k}} \right)_{\bm{k} \in I_\mathbf{N}} \in \mathbb{C}^M\\
  \bm{A} &:=  \left( \mathrm{e}^{2 \pi \mathrm{i} \, \bm{k} \cdot \bm{x}_j} \right)_{j=1,\dots,M; \bm{k} \in I_{\mathbf{N}}} \in \mathbb{C}^{M \times \mathbf{N}}
\end{aligned}
```
The adjoint can than be written as
```math
 \hat{\bm{g}} = \bm{A}^{\mathsf{H}}  \bm{f}
```
where ``\hat{\bm{g}} \in \mathbb{C}^\mathbf{N}``.


## NFFT

The NFFT is an approximative algorithm that realizes the NDFT in just ``{\mathcal O}(N \log N + M)`` steps where ``N := \text{prod}(\bm{N})``. This is at the same level as the ordinary FFT with the exception that of the additional linear dependence on ``M`` which is unavoidable since all nodes need to be touched as least once.

The NFFT has two important parameters that influence its accuracy:
* the window width ``m \in \mathbb{N}``
* the oversampling factor ``\sigma \in \mathbb{R}`` with ``\sigma > 1``
From the later we can derive ``\bm{n} = \sigma \bm{N} \in (2\mathbb{N})^d``. As the definition indicates, the oversampling factor ``\sigma`` is usually adjusted such that ``\bm{n}`` consists of even integers.

The NFFT now approximates ``\bm{A}``by the product of three matrices
```math
\bm{A} \approx \bm{B} \bm{F} \bm{D}
```
where 
* ``\bm{F} \in \mathbb{C}^{\mathbf{n}\times \mathbf{n}}`` is the regular DFT matrix.
* ``\bm{D} \in \mathbb{C}^{\mathbf{n}\times \mathbf{N}}`` is a diagonal matrix that additionally includes zero filling and the fftshift. We call this the *deconvolve* matrix.
* ``\bm{B} \in \mathbb{C}^{M \times \mathbf{n}}`` is a sparse matrix implementing the discrete convolution with a window function ``\varphi``. We call this the *convolution* matrix.

The NFFT is based on the convolution theorem. It applies a convolution in the time/space domain, which is evaluated at equidistant sampling nodes. This convolution is then corrected in the Fourier space by division with the Fourier transform ``\psi`` of ``\varphi``. The adjoint NFFT matrix approximates ``\bm{A}^{\mathsf{H}}`` by

```math
\bm{A}^{\mathsf{H}} \approx \bm{D}^{\mathsf{H}} \bm{F}^{\mathsf{H}}  \bm{B}^{\mathsf{H}} 
```

Implementation-wise, the matrix-vector notation illustrates that the NFFT consists of three independent steps that are performed successively. 
* The multiplication with ``\bm{D}`` is a scalar multiplication with the input-vector plus the shifting of data, which can be done inplace.
* The FFT is done with a high-performance FFT library such as the FFTW.
* The multiplication with ``\bm{B}`` needs to run only over a subset of the indices and is the most challenging step.
Since in practice the multiplication with ``\bm{B}`` is also the most expansive step, an NFFT library needs to pay special attention to optimizing it appropriately.

## Directional NFFT

In many cases one not just needs to apply a single NFFT but needs to apply many on different data. This leads us to the directional NFFT. The directional NFFT is defined as

```math
  	f_{\bm{l},j,\bm{r}} := \sum_{ \bm{k} \in I_{\bm{N}_\text{sub}}} \hat{f}_{\bm{l},\bm{k},\bm{r}} \, \mathrm{e}^{-2\pi\mathrm{i}\,\bm{k}\cdot\bm{x}}
```

where now ``(\bm{l}, \bm{k}, \bm{r}) \in I_\mathbf{N}`` and ``\bm{N}_\text{sub}`` is a subset of ``\bm{N}``. The transform thus maps a ``D``-dimensional tensor ``\hat{f}_{\bm{l},\bm{k},\bm{r}}`` to an ``R``-dimensional tensor ``f_{\bm{l},j,\bm{r}}``. ``\bm{N}_\text{sub}`` is thus a vector of length ``D-R+1`` The indices ``\bm{l}`` and ``\bm{r}`` can also have length zero. Thus, for ``R=1``, the conventional NFFT arises as a special case of the directional.

!!! note
    The directional NFFT can also be considered to be a slicing of a tensor with subsequent application of a regular NFFT. But the aforementioned formulation can be used to implement a much more efficient algorithm than can be achieved with slicing.

## NNDFT / NNFFT

*Under construction*