
# Mathematical Background

On this page we give a brief overview of the mathematical background of the NFFT. For a full background including a derivation of the NFFT we refer to the [NFFT.jl paper](http://LinkToArXiv.com).

!!! note
    In the literature the NFFT has different names. Often it is called NUFFT, and in the MRI context gridding.

## NDFT

We first define the non-equidistant discrete Fourier transform (NDFT) that corresponds to the ordinary DFT. Let ``\bm{N} \in (2\mathbb{N})^d`` with ``d \in \mathbb{N}`` be the size of the ``d``-dimensional  equidistantly sampled signal ``f_{\bm{n}}, k \in I_{\bm{N}}``. ``f`` can for instance be a time or spatial domain signal. The signal is indexed using the multi-index set
```math
I_{\bm{N}} := \left\lbrace \pmb{n} \in \mathbb{Z}^d: -\frac{N_i}{2} \leq \bm{k}_i \leq \frac{N_i}{2}-1, i=1,2,\ldots,d \right\rbrace
```
and thus represents a regular sampling that would also be considered when applying an ordinary DFT. The NDFT now maps from the equidistant domain to the non-equidistant domain and is defined as
```math
  	\hat{f}_j := \sum_{ \bm{n} \in I_{\bm{N}}} f_{\bm{n}} \, \mathrm{e}^{-2\pi\mathrm{i}\,\bm{n}\cdot\bm{k}_j}, \quad j=1,\dots, J \qquad \textit{(equidistant to non-equidistant)}
```
where ``\bm{k}_j \in \mathbb{T}^d, j=1,\dots, J`` with ``J \in \mathbb{N}`` are the nonequidistant sampling nodes, ``\mathbb{T} := [1/2,1/2)`` is the torus and ``\hat{f}`` is the ``d``-dimensional frequency domain signal.

The direct NDFT has an associated adjoint that can be formulated as
```math
	y_{\bm{n}} = \sum_{j = 1}^{J} f_j \, \mathrm{e}^{2 \pi \mathrm{i} \, \bm{n} \cdot \bm{k}_j}, \bm{n} \in I_{\bm{N}} \qquad \textit{(non-equidistant to equidistant)}.
```
We note that in general the adjoint NDFT is not the inverse NDFT. 

!!! note
    The indices in the index set ``I_{\bm{N}}`` are centered around zero, which is the common definition of the NDFT. In contrast the indices of the DFT usually run from ``1,\dots,N_d``. This means an `fftshift` needs to be applied to change from one representation to the other.

!!! note
     Instead of the direct/adjoint NDFT terminology, there is an alternative classification that consists of three types. Type 1 corresponds to the adjoint NDFT, type 2 corresponds to the direct NDFT and type 3 corresponds to the NNDFT that has non-equidistant samples in both domains. Further information on this alternative formulation can be found [here](https://finufft.readthedocs.io/en/latest/math.html). 

## Matrix-Vector Notation

The NDFT can be written as
```math
 \hat{\bm{f}} = \bm{A} \bm{f}
```
where
```math
\begin{aligned}
 \hat{\bm{f}} &:= \left( \hat{f}(\bm{k}_j) \right)_{j=1}^{J} \in \mathbb{C}^J \\
 \bm{f} &:= \left( f_{\bm{n}} \right)_{\bm{n} \in I_\mathbf{N}} \in \mathbb{C}^\mathbf{N}\\
  \bm{A} &:=  \left( \mathrm{e}^{2 \pi \mathrm{i} \, \bm{n} \cdot \bm{k}_j} \right)_{j=1,\dots,J; \bm{n} \in I_{\mathbf{N}}} \in \mathbb{C}^{J \times \mathbf{N}}
\end{aligned}
```
The adjoint can then be written as
```math
 \bm{y} = \bm{A}^{\mathsf{H}}  \hat{\bm{f}}
```
where ``\bm{y} \in \mathbb{C}^\mathbf{N}``.


## NFFT

The NFFT is an approximative algorithm that realizes the NDFT in just ``{\mathcal O}(|\bm{N}| \log |\bm{N}| + J)`` steps where ``|\bm{N}| := \text{prod}(\bm{N})``. This is at the same level as the ordinary FFT with the exception that of the additional linear term ``J``, which is unavoidable since all nodes need to be touched as least once.

The NFFT has two important parameters that influence its accuracy:
* the window size parameter ``m \in \mathbb{N}``
* the oversampling factor ``\sigma \in \mathbb{R}`` with ``\sigma > 1``
From the later we can derive ``\tilde{\bm{N}} = \sigma \bm{N} \in (2\mathbb{N})^d``. As the definition indicates, the oversampling factor ``\sigma`` is usually adjusted such that ``\tilde{\bm{N}}`` consists of even integers.

The NFFT now approximates ``\bm{A}`` by the product of three matrices
```math
\bm{A} \approx \bm{B} \bm{F} \bm{D}
```
where 
* ``\bm{F} \in \mathbb{C}^{\tilde{\mathbf{N}}\times \tilde{\mathbf{N}}}`` is the regular DFT matrix.
* ``\bm{D} \in \mathbb{C}^{\tilde{\mathbf{N}}\times \mathbf{N}}`` is a diagonal matrix that additionally includes zero filling and the fftshift. We call this the *deconvolution* matrix.
* ``\bm{B} \in \mathbb{C}^{M \times \tilde{\mathbf{N}}}`` is a sparse matrix implementing the discrete convolution with a window function ``\hat{\varphi}``. We call this the *convolution* matrix.

The NFFT is based on the convolution theorem. It applies a convolution in the non-equidistant domain, which is evaluated at equidistant sampling nodes. This convolution is then corrected in the the equidistant domain by division with the inverse Fourier transform ``\hat{\varphi}``. 

The adjoint NFFT matrix approximates ``\bm{A}^{\mathsf{H}}`` by

```math
\bm{A}^{\mathsf{H}} \approx \bm{D}^{\mathsf{H}} \bm{F}^{\mathsf{H}}  \bm{B}^{\mathsf{H}} 
```

Implementation-wise, the matrix-vector notation illustrates that the NFFT consists of three independent steps that are performed successively. 
* The multiplication with ``\bm{D}`` is a scalar multiplication with the input-vector plus the shifting of data, which can be done inplace.
* The FFT can be done with a high-performance FFT library such as the FFTW.
* The multiplication with ``\bm{B}`` needs to run only over a subset of the indices and is the most challenging step.
Since in practice the multiplication with ``\bm{B}`` is also the most expensive step, an NFFT library needs to pay special attention to optimizing it appropriately.

## Directional NFFT

In many cases one not just needs to apply a single NFFT but needs to apply many on different data. This leads us to the directional NFFT. The directional NFFT is defined as

```math
  	f_{\bm{l},j,\bm{r}} := \sum_{ \bm{k} \in I_{\bm{N}_\text{sub}}} \hat{f}_{\bm{l},\bm{k},\bm{r}} \, \mathrm{e}^{-2\pi\mathrm{i}\,\bm{k}\cdot\bm{x}}
```

where now ``(\bm{l}, \bm{k}, \bm{r}) \in I_\mathbf{N}`` and ``\bm{N}_\text{sub}`` is a subset of ``\bm{N}``. The transform thus maps a ``D``-dimensional tensor ``\hat{f}_{\bm{l},\bm{k},\bm{r}}`` to an ``R``-dimensional tensor ``f_{\bm{l},j,\bm{r}}``. ``\bm{N}_\text{sub}`` is thus a vector of length ``D-R+1`` The indices ``\bm{l}`` and ``\bm{r}`` can also have length zero. Thus, for ``R=1``, the conventional NFFT arises as a special case of the directional NFFT.

!!! note
    The directional NFFT can also be considered to be a slicing of a tensor with subsequent application of a regular NFFT. But the aforementioned formulation can be used to implement a much more efficient algorithm than can be achieved with slicing.

