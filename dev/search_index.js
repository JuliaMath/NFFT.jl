var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#NFFT.jl-1",
    "page": "Home",
    "title": "NFFT.jl",
    "category": "section",
    "text": "Julia package for the Non-equidistant Fast Fourier Transform"
},

{
    "location": "index.html#Introduction-1",
    "page": "Home",
    "title": "Introduction",
    "category": "section",
    "text": "This package provides a Julia implementation of the Non-equidistant Fast Fourier Transform (NFFT). For a detailed introduction into the NFFT and its application please have a look at www.nfft.org.The NFFT is a fast implementation of the Non-equidistant Discrete Fourier Transform (NDFT) that is basically a DFT with non-equidistant sampling nodes in either Fourier or time/space domain. In contrast to the FFT, the NFFT is an approximative algorithm whereas the accuracy can be controlled by two parameters: the window width m and the oversampling factor sigma."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Start julia and open the package mode by entering ]. Then enteradd NFFTThis will install the packages NFFT.jl and all its dependencies."
},

{
    "location": "index.html#License-/-Terms-of-Usage-1",
    "page": "Home",
    "title": "License / Terms of Usage",
    "category": "section",
    "text": "The source code of this project is licensed under the MIT license. This implies that you are free to use, share, and adapt it. However, please give appropriate credit by citing the project."
},

{
    "location": "index.html#Contact-1",
    "page": "Home",
    "title": "Contact",
    "category": "section",
    "text": "If you have problems using the software, find mistakes, or have general questions please use the issue tracker to contact us."
},

{
    "location": "index.html#Contributors-1",
    "page": "Home",
    "title": "Contributors",
    "category": "section",
    "text": "Tobias Knopp"
},

{
    "location": "overview.html#",
    "page": "Overview",
    "title": "Overview",
    "category": "page",
    "text": ""
},

{
    "location": "overview.html#Overview-1",
    "page": "Overview",
    "title": "Overview",
    "category": "section",
    "text": "Basic usage of NFFT.jl is shown in the following example for 1D:using NFFT\n\nM, N = 1024, 512\nx = range(-0.4, stop=0.4, length=M)  # nodes at which the NFFT is evaluated\nfHat = randn(ComplexF64,M)           # data to be transformed\np = NFFTPlan(x, N)                   # create plan. m and sigma are optional parameters\nf = nfft_adjoint(p, fHat)            # calculate adjoint NFFT\ng = nfft(p, f)                       # calculate forward NFFTIn 2D:M, N = 1024, 16\nx = rand(2, M) .- 0.5\nfHat = randn(ComplexF64,M)\np = NFFTPlan(x, (N,N))\nf = nfft_adjoint(p, fHat)\ng = nfft(p, f)Currently, the eltype of the arguments f and fHat must be compatible that of the variable x used in the NFFTPlan call. For example, if one wants to use Float32 types to save memory, then one can make the plan using something like this:x = Float32.(LinRange(-0.5,0.5,64))\np = NFFTPlan(x, N)The plan will then internally use Float32 types. Then the arguments f and fHat above should have eltype Complex{Float32} or equivalently ComplexF32, otherwise there will be error messages."
},

{
    "location": "directional.html#",
    "page": "Directional",
    "title": "Directional",
    "category": "page",
    "text": ""
},

{
    "location": "directional.html#Directional-NFFT-1",
    "page": "Directional",
    "title": "Directional NFFT",
    "category": "section",
    "text": "There are special methods for computing 1D NFFT\'s for each 1D slice along a particular dimension of a higher dimensional array.M = 11\ny = rand(M) .- 0.5\nN = (16,20)\nP1 = NFFTPlan(y, 1, N)\nf = randn(ComplexF64,N)\nfHat = nfft(P1, f)Here size(f) = (16,20) and size(fHat) = (11,20) since we compute an NFFT along the first dimension. To compute the NFFT along the second dimensionP2 = NFFTPlan(y, 2, N)\nfHat = nfft(P2, f)Now size(fHat) = (16,11)."
},

{
    "location": "density.html#",
    "page": "Density",
    "title": "Density",
    "category": "page",
    "text": ""
},

{
    "location": "density.html#Sampling-Density-1",
    "page": "Density",
    "title": "Sampling Density",
    "category": "section",
    "text": "TODO"
},

]}
