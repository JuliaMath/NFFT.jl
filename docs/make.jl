using Documenter, NFFT, NFFTTools

# Updating doctests:
# 
# julia> using Documenter, NFFT, NFFTTools
# julia> DocMeta.setdocmeta!(NFFT, :DocTestSetup, :(using NFFT, NFFTTools); recursive=true)
# julia> doctest(NFFT, fix=true)

DocMeta.setdocmeta!(NFFT, :DocTestSetup, :(using NFFT, NFFTTools); recursive=true)
DocMeta.setdocmeta!(NFFTTools, :DocTestSetup, :(using NFFT, NFFTTools); recursive=true)

makedocs(;
    doctest = true,
    #strict = :doctest,
    modules = [NFFT,NFFTTools],
    sitename = "NFFT",
    authors = "Tobias Knopp and contributors",
    repo="https://github.com/JuliaMath/NFFT.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tknopp.github.io/NFFT.jl",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",
        "Background" => "background.md",
        "Overview" => "overview.md",
        "Performance" => "performance.md",
        "Tools" => "tools.md",
        #"Implementation" => "implementation.md",
        "AbstractNFFTs" => "abstract.md",
        "API" => "api.md",
    ],
    doctestfilters = [r"(\d*)\.(\d{1})\d*" => s"\1.\2***"],
    doctest = false
)

deploydocs(repo   = "github.com/JuliaMath/NFFT.jl.git")
