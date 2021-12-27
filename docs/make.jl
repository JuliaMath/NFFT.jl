using Documenter, NFFT

DocMeta.setdocmeta!(NFFT, :DocTestSetup, :(using NFFT); recursive=true)

makedocs(;
    doctest = false,
    modules = [NFFT],
    sitename = "NFFT",
    authors = "Tobias Knopp and contributors",
    repo="https://github.com/tknopp/NFFT.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tknopp.github.io/NFFT.jl",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Directional" => "directional.md",
        "Density" => "density.md",
        "API" => "api.md",
    ]
)

deploydocs(repo   = "github.com/tknopp/NFFT.jl.git")
