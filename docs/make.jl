using Documenter, Literate, NFFT, NFFTTools

# Examples are Literate scripts in literate/<section>/, rendered to src/generated/<section>/
# and run when the documentation is built.
const LITERATE = joinpath(@__DIR__, "literate")
const GENERATED = joinpath(@__DIR__, "src", "generated")
for dir in readdir(LITERATE), file in filter(endswith(".jl"), readdir(joinpath(LITERATE, dir)))
  Literate.markdown(joinpath(LITERATE, dir, file), joinpath(GENERATED, dir); documenter=true)
end

DocMeta.setdocmeta!(NFFT, :DocTestSetup, :(using NFFT); recursive=true)

makedocs(;
  modules=[NFFT, AbstractNFFTs, NFFTTools],
  sitename="NFFT.jl",
  authors="Tobias Knopp and contributors",
  repo=Remotes.GitHub("JuliaMath", "NFFT.jl"),
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://JuliaMath.github.io/NFFT.jl",
    assets=["assets/benchmarks/explorer.js", "assets/benchmarks/explorer.css"],
    size_threshold=500_000,
  ),
  pages=[
    "Home" => "index.md",
    "Mathematical Background" => "background.md",
    "Getting Started" => "generated/examples/getting_started.md",
    "Transforms along Dimensions" => "generated/examples/directional.md",
    "Accuracy and Parameters" => "generated/examples/accuracy.md",
    "Stages of the NFFT" => "generated/examples/stages.md",
    "Real Data" => "generated/examples/rnfft.md",
    "Cosine and Sine Transforms" => "generated/examples/real.md",
    "Non-equidistant in Both Domains" => "generated/examples/nnfft.md",
    "Density Compensation and Toeplitz" => "generated/examples/tools.md",
    "GPU" => "gpu.md",
    "Performance" => "performance.md",
    "Abstract Interface" => "abstract.md",
    "Migrating from NFFT.jl 0.x" => "migration.md",
    "API" => "api.md",
  ],
)

deploydocs(repo="github.com/JuliaMath/NFFT.jl.git", devbranch="main", push_preview=true)
