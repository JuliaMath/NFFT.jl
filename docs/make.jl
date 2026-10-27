using Documenter, Literate, NFFT, NFFTTools, SparseArrays

# Examples are Literate scripts in literate/<section>/, rendered to src/generated/<section>/
# and run when the documentation is built.
const LITERATE = joinpath(@__DIR__, "literate")
const GENERATED = joinpath(@__DIR__, "src", "generated")
for dir in readdir(LITERATE), file in filter(endswith(".jl"), readdir(joinpath(LITERATE, dir)))
  Literate.markdown(joinpath(LITERATE, dir, file), joinpath(GENERATED, dir); documenter=true)
end

DocMeta.setdocmeta!(NFFT, :DocTestSetup, :(using NFFT); recursive=true)

makedocs(;
  modules=[NFFT, AbstractNFFTs, NFFTTools, Base.get_extension(NFFT, :NFFTSparseArraysExt)],
  sitename="NFFT.jl",
  authors="Tobias Knopp and contributors",
  repo=Remotes.GitHub("JuliaMath", "NFFT.jl"),
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://JuliaMath.github.io/NFFT.jl",
    assets=["assets/benchmarks/explorer.js", "assets/benchmarks/explorer.css"],
    size_threshold=500_000,
  ),
  # Diátaxis: a tutorial to learn, how-to guides for tasks, explanation for understanding,
  # reference for looking things up
  pages=[
    "Home" => "index.md",
    "Tutorial" => ["Getting Started" => "generated/examples/getting_started.md"],
    "How-to Guides" => [
      "Transforms along Dimensions" => "generated/examples/directional.md",
      "Real Data" => "generated/examples/rnfft.md",
      "Cosine and Sine Transforms" => "generated/examples/real.md",
      "Non-equidistant in Both Domains" => "generated/examples/nnfft.md",
      "GPU" => "gpu.md",
      "Density Compensation and Toeplitz" => "generated/examples/tools.md",
      "Choosing Parameters" => "generated/examples/parameters.md",
      "Stages of the NFFT" => "generated/examples/stages.md",
      "Migrating from NFFT.jl 0.x" => "migration.md",
    ],
    "Explanation" => [
      "Mathematical Background" => "background.md",
      "Accuracy and Parameters" => "generated/examples/accuracy.md",
      "How NFFT.jl Works" => "internals.md",
      "Performance" => "performance.md",
    ],
    "Reference" => [
      "API" => "api.md",
      "Abstract Interface" => "abstract.md",
    ],
  ],
)

deploydocs(repo="github.com/JuliaMath/NFFT.jl.git", devbranch="main", push_preview=true)
