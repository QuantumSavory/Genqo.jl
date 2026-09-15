using Documenter
using DocumenterCitations
using Genqo

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"))
makedocs(
    sitename = "Genqo.jl",
    format   = Documenter.HTML(edit_link = "main"),
    modules  = [Genqo, Genqo.tools, Genqo.tmsv, Genqo.spdc, Genqo.zalm, Genqo.sigsag],
    pages = [
        "Overview" => [
            "Home"            => "index.md",
            "Getting Started" => "getting_started.md",
        ],
        "API" => [
            "Overview"                    => "api/index.md",
            "Click States and Projectors" => "api/clicks.md",
            "Projection and Metrics"      => "api/projection.md",
            "Memory Loading"              => "api/memory.md",
            "Gaussian Unitaries"          => "api/unitaries.md",
            "Wick Contraction"            => "api/wick.md",
        ],
        "Legacy" => [
            "Overview" => "legacy/index.md",
            "ZALM"     => "legacy/zalm.md",
            "SPDC"     => "legacy/spdc.md",
            "TMSV"     => "legacy/tmsv.md",
            "SIGSAG"   => "legacy/sigsag.md",
            "Tools"    => "legacy/tools.md",
        ],
    ],
    checkdocs = :exports,
    plugins=[bib],
)

deploydocs(
    repo = "github.com/QuantumSavory/Genqo.jl.git",
    devbranch = "main",
    push_preview = true,
)
