using Documenter
using Genqo

makedocs(
    sitename = "Genqo.jl",
    modules  = [Genqo, Genqo.tools, Genqo.tmsv, Genqo.spdc, Genqo.zalm, Genqo.sigsag],
    pages = [
        "Overview" => [
            "Home"            => "index.md",
            "Getting Started" => "getting_started.md",
        ],
        "Reference" => [
            "ZALM"   => "reference/zalm.md",
            "SPDC"   => "reference/spdc.md",
            "TMSV"   => "reference/tmsv.md",
            "SIGSAG" => "reference/sigsag.md",
            "Tools"  => "reference/tools.md",
        ],
    ],
    checkdocs = :exports,
)
