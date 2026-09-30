# Shared fixtures for the v2-vs-ground-truth tests. Loads the JLD2 file produced by
# generate_ground_truth.jl and guards against drift between the committed data and
# the parameter sets defined in gt_common.jl.
@testmodule GroundTruth begin

using JLD2

include("gt_common.jl")

const GT = load(GT_FILE)
const PARAMS = gt_params()

for src in GT_SOURCES
    PARAMS[src] == GT["$src/params"] ||
        error("Ground-truth parameters drifted from test/data/ground_truth.jld2; re-run test/generate_ground_truth.jl and commit the result")
end

"Bell state (|1001⟩ + |0110⟩)/√2 as a ClickStateKet over 4 modes."
bell4() = (Genqo.clicks([1, 0, 0, 1]) + Genqo.clicks([0, 1, 1, 0])) / √2

end
