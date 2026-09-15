# Precompute ground-truth results for validating the v2 generalized framework. Run manually
# whenever the parameter sets in gt_common.jl change, then commit the updated JLD2 file:
#
#     just ground-truth
#     # or: julia --project=test test/generate_ground_truth.jl
#
# The numbers come from `test/ground_truth/GenqoV1.jl`: the original v1 implementation, frozen
# at the commit before `src/legacy/` was rewritten as a thin wrapper over v2. Everything here
# is qualified as `GenqoV1.<source>` rather than using the `Genqo.<source>` submodules of the
# same name, because those now compute through the very framework this data validates.
#
# Covariance matrices are stored in qqpp ordering / ħ=1 convention (see `legacy_state` in
# gt_common.jl). ZALM uses dark_counts = 0 since v2 has no dark-count model.

using JLD2

include("gt_common.jl")
include(joinpath(@__DIR__, "ground_truth", "GenqoV1.jl"))

function generate()
    params = gt_params()
    data = Dict{String,Any}("seed" => GT_SEED)

    for (src, m) in params
        data["$src/params"] = m
    end

    let m = params["tmsv"]
        data["tmsv/covariance"] = stack(GenqoV1.tools.reorder(GenqoV1.tmsv.covariance_matrix(m[i, 1])) for i in 1:GT_NCASES)
        data["tmsv/pgen"] = [GenqoV1.tmsv.probability_success(m[i, 1], m[i, 3]) for i in 1:GT_NCASES]
    end

    let m = params["spdc"]
        data["spdc/covariance"] = stack(GenqoV1.tools.reorder(GenqoV1.spdc.covariance_matrix(m[i, 1])) for i in 1:GT_NCASES)
        data["spdc/fidelity"] = [GenqoV1.spdc.fidelity(m[i, 1], m[i, 2], m[i, 3]) for i in 1:GT_NCASES]
        data["spdc/sdm"] = stack(GenqoV1.spdc.spin_density_matrix(m[i, 1], m[i, 2], m[i, 3], GT_NVEC_SPDC) for i in 1:GT_NCASES)
    end

    let m = params["zalm"]
        data["zalm/covariance"] = stack(GenqoV1.zalm.covariance_matrix(m[i, 1]) for i in 1:GT_NCASES)
        data["zalm/pgen"] = [GenqoV1.zalm.probability_success(m[i, 1], m[i, 2], m[i, 3], m[i, 4], 0.0) for i in 1:GT_NCASES]
        data["zalm/fidelity"] = [GenqoV1.zalm.fidelity(m[i, 1], m[i, 2], m[i, 3], m[i, 4]) for i in 1:GT_NCASES]
        data["zalm/sdm"] = stack(GenqoV1.zalm.spin_density_matrix(m[i, 1], m[i, 2], m[i, 3], m[i, 4], GT_NVEC_ZALM) for i in 1:GT_NCASES)
    end

    let m = params["sigsag"]
        data["sigsag/covariance"] = stack(GenqoV1.sigsag.covariance_matrix(m[i, 1]) for i in 1:GT_NCASES)
        data["sigsag/pgen"] = [GenqoV1.sigsag.probability_success(m[i, 1], m[i, 2], m[i, 3]) for i in 1:GT_NCASES]
        data["sigsag/fidelity"] = [GenqoV1.sigsag.fidelity(m[i, 1], m[i, 2], m[i, 3]) for i in 1:GT_NCASES]
    end

    mkpath(dirname(GT_FILE))
    jldopen(GT_FILE, "w") do f
        for (k, v) in data
            f[k] = v
        end
    end
    @info "Ground truth written" file = GT_FILE keys = sort(collect(keys(data)))
end

generate()
