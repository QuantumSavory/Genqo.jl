# Precompute ground-truth results with the v1 legacy code for validating the v2
# generalized framework. Run manually whenever the parameter sets in gt_common.jl
# change, then commit the updated JLD2 file:
#
#     julia --project=test test/generate_ground_truth.jl
#
# Covariance matrices are stored in qqpp ordering / ħ=1 convention (see
# `legacy_state` in gt_common.jl). ZALM uses dark_counts = 0 since v2 has no
# dark-count model.

using Genqo
using JLD2

include("gt_common.jl")

function generate()
    params = gt_params()
    data = Dict{String,Any}("seed" => GT_SEED)

    for (src, m) in params
        data["$src/params"] = m
    end

    let m = params["tmsv"]
        data["tmsv/covariance"] = stack(tools.reorder(tmsv.covariance_matrix(m[i, 1])) for i in 1:GT_NCASES)
        data["tmsv/pgen"] = [tmsv.probability_success(m[i, 1], m[i, 3]) for i in 1:GT_NCASES]
    end

    let m = params["spdc"]
        data["spdc/covariance"] = stack(tools.reorder(spdc.covariance_matrix(m[i, 1])) for i in 1:GT_NCASES)
        data["spdc/fidelity"] = [spdc.fidelity(m[i, 1], m[i, 2], m[i, 3]) for i in 1:GT_NCASES]
        data["spdc/sdm"] = stack(spdc.spin_density_matrix(m[i, 1], m[i, 2], m[i, 3], GT_NVEC_SPDC) for i in 1:GT_NCASES)
    end

    let m = params["zalm"]
        data["zalm/covariance"] = stack(zalm.covariance_matrix(m[i, 1]) for i in 1:GT_NCASES)
        data["zalm/pgen"] = [zalm.probability_success(m[i, 1], m[i, 2], m[i, 3], m[i, 4], 0.0) for i in 1:GT_NCASES]
        data["zalm/fidelity"] = [zalm.fidelity(m[i, 1], m[i, 2], m[i, 3], m[i, 4]) for i in 1:GT_NCASES]
        data["zalm/sdm"] = stack(zalm.spin_density_matrix(m[i, 1], m[i, 2], m[i, 3], m[i, 4], GT_NVEC_ZALM) for i in 1:GT_NCASES)
    end

    let m = params["sigsag"]
        data["sigsag/covariance"] = stack(sigsag.covariance_matrix(m[i, 1]) for i in 1:GT_NCASES)
        data["sigsag/pgen"] = [sigsag.probability_success(m[i, 1], m[i, 2], m[i, 3]) for i in 1:GT_NCASES]
        data["sigsag/fidelity"] = [sigsag.fidelity(m[i, 1], m[i, 2], m[i, 3]) for i in 1:GT_NCASES]
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
