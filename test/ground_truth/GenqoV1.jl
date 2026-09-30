# The v1 legacy source models, frozen at commit 8d75cae — the last revision in which
# `src/legacy/` still carried its own implementation, before those modules were rewritten as
# thin wrappers over the v2 generalized framework.
#
# This exists for exactly one reason: to be the *independent* oracle that
# test/generate_ground_truth.jl writes into test/data/ground_truth.jld2, which the v2 test
# suite then validates against. Once `src/legacy/` became a v2 wrapper it stopped being an
# oracle at all — regenerating from it would have compared v2 against itself — so the original
# implementation lives on here instead.
#
# What makes it independent: it builds its covariance matrices by hand in qpqp ordering,
# assembles `A = tools.k_function_matrix(cov) + loss_matrix` and inverts it whole, and
# contracts hand-written moment polynomials against that inverse. The v2 path shares none of
# that — it builds states with Gabs, gets A⁻¹ from the closed-form block inversion in
# `Genqo._invA`/`_invA_UL`, and derives its moment polynomials from click patterns.
#
# The one thing the two paths do share is the Wick evaluator itself (`W`/`extract_W_terms`
# from src/wick.jl), exactly as they did historically. A change to the hafnian kernel would
# therefore move both sides together; everything upstream of it is checked independently.
#
# Do not edit the files below to make a failing test pass. If the v2 framework disagrees with
# this code, the v2 framework is what changed. The only legitimate reason to regenerate the
# JLD2 is a change to the parameter sets in test/gt_common.jl.
module GenqoV1

using Genqo # for the shared Wick evaluator: wick_out, W, WTerms, extract_W_terms

include("tools.jl")
include("tmsv.jl")
include("spdc.jl")
include("zalm.jl")
include("sigsag.jl")

import .tools
import .tmsv
import .spdc
import .zalm
import .sigsag

export tools, tmsv, spdc, zalm, sigsag

end # module
