# Smoke tests for the v1 legacy submodules. Their numerics are validated against the
# reference Python implementation by the comparison suite (just test-py); here we only
# check that the entry points run and return finite values.

@testitem "legacy TMSV" begin
    @test !isnan(tmsv.probability_success(1e-2, 0.9))
end

@testitem "legacy SPDC" begin
    @test !any(isnan, spdc.spin_density_matrix(1e-4, 0.9, 0.6, [0, 1, 0, 1]))
    @test !isnan(spdc.fidelity(1e-2, 0.8, 0.6))
end

@testitem "legacy ZALM" begin
    @test !any(isnan, zalm.spin_density_matrix(1e-4, 0.9, 0.6, 0.8, [1, 0, 1, 1, 0, 0, 1, 0]))
    @test !isnan(zalm.probability_success(1e-2, 0.8, 0.6, 0.9, 0.2))
    @test !isnan(zalm.fidelity(1e-2, 0.8, 0.6, 0.9))
end

@testitem "legacy SIGSAG" begin
    @test !isnan(sigsag.probability_success(1e-2, 0.8, 0.6))
    @test !isnan(sigsag.fidelity(1e-2, 0.8, 0.6))
end
