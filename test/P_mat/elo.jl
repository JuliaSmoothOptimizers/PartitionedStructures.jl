
using PartitionedStructures
using PartitionedStructures.ModElemental_elo_sr1, PartitionedStructures.ModElemental_elo_bfgs

@testset "elo LBFGS" begin
  n = 50
  nie = 10
  indices = sample(1:n, nie, replace = false)

  eelo1 = init_eelo_LBFGS(indices)
  eelo2 = LBFGS_eelo_rand(nie)
  eelo3 = LBFGS_eelo(nie)

  reset_eelo_bfgs!(eelo1)

  eelo1_copy = copy(eelo1)
  eelo2_copy = copy(eelo2)
  eelo3_copy = copy(eelo3)

  eelo1_similar = similar(eelo1)
  eelo2_similar = similar(eelo2)
  eelo3_similar = similar(eelo3)

  @test eelo1 == eelo1_copy
  @test eelo1 != eelo2
  @test eelo2 != eelo3

  # equal since no update
  @test eelo1 == eelo1_similar
  @test eelo2 == eelo2_similar
  @test eelo3 == eelo3_similar
end

@testset "elo LSR1" begin
  n = 50
  nie = 10
  indices = sample(1:n, nie, replace = false)

  eelo1 = init_eelo_LSR1(indices)
  eelo2 = LSR1_eelo_rand(nie)
  eelo3 = LSR1_eelo(nie)

  reset_eelo_sr1!(eelo1)

  eelo1_copy = copy(eelo1)
  eelo2_copy = copy(eelo2)
  eelo3_copy = copy(eelo3)

  eelo1_similar = similar(eelo1)
  eelo2_similar = similar(eelo2)
  eelo3_similar = similar(eelo3)

  @test eelo1 == eelo1_copy
  @test eelo1 != eelo2
  @test eelo2 != eelo3

  @test eelo1 == eelo1_similar
  @test eelo2 == eelo2_similar
  @test eelo3 == eelo3_similar
end
