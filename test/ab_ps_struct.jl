using LinearAlgebra, SparseArrays
using StatsBase
using PartitionedStructures.M_part_v
using PartitionedStructures.ModElemental_pv
using PartitionedStructures.ModElemental_ev
using PartitionedStructures.ModElemental_pm
using PartitionedStructures.ModElemental_em
using PartitionedStructures.M_abstract_element_struct
using PartitionedStructures.M_abstract_part_struct


@testset "Abstract partitioned structures" begin
  N = 15
  n = 20
  nie = 5
  element_variables = map((i -> sample(1:n, nie, replace=false)), 1:N)
  epv = create_epv(element_variables, n)
  epm = epm_from_epv(epv)

  @test get_component_list(epv) == get_component_list(epm)
  for i in 1:n
    @test get_component_list(epv, i) == get_component_list(epm, i)
  end
  @test get_N(epv) == N
  @test get_N(epm) == N
  
  @test get_n(epv) == n
  @test get_n(epm) == n

  @test full_check_epv_epm(epv, epm)
  z = @allocated full_check_epv_epm(epv, epm)
  @test z == 0

  @test get_ee_struct(epv) == epv.eev_set
  @test get_ee_struct(epm) == epm.eem_set
  for i in 1:N
    @test get_ee_struct(epv, i) == epv.eev_set[i]
    @test get_ee_struct(epm, i) == epm.eem_set[i]
  end
end

@testset "dynamical check methods" begin

  mutable struct TestPartitionedStruct{T} <: M_abstract_part_struct.AbstractPartitionedStructure{T}
    c::T
  end 

  ps = TestPartitionedStruct{Int}(5)
  
  @test_throws ErrorException get_ee_struct(ps)
  @test_throws ErrorException initialize_component_list!(ps)

end