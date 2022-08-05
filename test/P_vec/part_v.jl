using LinearAlgebra, SparseArrays
using StatsBase
using PartitionedStructures.M_part_v
using PartitionedStructures.ModElemental_pv
using PartitionedStructures.ModElemental_ev
using PartitionedStructures.M_abstract_element_struct
using PartitionedStructures.M_abstract_part_struct

@testset "partitioned vector methods" begin
  N = 15
  n = 20
  nie = 5
  element_variables = map((i -> sample(1:n, nie, replace=false)), 1:N)
  epv = create_epv(element_variables, n)

  v = ones(n)
  set_v!(epv, v)
  @test get_v(epv) == v

  set_v!(epv, n, 3.)
  @test get_v(epv)[n] == 3.

  add_v!(epv, n, 2.)
  @test get_v(epv)[n] == 5.

  indices = 1:2:5
  vec_indices = [indices;]
  values = [1.:3.;]

  add_v!(epv, vec_indices, values)
  @test get_v(epv)[indices] == values + ones(3)

  reset_v!(epv)
  @test get_v(epv) == zeros(n)

  @test build_v(epv) != zeros(n)
end