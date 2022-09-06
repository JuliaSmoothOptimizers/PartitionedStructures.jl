@testset "PartitionedVectors" begin
  N = 15
  n = 20
  nie = 5
  element_variables =
    vcat(map((i -> sample(1:n, nie, replace = false)), 1:(N - 1)), [[1, 8, 12, 16, 20]])

  epv = create_epv(element_variables; type=Float32)
  _pv32 = PartitionedVector(epv)
  pv32 = PartitionedVector(element_variables; T=Float32)

  pv64 = PartitionedVector(element_variables)

  pv_sim = similar(pv32)
  pv_copy = copy(pv32)
  pv_res = copy(pv32)

  pv_copy .= pv32 .+ pv32
  pv_res .= pv_copy .- pv32

  @test pv_res == pv32
  @test pv_res == -pv32 + pv32 + pv32
end