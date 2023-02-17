using PartitionedStructures
using PartitionedStructures.Instances, PartitionedStructures.Link
using PartitionedStructures.M_abstract_part_struct, PartitionedStructures.M_part_v

@testset "Test Link" begin
  epm1, epv1 = create_epv_epm(; n = 9, nie = 7, overlapping = 6, mul_m = 5.0, mul_v = 100.0)
  x = ones(9)
  epm1_x = mul_epm_vector(epm1, x)
  @test Matrix(epm1) * x == epm1_x

  epm2, epv2 = create_epv_eplo_bfgs(; n = 9, nie = 7, overlapping = 6, mul_v = 100.0)
  x = ones(9)
  epm2_x = mul_epm_vector(epm2, x)
  @test Matrix(epm2) * x == epm2_x

  _epv2 = epv_from_epm(epm2)
  @test check_epv_epm(epm2, _epv2)
  @test full_check_epv_epm(epm2, _epv2)

  _epm2 = epm_from_epv(epv2)
  @test check_epv_epm(epv2, _epm2)
  @test full_check_epv_epm(epv2, _epm2)

  _eplo2 = eplo_lbfgs_from_epv(epv2)
  @test check_epv_epm(epv2, _eplo2)
  @test full_check_epv_epm(epv2, _eplo2)

  epv = epv_from_eplo(_eplo2)
  @test check_epv_epm(epv, _eplo2)
  @test full_check_epv_epm(epv, _eplo2)

  @test mul_epm_epv(_eplo2, epv) == mul_epm_epv(_epm2, epv)

  @test string_counters_iter(_epm2) ==
        "\t structure: Elemental_pm{Float64} based from 3 elements; update: 0, untouch: 0, reset: 0 \n"

  @test string_counters_total(_epm2) ==
        "\t structure: Elemental_pm{Float64} based from 3 elements; update: 0, untouch: 0, reset: 0 \n"
end


@testset "mul_epm_epv with linear_vector" begin
  N = 4
  n = 8
  element_variables = [[1, 2, 5, 7], [3, 6, 7, 8], [2, 4, 6, 8], [1, 3, 5, 6, 7]]

  epv = PartitionedStructures.epv_from_v(ones(n), create_epv(element_variables))
  # without linear_vectors
  B = identity_epm(element_variables)  

  res = mul_epm_epv(B, epv)
  vres = Vector(res)
  @test sum(vres) == 17
  @test vres[1] == 2
  @test vres[2] == 2
  @test vres[3] == 2
  @test vres[4] == 1
  @test vres[5] == 2
  @test vres[6] == 3
  @test vres[7] == 3
  @test vres[8] == 2

  # with linear_vectors
  linears = [true,false,false,true]
  B = identity_epm(element_variables; linear_vector=linears)

  res = mul_epm_epv(B, epv)
  vres = Vector(res)
  @test sum(vres) == 8
  @test vres[1] == 0
  @test vres[2] == 1
  @test vres[3] == 1
  @test vres[4] == 1
  @test vres[5] == 0
  @test vres[6] == 2
  @test vres[7] == 1
  @test vres[8] == 2
end
