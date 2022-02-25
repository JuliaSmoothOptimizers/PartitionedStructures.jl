using PartitionedStructures
using PartitionedStructures.Link
using PartitionedStructures.M_abstract_part_struct, PartitionedStructures.M_part_v

@testset "Test Link" begin
  epm1,epv1 = create_epv_epm(;n=9,nie=5,overlapping=1,mul_m=5., mul_v=100.)
  epm2,epv2 = create_epv_epm(;n=9,nie=3,overlapping=0,mul_m=5., mul_v=100.)

  @test check_epv_epm(epm1,epv1)	
  @test full_check_epv_epm(epm1,epv1)	

  @test check_epv_epm(epm2,epv2)	
  @test full_check_epv_epm(epm2,epv2)	

  @test full_check_epv_epm(epm2,epv1) == false

  epm3,epv3 = create_epv_epm(;n=16,nie=6,overlapping=1,mul_m=5., mul_v=100.)

  @test check_epv_epm(epm3,epv1) == false
  @test full_check_epv_epm(epm3,epv1) == false

  epm4,epv4 = create_epv_epm(;n=9,nie=7,overlapping=6,mul_m=5., mul_v=100.)
  @test check_epv_epm(epm2,epv4)
  @test full_check_epv_epm(epm2,epv4)	== false

  x = ones(9)
  epm4_x = mul_epm_vector(epm4,x)
  @test Matrix(epm4) * x == epm4_x

  epm5,epv5 = create_epv_eplom_bfgs(;n=9,nie=7,overlapping=6, mul_v=100.)
  x = ones(9)
  epm5_x = mul_epm_vector(epm5,x)
  @test Matrix(epm5) * x == epm5_x
end