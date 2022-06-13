using PartitionedStructures
using PartitionedStructures.Instances
using PartitionedStructures.M_abstract_part_struct, PartitionedStructures.M_part_v

@testset "Creation of instances" begin
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
end

@testset "Matrix interface" begin 
  N = 5 
  n = 10 #by default must be a mulitple of 5
  id_epm = identity_epm(N,n)
  id_m = Matrix(id_epm)

  ones_pm = ones_epm(N,n)
  one_m = Matrix(ones_pm)

  ones_id_pm = ones_epm_and_id(N,n)
  one_id_m = Matrix(ones_id_pm)

  n_i_sep_pm = n_i_sep(n)
  sep_m = Matrix(n_i_sep_pm)

  n_i_sps_pm = n_i_SPS(n; overlapping=1)
  sps_sp_m = SparseMatrixCSC(n_i_sps_pm)	
  sps_m = Matrix(n_i_sps_pm)
end 

@testset "SparseMatrixCSC" begin
  N = 5 
  n = 10 #by default must be a mulitple of 5
  id_epm = identity_epm(N,n)
  id_m = Matrix(id_epm)

  ones_pm = ones_epm(N,n)
  one_m = Matrix(ones_pm)

  ones_id_pm = ones_epm_and_id(N,n)
  one_id_m = Matrix(ones_id_pm)

  n_i_sep_pm = n_i_sep(n)
  sep_m = Matrix(n_i_sep_pm)

  n_i_sps_pm = n_i_SPS(n; overlapping=1)
  sps_sp_m = SparseMatrixCSC(n_i_sps_pm)
  ldl(sps_sp_m)
  sps_m = Matrix(n_i_sps_pm)

  sp_m = sprand(10,10,0.1)
  sp_m2 = copy(sp_m)
  m = Matrix(sp_m)
  sm = Symmetric(triu(sp_m2), :U) # get upper triangle and apply Symmetric wrapper
end