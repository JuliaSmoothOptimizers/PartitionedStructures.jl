using StatsBase, SparseArrays
using PartitionedStructures.M_elt_vec
using PartitionedStructures.ModElemental_em
using PartitionedStructures.ModElemental_pm
using PartitionedStructures.M_part_mat
using PartitionedStructures.M_abstract_part_struct

@testset "first tests on epm" begin
  N = 4
  n = 10
  nie = 3
  pm1 = identity_epm(N, n; nie = nie)
  @test mapreduce(eem -> eem.convex == false, &, pm1.eem_set)

  set_spm!(pm1)
  pm2 = ones_epm(N, n; nie = nie)
  set_spm!(pm2)

  @test Matrix(pm2.spm) == transpose(Matrix(pm2.spm))
  @test sum(Matrix(pm2.spm)) == N * nie^2
  @test sum(Matrix(pm1.spm)) == N * nie

  reset_spm!(pm1)
  @test (@allocated reset_spm!(pm1)) == 0

  set_spm!(pm1)
  original_spm1 = copy(get_spm(pm1))
  original_pm1 = copy(pm1)

  p = [1:n;]
  permute!(pm1, p)
  set_spm!(pm1)
  id_spm1 = copy(get_spm(pm1))
  id_pm1 = copy(pm1)
  @test id_spm1 == original_spm1

  p = sample(1:n, n, replace = false)
  permute!(pm1, p)
  set_spm!(pm1)
  perm_spm1 = copy(get_spm(pm1))
  perm_pm1 = copy(pm1)
  @test perm_pm1 != original_spm1
end

@testset "Partitioned Matrices" begin
  N = 4
  n = 8
  element_variables = [[1, 2, 5, 7], [3, 6, 7, 8], [2, 4, 6, 8], [1, 3, 5, 6, 7]]
  bools = [true, true, true, true]

  epm = identity_epm(element_variables)
  epm_true = identity_epm(element_variables; convex_vector = bools)

  @test identity_epm(element_variables, N, n) == identity_epm(element_variables)
  @test mapreduce(eem -> eem.convex == false, &, epm.eem_set)

  copy_epm = copy(epm)
  similar_epm = similar(epm)

  @test epm == copy_epm
  @test check_epv_epm(epm, copy_epm)
  @test check_epv_epm(epm, similar_epm)
  @test check_epv_epm(epm, epm_true)

  @test full_check_epv_epm(epm, copy_epm)
  @test full_check_epv_epm(epm, similar_epm)
  @test full_check_epv_epm(epm, epm_true)

  @test epm == copy_epm
  @test epm != similar_epm
  @test epm != epm_true

  epm = identity_epm(element_variables)

  @test correlated_var(epm, 1) == [1, 2, 5, 7, 3, 6]
  @test correlated_var(epm, 8) == [3, 6, 7, 8, 2, 4]
  for i = 1:3
    @test get_eem_set_Bie(epm, i) == [i == j ? 1.0 : 0.0 for i = 1:4, j = 1:4]
  end
  @test get_eem_set_Bie(epm, 4) == [i == j ? 1.0 : 0.0 for i = 1:5, j = 1:5]

  indices = [1, 2]
  @test get_eem_sub_set(epm, indices) == epm.eem_set[indices]
end

@testset "Generate partitioned matrices" begin
  epm1 = ones_epm_and_id(6, 9)
  epm2 = n_i_sep(25)
  epm3 = n_i_SPS(20)

  @test epm1 != epm2
  @test epm1 != epm3
  @test epm2 != epm3
end

@testset "Allocation partitioned matrices (dense)" begin
  N = 4
  n = 8
  element_variables = [[1, 2, 5, 7], [3, 6, 7, 8], [2, 4, 6, 8], [1, 3, 5, 6, 7]]
  epm = identity_epm(element_variables)
  epv = epv_from_epm(epm)
  epv2 = similar(epv)

  PartitionedStructures.mul_epm_epv!(epv2, epm, epv) # warm-up
  a = @allocated PartitionedStructures.mul_epm_epv!(epv2, epm, epv)
  @test a == 0

  update!(epm, epv, epv2; name = :pbfsg, verbose = false)
  a = @allocated update!(epm, epv, epv2; name = :pbfsg, verbose = false)
  @test a == 0

  update!(epm, epv, epv2; name = :psr1, verbose = false)
  a = @allocated update!(epm, epv, epv2; name = :psr1, verbose = false)
  @test a == 0

  update!(epm, epv, epv2; name = :pse, verbose = false)
  a = @allocated update!(epm, epv, epv2; name = :pse, verbose = false)
  @test a == 0
end

@testset "Matrix and SparseMatrix with linear element matrices" begin
  N = 4
  n = 8
  element_variables = [[1, 2, 5, 7], [3, 6, 7, 8], [2, 4, 6, 8], [1, 3, 5, 6, 7]]
  
  linears = [true,false,false,true]
  B = identity_epm(element_variables; linear_vector=linears)
  Matrix(B)
end