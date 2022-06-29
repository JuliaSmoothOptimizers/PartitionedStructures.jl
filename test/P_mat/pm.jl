using StatsBase, SparseArrays
using PartitionedStructures.M_elt_vec
using PartitionedStructures.ModElemental_em
using PartitionedStructures.ModElemental_pm
using PartitionedStructures.M_part_mat

@testset "first tests on epm" begin
  N = 4
  n = 10
  nie = 3
  pm1 = identity_epm(N,n; nie=nie)
  set_spm!(pm1)
  pm2 = ones_epm(N,n; nie=nie)
  set_spm!(pm2)

  @test Matrix(pm2.spm)==transpose(Matrix(pm2.spm))
  @test sum(Matrix(pm2.spm))==N * nie^2
  @test sum(Matrix(pm1.spm))==N * nie

  @test (@allocated reset_spm!(pm1))==0

  set_spm!(pm1)
  original_spm1 = copy(get_spm(pm1))
  original_pm1 = copy(pm1)

  p = [1:n;]
  permute!(pm1,p)
  set_spm!(pm1)
  id_spm1 = copy(get_spm(pm1))
  id_pm1 = copy(pm1)
  @test id_spm1==original_spm1

  p = sample(1:n, n,replace=false)
  permute!(pm1,p)
  set_spm!(pm1)
  perm_spm1 = copy(get_spm(pm1))
  perm_pm1 = copy(pm1)
  @test perm_pm1 != original_spm1
end

@testset "pm PartiallySeparableNLPModels" begin
  N = 15
  n = 20
  nie = 5
  element_variables = vcat(map( (i -> rand(1:n,nie) ),1:N-1), [[4,8,12,16,20]])
  identity_epm(element_variables,N,n)
  identity_epm(element_variables)
  @test identity_epm(element_variables,N,n)==identity_epm(element_variables)
end