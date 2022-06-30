using LinearAlgebra, LinearOperators, SparseArrays
using PartitionedStructures
using PartitionedStructures.M_elt_mat
using PartitionedStructures.ModElemental_elo_bfgs, PartitionedStructures.ModElemental_elo_sr1
using PartitionedStructures.ModElemental_plo, PartitionedStructures.ModElemental_plo_bfgs
using PartitionedStructures.Instances, PartitionedStructures.Link, PartitionedStructures.Utils

@testset "test elemental element linear-operator matrix" begin
  for index in 3:3:15
    for T in [Float16,Float32,Float64]
      nie=5
      indices = [index:1:index+nie-1;]
      Bie_bfgs = LinearOperators.LBFGSOperator(T, nie)
      Bie_sr1 = LinearOperators.LSR1Operator(T, nie)
      counter = Counter_elt_mat()
      @test Elemental_elo_bfgs{T}(nie,indices,Bie_bfgs, counter)==LBFGS_eelo(nie;T=T, index=index)
      @test Elemental_elo_sr1{T}(nie,indices,Bie_sr1, counter)==LSR1_eelo(nie;T=T, index=index)
    end
  end

  a = LBFGS_eelo(5)
  A = Matrix(get_Bie(a))
  @test A==transpose(A)

  a = LSR1_eelo(5)
  A = Matrix(get_Bie(a))
  @test A==transpose(A)
end

@testset "test elemental partitioned linear-operator matrix (PBFGS operator)" begin
  n=10
  nie=4
  over=2
  (eplo_B,epv_y) = create_epv_eplo_bfgs(; n=n, nie=nie, overlapping=over)
  B = Matrix(eplo_B)
  @test B==transpose(B)

  @test mapreduce((x -> x>0), my_and, eigvals(B)) # test definite positiveness
end

@testset "test elemental partitioned linear-operator matrix (PLSR1 operator)" begin
  n=10
  nie=4
  over=2
  (eplo_B,epv_y) = create_epv_eplo_sr1(; n=n, nie=nie, overlapping=over)
  B = Matrix(eplo_B)
  @test B==transpose(B)

  @test mapreduce((x -> x>0), my_and, eigvals(B)) # test definite positiveness
end

@testset "PL-BFGS-SR1 matrices" begin
  n=10
  nie=4
  over=2
  eplo = PLBFGSR1_eplo(;n=n,nie=nie,overlapping=over)
  @test Matrix(eplo)==transpose(Matrix(eplo))

  eplo_B,epv_y = create_epv_eplo(;n=n,nie=nie,overlapping=over)
  s = ones(n)
  B = Matrix(eplo_B)
  @test B==transpose(B)
end

@testset "eplo_bfgs PartiallySeparableNLPModels" begin
  N = 15
  n = 20
  nie = 5
  s = rand(n)
  element_variables = vcat(map((i -> rand(1:n,nie)),1:N-1), [[4,8,12,16,20]])
  eplo = identity_eplo_LBFGS(element_variables, N, n)
  @test eplo==identity_eplo_LBFGS(element_variables)
  epv = epv_from_epm(eplo)
  update(eplo, epv, s)
end

@testset "eplo_sr1 PartiallySeparableNLPModels" begin
  N = 15
  n = 20
  nie = 5
  s = rand(n)
  element_variables = vcat(map((i -> rand(1:n,nie)),1:N-1), [[4,8,12,16,20]])
  eplo = identity_eplo_LSR1(element_variables, N, n)
  @test eplo==identity_eplo_LSR1(element_variables)
  epv = epv_from_epm(eplo)
  update(eplo, epv, s)
end

@testset "eplo_se PartiallySeparableNLPModels" begin
  N = 15
  n = 20
  nie = 5
  element_variables = vcat(map((i -> rand(1:n,nie)),1:N-1), [[4,8,12,16,20]])
  eplo = identity_eplo_LOSE(element_variables, N, n)
  @test eplo==identity_eplo_LOSE(element_variables)
  s = rand(n)
  epv = epv_from_epm(eplo)
  update(eplo, epv, s)
end