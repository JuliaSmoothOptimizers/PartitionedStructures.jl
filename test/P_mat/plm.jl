using LinearAlgebra, LinearOperators, SparseArrays
using PartitionedStructures
using PartitionedStructures.M_elt_mat
using PartitionedStructures.ModElemental_elom_bfgs, PartitionedStructures.ModElemental_elom_sr1
using PartitionedStructures.ModElemental_plom, PartitionedStructures.ModElemental_plom_bfgs
using PartitionedStructures.Instances, PartitionedStructures.Link, PartitionedStructures.Utils

@testset "test elemental element linear operator matrix" begin
  for index in 3:3:15
    for T in [Float16,Float32,Float64]
      nie=5
      indices = [index:1:index+nie-1;]
      Bie_bfgs = LinearOperators.LBFGSOperator(T, nie)
      Bie_sr1 = LinearOperators.LSR1Operator(T, nie)
      counter = Counter_elt_mat()
      @test Elemental_elom_bfgs{T}(nie,indices,Bie_bfgs, counter) == LBFGS_eelom(nie;T=T, index=index)
      @test Elemental_elom_sr1{T}(nie,indices,Bie_sr1, counter) == LSR1_eelom(nie;T=T, index=index)
    end
  end

  a = LBFGS_eelom(5)
  A = Matrix(get_Bie(a))
  @test A == transpose(A)

  a = LSR1_eelom(5)
  A = Matrix(get_Bie(a))
  @test A == transpose(A)
end

@testset "test elemental partitioned linear operator matrix (PBFGS operator)" begin
  n=10
  nie=4
  over=2
  (eplom_B,epv_y) = create_epv_eplom_bfgs(; n=n, nie=nie, overlapping=over)
  B = Matrix(eplom_B)
  @test B == transpose(B)

  @test mapreduce((x -> x>0), my_and, eigvals(B)) # test definite positiveness
end


@testset "test elemental partitioned linear operator matrix (PLSR1 operator)" begin
  n=10
  nie=4
  over=2
  (eplom_B,epv_y) = create_epv_eplom_sr1(; n=n, nie=nie, overlapping=over)
  B = Matrix(eplom_B)
  @test B == transpose(B)

  @test mapreduce((x -> x>0), my_and, eigvals(B)) # test definite positiveness
end

@testset "PL-BFGS-SR1 matrices" begin
  n=10
  nie=4
  over=2
  eplom = PLBFGSR1_eplom(;n=n,nie=nie,overlapping=over)
  @test Matrix(eplom) == transpose(Matrix(eplom))

  eplom_B,epv_y = create_epv_eplom(;n=n,nie=nie,overlapping=over)
  s = ones(n)
  B = Matrix(eplom_B)
  @test B == transpose(B)
end

@testset "eplom_bfgs PartiallySeparableNLPModels" begin
  N = 15
  n = 20
  nie = 5
  s = rand(n)
  element_variables = map( (i -> rand(1:n,nie) ),1:N)
  eplom = identity_eplom_LBFGS(element_variables, N, n)
  epv = epv_from_epm(eplom)
  update(eplom, epv, s)
end

@testset "eplom_sr1 PartiallySeparableNLPModels" begin
  N = 15
  n = 20
  nie = 5
  s = rand(n)
  element_variables = map( (i -> rand(1:n,nie) ),1:N)
  eplom = identity_eplom_LSR1(element_variables, N, n)
  epv = epv_from_epm(eplom)
  update(eplom, epv, s)
end

@testset "eplom_se PartiallySeparableNLPModels" begin
  N = 15
  n = 20
  nie = 5
  s = rand(n)
  element_variables = map( (i -> rand(1:n,nie) ),1:N)
  eplom = identity_eplom_LOSE(element_variables, N, n)
  epv = epv_from_epm(eplom)
  update(eplom, epv, s)
end