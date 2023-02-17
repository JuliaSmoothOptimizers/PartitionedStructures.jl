using LinearAlgebra, LinearOperators, SparseArrays
using PartitionedStructures
using PartitionedStructures.M_elt_mat
using PartitionedStructures.ModElemental_elo_bfgs, PartitionedStructures.ModElemental_elo_sr1
using PartitionedStructures.ModElemental_plo,
  PartitionedStructures.ModElemental_plo_bfgs, PartitionedStructures.ModElemental_plo_sr1
using PartitionedStructures.Instances, PartitionedStructures.Link, PartitionedStructures.Utils
using PartitionedStructures.M_abstract_part_struct

@testset "test elemental element linear-operator matrix" begin
  for index = 3:3:15
    for T in [Float16, Float32, Float64]
      nie = 5
      indices = [index:1:(index + nie - 1);]
      Bie_bfgs = LinearOperators.LBFGSOperator(T, nie)
      Bie_sr1 = LinearOperators.LSR1Operator(T, nie)
      counter = Counter_elt_mat()
      linear = false
      @test Elemental_elo_bfgs{T}(nie, indices, Bie_bfgs, counter, linear) ==
            LBFGS_eelo(nie; T = T, index = index)
      @test Elemental_elo_sr1{T}(nie, indices, Bie_sr1, counter, linear) ==
            LSR1_eelo(nie; T = T, index = index)
    end
  end

  a = LBFGS_eelo(5)
  A = Matrix(get_Bie(a))
  @test A == transpose(A)

  a = LSR1_eelo(5)
  A = Matrix(get_Bie(a))
  @test A == transpose(A)
end

@testset "test elemental partitioned linear-operator matrix (PBFGS operator)" begin
  n = 10
  nie = 4
  over = 2
  (eplo, epv_y) = create_epv_eplo_bfgs(; n = n, nie = nie, overlapping = over)
  B = Matrix(eplo)
  @test B == transpose(B)

  @test mapreduce((x -> x > 0), my_and, eigvals(B)) # test definite positiveness

  copy_eplo = copy(eplo)
  similar_eplo = similar(eplo)
  @test eplo == copy_eplo
  @test eplo == similar_eplo

  @test check_epv_epm(eplo, copy_eplo)
  @test check_epv_epm(eplo, similar_eplo)

  @test full_check_epv_epm(eplo, copy_eplo)
  @test full_check_epv_epm(eplo, similar_eplo)
end

@testset "test elemental partitioned linear-operator matrix (PLSR1 operator)" begin
  n = 10
  nie = 4
  over = 2
  (eplo, epv_y) = create_epv_eplo_sr1(; n = n, nie = nie, overlapping = over)
  B = Matrix(eplo)
  @test B == transpose(B)

  @test mapreduce((x -> x > 0), my_and, eigvals(B)) # test definite positiveness

  copy_eplo = copy(eplo)
  similar_eplo = similar(eplo)
  @test eplo == copy_eplo
  @test check_epv_epm(eplo, copy_eplo)
  @test check_epv_epm(eplo, similar_eplo)

  @test full_check_epv_epm(eplo, copy_eplo)
  @test full_check_epv_epm(eplo, similar_eplo)
end

@testset "PL-BFGS-SR1 matrices" begin
  n = 10
  nie = 4
  over = 2
  eplo = PLBFGSR1_eplo(; n = n, nie = nie, overlapping = over)
  @test Matrix(eplo) == transpose(Matrix(eplo))

  copy_eplo = copy(eplo)
  similar_eplo = similar(eplo)
  @test eplo == copy_eplo
  @test check_epv_epm(eplo, copy_eplo)
  @test check_epv_epm(eplo, similar_eplo)

  @test full_check_epv_epm(eplo, copy_eplo)
  @test full_check_epv_epm(eplo, similar_eplo)

  eplo_B, epv_y = create_epv_eplo(; n = n, nie = nie, overlapping = over)
  s = ones(n)
  B = Matrix(eplo_B)
  @test B == transpose(B)
end

@testset "eplo_bfgs PartiallySeparableNLPModels" begin
  N = 4
  n = 8
  element_variables = [[1, 2, 5, 7], [3, 6, 7, 8], [2, 4, 6, 8], [1, 3, 5, 6, 7]]
  s = rand(n)

  eplo = identity_eplo_LBFGS(element_variables, N, n)
  @test eplo == identity_eplo_LBFGS(element_variables)
  epv = epv_from_epm(eplo)
  update(eplo, epv, s; verbose = false)

  copy_eplo = copy(eplo)
  similar_eplo = similar(eplo)
  @test eplo == copy_eplo
  @test check_epv_epm(eplo, copy_eplo)
  @test check_epv_epm(eplo, similar_eplo)

  @test full_check_epv_epm(eplo, copy_eplo)
  @test full_check_epv_epm(eplo, similar_eplo)
end

@testset "eplo_sr1 PartiallySeparableNLPModels" begin
  N = 15
  n = 20
  nie = 5
  s = rand(n)
  element_variables = vcat(map((i -> rand(1:n, nie)), 1:(N - 1)), [[4, 8, 12, 16, 20]])
  eplo = identity_eplo_LSR1(element_variables, N, n)
  @test eplo == identity_eplo_LSR1(element_variables)
  epv = epv_from_epm(eplo)
  update(eplo, epv, s; verbose = false)

  copy_eplo = copy(eplo)
  similar_eplo = similar(eplo)
  @test eplo == copy_eplo
  @test check_epv_epm(eplo, copy_eplo)
  @test check_epv_epm(eplo, similar_eplo)

  @test full_check_epv_epm(eplo, copy_eplo)
  @test full_check_epv_epm(eplo, similar_eplo)
end

@testset "eplo_se PartiallySeparableNLPModels" begin
  N = 15
  n = 20
  nie = 5
  element_variables = vcat(map((i -> rand(1:n, nie)), 1:(N - 1)), [[4, 8, 12, 16, 20]])
  eplo = identity_eplo_LOSE(element_variables, N, n)
  @test eplo == identity_eplo_LOSE(element_variables)
  s = rand(n)
  epv = epv_from_epm(eplo)
  update(eplo, epv, s; verbose = false)

  copy_eplo = copy(eplo)
  similar_eplo = similar(eplo)
  @test eplo == copy_eplo
  @test check_epv_epm(eplo, copy_eplo)
  @test check_epv_epm(eplo, similar_eplo)

  @test full_check_epv_epm(eplo, copy_eplo)
  @test full_check_epv_epm(eplo, similar_eplo)
end

@testset "plo_lbfgs" begin
  N = 20
  n = 50
  eplo_lbfgs = PLBFGS_eplo()
  eplo_lbfgs_rand = PLBFGS_eplo_rand(N, n)

  @test check_epv_epm(eplo_lbfgs, eplo_lbfgs_rand) == false
  @test full_check_epv_epm(eplo_lbfgs, eplo_lbfgs_rand) == false
  @test eplo_lbfgs != eplo_lbfgs_rand
end

@testset "plo_lsr1" begin
  N = 20
  n = 50
  eplo_lsr1 = PLSR1_eplo()
  eplo_lsr1_rand = PLSR1_eplo_rand(N, n)

  @test check_epv_epm(eplo_lsr1, eplo_lsr1_rand) == false
  @test full_check_epv_epm(eplo_lsr1, eplo_lsr1_rand) == false
  @test eplo_lsr1 != eplo_lsr1_rand
end

@testset "plo_l_bfgs-sr1" begin
  N = 20
  n = 50
  eplo_lsr1 = PLBFGSR1_eplo()
  eplo_lsr1_rand = PLBFGSR1_eplo_rand(N, n)

  @test check_epv_epm(eplo_lsr1, eplo_lsr1_rand) == false
  @test full_check_epv_epm(eplo_lsr1, eplo_lsr1_rand) == false
  @test eplo_lsr1 != eplo_lsr1_rand
end

@testset "spm" begin
  N = 15
  n = 20
  nie = 5
  element_variables =
    vcat(map((i -> sample(1:n, nie, replace = false)), 1:(N - 1)), [[4, 8, 12, 16, 20]])
  eplose = identity_eplo_LOSE(element_variables, N, n)
  eplosr1 = identity_eplo_LSR1(element_variables)
  @test SparseMatrixCSC(eplose) == SparseMatrixCSC(eplosr1)
end

@testset "Matrix and SparseMatrix with linear element matrices (LinearOperators)" begin
  N = 4
  n = 8
  element_variables = [[1, 2, 5, 7], [3, 6, 7, 8], [2, 4, 6, 8], [1, 3, 5, 6, 7]]

  linears = [true, false, false, true]
  B = identity_eplo_LSR1(element_variables; linear_vector = linears)
  Matrix(B)
end
