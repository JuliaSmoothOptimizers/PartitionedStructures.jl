using PartitionedStructures.Instances
using StatsBase

@testset "Test update/update!" begin
  @testset " partitioned-matrix update" begin
    n = 9
    epm_bfgs, epv_y = create_epv_epm(; n = n, nie = 5, overlapping = 1, mul_m = 5.0, mul_v = 100.0)
    epm_sr1 = copy(epm_bfgs)
    s = ones(n)
    mat_PBFGS = update(epm_bfgs, epv_y, s; verbose = false)
    mat_PSR1 = update(epm_sr1, epv_y, s; name = :psr1, verbose = false)
  end

  @testset "partitioned linear-operators update" begin
    n = 10
    nie = 4
    over = 2
    s = ones(n)
    eplo_bfgs, epv_y = create_epv_eplo_bfgs(; n = n, nie = nie, overlapping = over)
    mat_PLBFGS = update(eplo_bfgs, epv_y, s; verbose = false)
    eplo_sr1, epv_y = create_epv_eplo_sr1(; n = n, nie = nie, overlapping = over)
    mat_PSR1 = update(eplo_sr1, epv_y, s; verbose = false)
    eplo_bfgs_sr1, epv_y = create_epv_eplo(; n = n, nie = nie, overlapping = over)
    mat_PLBFGS_SR1 = update(eplo_bfgs_sr1, epv_y, s; verbose = false)
  end

  @testset "partitioned linear-operators update!" begin
    n = 10
    nie = 4
    over = 2
    s = ones(n)
    eplo_bfgs, epv_y = create_epv_eplo_bfgs(; n = n, nie = nie, overlapping = over)
    mat_PLBFGS = update!(eplo_bfgs, epv_y, s; verbose = false)
    eplo_sr1, epv_y = create_epv_eplo_sr1(; n = n, nie = nie, overlapping = over)
    mat_PSR1 = update!(eplo_sr1, epv_y, s; verbose = false)
    eplo_bfgs_sr1, epv_y = create_epv_eplo(; n = n, nie = nie, overlapping = over)
    mat_PLBFGS_SR1 = update!(eplo_bfgs_sr1, epv_y, s; verbose = false)
  end
end

@testset "General update" begin
  n = 150
  N = 100
  ni = 9
  element_variables = map((i -> sample(1:n, ni, replace = false)), 1:N)
  epv = create_epv(element_variables)
  build_v!(epv)
  y = get_v(epv)
  epm = epm_from_epv(epv)
  eplo_lbfgs = eplo_lbfgs_from_epv(epv)
  eplo_sr1 = eplo_lsr1_from_epv(epv)
  eplo_lose = eplo_lose_from_epv(epv)

  s = ones(n)

  epm_bfgs = PBFGS_update(epm, epv, s; verbose = false)
  epm_sr1 = PSR1_update(epm, epv, s; verbose = false)
  epm_se = PSE_update(epm, epv, s; verbose = false)
  epm_plbfgs = PLBFGS_update(eplo_lbfgs, epv, s; verbose = false)
  epm_plsr1 = PLSR1_update(eplo_sr1, epv, s; verbose = false)
  epm_plse = PLSE_update(eplo_lose, epv, s; verbose = false)
end

@testset "Concrete example of the partitioned updates (verify the secant equation)" begin
  f(x) = x[1]^2 + x[2]^2 + x[3]^2 + x[1] * x[2] + 3x[2] * x[3]
  f1(x) = x[1]^2 + x[1] * x[2]
  f2(x) = x[1]^2 + x[2]^2 + 3x[1] * x[2]

  U1 = [1, 2]
  U2 = [2, 3]
  U = [U1, U2]
  f_pss(x, U) = f1(x[U[1]]) + f2(x[U[2]])

  using Test
  x0 = [2.0, 3.0, 4.0]
  x1 = [1.0, 2.0, 3.0]
  x2 = [0.0, 1.0, 1.0]
  s1 = x1 - x0
  s2 = x2 - x1

  @test f(x0) == f_pss(x0, U)
  @test f(x1) == f_pss(x1, U)
  @test f(x2) == f_pss(x2, U)

  ∇f(x) = [2x[1] + x[2], x[1] + 2x[2] + 3x[3], 2x[3] + 3x[2]]
  ∇f1(x) = [2x[1] + x[2], x[1]]
  ∇f2(x) = [2x[1] + 3x[2], 2x[2] + 3x[1]]
  function ∇f_pss(x, U)
    gradient = zeros(length(x))
    gradient[U1] = ∇f1(x[U[1]])
    gradient[U2] .+= ∇f2(x[U[2]])
    return gradient
  end
  @test ∇f(x0) == ∇f_pss(x0, U)
  y1 = (∇f(x1) .- ∇f(x0))
  element1_y1 = ∇f1(x1[U1]) .- ∇f1(x0[U1])
  element2_y1 = ∇f2(x1[U2]) .- ∇f2(x0[U2])
  y2 = (∇f(x2) .- ∇f(x1))
  element1_y2 = ∇f1(x2[U1]) .- ∇f1(x1[U1])
  element2_y2 = ∇f2(x2[U2]) .- ∇f2(x1[U2])

  ∇²f() = [
    2.0 1.0 0.0
    1.0 2.0 3.0
    0.0 3.0 2.0
  ]
  ∇²f1() = [
    2.0 1.0
    1.0 0.0
  ]
  ∇²f2() = [
    2.0 3.0
    3.0 2.0
  ]
  function ∇²f_pss(x, U)
    n = length(x)
    hessian = zeros(n, n)
    hessian[U1, U1] = ∇²f1(x[U[1]])
    hessian[U2, U2] .+= ∇²f2(x[U[2]])
    return hessian
  end

  n = length(x0)
  partitioned_gradient_x0 = create_epv(U, n)
  partitioned_gradient_x1 = create_epv(U, n)
  partitioned_gradient_x2 = create_epv(U, n)

  # vector_gradient_element(x, U) = Vector{Vector{Float64}}([∇f1(x[U[1]]), ∇f2(x[U[2]])])
  vector_gradient_element(x, U) = [∇f1(x[U[1]]), ∇f2(x[U[2]])]::Vector{Vector{Float64}}
  set_epv!(partitioned_gradient_x0, vector_gradient_element(x0, U))
  set_epv!(partitioned_gradient_x1, vector_gradient_element(x1, U))

  set_epv!(partitioned_gradient_x2, vector_gradient_element(x2, U))

  build_v!(partitioned_gradient_x0)
  build_v!(partitioned_gradient_x1)
  build_v!(partitioned_gradient_x2)
  @test get_v(partitioned_gradient_x0) == ∇f(x0)
  @test get_v(partitioned_gradient_x1) == ∇f(x1)
  @test get_v(partitioned_gradient_x2) == ∇f(x2)

  # build partitioned-gradient difference, epv_y1
  partitioned_gradient_difference = copy(partitioned_gradient_x0)
  minus_epv!(partitioned_gradient_difference)
  add_epv!(partitioned_gradient_x1, partitioned_gradient_difference)

  # build partitioned-gradient difference, epv_y2
  partitioned_gradient_difference2 = copy(partitioned_gradient_x1)
  minus_epv!(partitioned_gradient_difference2)
  add_epv!(partitioned_gradient_x2, partitioned_gradient_difference2)

  build_v!(partitioned_gradient_difference)
  build_v!(partitioned_gradient_difference2)
  @test get_v(partitioned_gradient_difference) == y1
  @test get_v(partitioned_gradient_difference2) == y2

  @test dot(s1, y1) > 0

  @test dot(s1[U1], element1_y1) > 0
  @test dot(s1[U2], element2_y1) > 0

  @test dot(s2[U2], element2_y2) > 0
  @test dot(s2[U1], element1_y2) > 0

  B0 = [i == j ? 1.0 : 0.0 for i = 1:n, j = 1:n]
  partitioned_matrix = epm_from_epv(partitioned_gradient_x0)
  Matrix(partitioned_matrix)
  B_BFGS1 = BFGS(s1, y1, B0)
  B_BFGS2 = BFGS(s2, y2, B_BFGS1)

  B_PBFGS1 =
    update(partitioned_matrix, partitioned_gradient_difference, s1; name = :pbfgs, verbose = false)
  @test norm(mul_epm_vector(partitioned_matrix, s1) - y1) == 0.0
  update!(partitioned_matrix, partitioned_gradient_difference2, s2; name = :pbfgs, verbose = false)
  B_PBFGS2 = Matrix(partitioned_matrix)

  norm(B_BFGS1 * s1 - y1)
  @test isapprox(norm(B_BFGS1 * s1 - y1), 0.0; atol = 1e-10)

  norm(B_PBFGS1 * s1 - y1)
  @test isapprox(norm(B_PBFGS1 * s1 - y1), 0.0)

  partitioned_matrix_PSR1 = epm_from_epv(partitioned_gradient_x0)
  partitioned_matrix_PSE = copy(partitioned_matrix_PSR1)
  partitioned_linear_operator_PLBFGS = eplo_lbfgs_from_epv(partitioned_gradient_x0)
  partitioned_linear_operator_PLSE = eplo_lose_from_epv(partitioned_gradient_x0)

  B_PSR1 = update(
    partitioned_matrix_PSR1,
    partitioned_gradient_difference,
    s1;
    name = :psr1,
    verbose = false,
  )
  B_PSE = update(
    partitioned_matrix_PSE,
    partitioned_gradient_difference,
    s1;
    name = :pse,
    verbose = false,
  ) # the default update

  B_PLBFGS =
    update(partitioned_linear_operator_PLBFGS, partitioned_gradient_difference, s1; verbose = false)
  B_PLSE =
    update(partitioned_linear_operator_PLSE, partitioned_gradient_difference, s1; verbose = false)

  @test isapprox(norm(B_PSR1 * s1 - y1), 0.0)
  @test isapprox(norm(B_PSE * s1 - y1), 0.0)
  @test isapprox(norm(B_PLBFGS * s1 - y1), 0.0)
  @test isapprox(norm(B_PLSE * s1 - y1), 0.0)

  # There is also a PLSR1 approximation, but is not fullt working since there is some issues with LSR1Operator
  partitioned_linear_operator_PLSR1 = eplo_lsr1_from_epv(partitioned_gradient_x0)
  B_PLSR1 =
    update(partitioned_linear_operator_PLSR1, partitioned_gradient_difference, s1; verbose = false)
  # @test norm(B_PLSR1 * s1 - y1)==0. # the second element hessian approximation is not update,
end

@testset "update with linear_vector" begin
  N = 4
  n = 8
  element_variables = [[1, 2, 5, 7], [3, 6, 7, 8], [2, 4, 6, 8], [1, 3, 5, 6, 7]]

  epv = PartitionedStructures.epv_from_v(ones(n), create_epv(element_variables))
  epv_y = similar(epv)
  epv_s = similar(epv)
  y_values = [Float64[1:4;], Float64[2:5;], Float64[3:6;], Float64[5:9;]]
  s_values = [Float64[1, 2, 5, 7], Float64[3, 6, 7, 8], Float64[2, 4, 6, 8], Float64[1, 3, 5, 6, 7]]
  set_epv!(epv_y, y_values)
  set_epv!(epv_s, s_values)

  # without linear_vectors
  B = identity_epm(element_variables)
  # with linear_vectors
  linears = [true,false,false,true]
  B_linear = identity_epm(element_variables; linear_vector=linears)

  Bv = Vector(mul_epm_epv(B, epv))
  B_linearv = Vector(mul_epm_epv(B_linear, epv))
  @test Bv != B_linearv

  # update
  PSR1_update!(B, epv_y, epv_s)
  PSR1_update!(B_linear, epv_y, epv_s)

  # verify the effectiveness of update
  @test Bv != Vector(mul_epm_epv(B, epv))
  @test B_linearv != Vector(mul_epm_epv(B_linear, epv))

  # verify different update
  @test Vector(mul_epm_epv(B, epv)) != Vector(mul_epm_epv(B_linear, epv))
end
