using PartitionedStructures.Instances

@testset "Test update/update!" begin
  @testset " partitioned-matrix update" begin
    n=9
    epm_bfgs,epv_y = create_epv_epm(;n=n,nie=5,overlapping=1,mul_m=5., mul_v=100.)
    epm_sr1 = copy(epm_bfgs)
    s = ones(n)
    mat_PBFGS = update(epm_bfgs, epv_y, s)
    mat_PSR1 = update(epm_sr1, epv_y, s; name=:psr1)
  end

  @testset "partitioned linear operators update" begin
    n=10
    nie=4
    over=2
    s = ones(n)
    eplom_bfgs,epv_y = create_epv_eplom_bfgs(; n=n, nie=nie, overlapping=over)
    mat_PLBFGS = update(eplom_bfgs, epv_y, s)
    eplom_sr1,epv_y = create_epv_eplom_sr1(; n=n, nie=nie, overlapping=over)
    mat_PSR1 = update(eplom_sr1, epv_y, s)
    eplom_bfgs_sr1,epv_y = create_epv_eplom(;n=n,nie=nie,overlapping=over)
    mat_PLBFGS_SR1 = update(eplom_bfgs_sr1, epv_y, s)
  end

  @testset "partitioned linear operators update!" begin
    n=10
    nie=4
    over=2
    s = ones(n)
    eplom_bfgs,epv_y = create_epv_eplom_bfgs(; n=n, nie=nie, overlapping=over)
    mat_PLBFGS = update!(eplom_bfgs, epv_y, s)
    eplom_sr1,epv_y = create_epv_eplom_sr1(; n=n, nie=nie, overlapping=over)
    mat_PSR1 = update!(eplom_sr1, epv_y, s)
    eplom_bfgs_sr1,epv_y = create_epv_eplom(;n=n,nie=nie,overlapping=over)
    mat_PLBFGS_SR1 = update!(eplom_bfgs_sr1, epv_y, s)
  end
end

@testset "General update" begin
  n = 150
  N = 100
  ni = 9
  sp_vec_elt(ni::Int, n; p=ni/n) = sprand(Float64, n, p)
  vec_sp_eev = map(i -> sp_vec_elt(ni, n), 1:N)
  map(spx -> begin spx.nzval .*= 100; spx.nzval .-= 50 end, vec_sp_eev)
  epv = create_epv(vec_sp_eev)
  build_v!(epv)
  y = get_v(epv)
  epm = epm_from_epv(epv)
  eplom_lbfgs = eplom_lbfgs_from_epv(epv)
  eplom_sr1 = eplom_lsr1_from_epv(epv)
  eplom_lose = eplom_lose_from_epv(epv)

  s = ones(n)

  epm_bfgs = PBFGS_update(epm, epv, s)
  epm_sr1 = PSR1_update(epm, epv, s)
  epm_se = PSE_update(epm, epv, s)
  epm_plbfgs = PLBFGS_update(eplom_lbfgs, epv, s)
  epm_plsr1 = PLSR1_update(eplom_sr1, epv, s)
  epm_plse = PLSE_update(eplom_lose, epv, s)
end


@testset "Concrete example of the partitioned updates (verify the secant equation)" begin
  f(x) = x[1]^2 + x[2]^2 + x[3]^2 + x[1]*x[2] + 3x[2]*x[3]
  f1(x) = x[1]^2 + x[1]*x[2]
  f2(x) = x[1]^2 + x[2]^2 + 3x[1]*x[2]

  U1 = [1, 2]
  U2 = [2, 3]
  U = [U1, U2]
  f_pss(x, U) = f1(x[U[1]]) + f2(x[U[2]])

  using Test
  x0 = [2., 3., 4.]
  x1 = [1., 2., 3.]
  x2 = [0., 1., 1.]
  s1 = x1 - x0
  s2 = x2 - x1

  @test f(x0)==f_pss(x0, U)
  @test f(x1)==f_pss(x1, U)
  @test f(x2)==f_pss(x2, U)

  ∇f(x) = [2x[1] + x[2], x[1] + 2x[2] + 3x[3], 2x[3] + 3x[2]]
  ∇f1(x) = [2x[1] + x[2], x[1]]
  ∇f2(x) = [2x[1] + 3x[2], 2x[2] + 3x[1]]
  function ∇f_pss(x, U)
    gradient = zeros(length(x))
    gradient[U1] = ∇f1(x[U[1]])
    gradient[U2] .+= ∇f2(x[U[2]])
    return gradient
  end
  @test ∇f(x0)==∇f_pss(x0, U)
  y1 = (∇f(x1) .- ∇f(x0))
  element1_y1 = ∇f1(x1[U1]) .- ∇f1(x0[U1])
  element2_y1 = ∇f2(x1[U2]) .- ∇f2(x0[U2])
  y2 = (∇f(x2) .- ∇f(x1))
  element1_y2 = ∇f1(x2[U1]) .- ∇f1(x1[U1])
  element2_y2 = ∇f2(x2[U2]) .- ∇f2(x1[U2])

  ∇²f() = [2. 1. 0.
            1. 2. 3.
            0. 3. 2.]
  ∇²f1() = [2. 1.
            1. 0.]
  ∇²f2() = [2. 3.
            3. 2.]
  function ∇²f_pss(x, U)
    n = length(x)
    hessian = zeros(n, n)
    hessian[U1,U1] = ∇²f1(x[U[1]])
    hessian[U2,U2] .+= ∇²f2(x[U[2]])
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
  @test get_v(partitioned_gradient_x0)==∇f(x0)
  @test get_v(partitioned_gradient_x1)==∇f(x1)
  @test get_v(partitioned_gradient_x2)==∇f(x2)

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
  @test get_v(partitioned_gradient_difference)==y1
  @test get_v(partitioned_gradient_difference2)==y2

  using LinearAlgebra
  @test dot(s1,y1) > 0

  @test dot(s1[U1], element1_y1) > 0
  @test dot(s1[U2], element2_y1) > 0

  @test dot(s2[U2], element2_y2) > 0
  @test dot(s2[U1], element1_y2) > 0

  B0 = [ i==j ? 1. : 0. for i in 1:n, j in 1:n]
  partitioned_matrix = epm_from_epv(partitioned_gradient_x0)
  Matrix(partitioned_matrix)
  B_BFGS1 = BFGS(s1,y1,B0)
  B_BFGS2 = BFGS(s2,y2,B_BFGS1)

  B_PBFGS1 = update(partitioned_matrix, partitioned_gradient_difference, s1; name=:pbfgs)
  @test norm(mul_epm_vector(partitioned_matrix, s1) - y1)==0.
  update!(partitioned_matrix, partitioned_gradient_difference2, s2; name=:pbfgs)
  B_PBFGS2 = Matrix(partitioned_matrix)

  norm(B_BFGS1 * s1 - y1)
  @test norm(B_BFGS1 * s1 - y1)==0.

  norm(B_PBFGS1 * s1 - y1)
  @test norm(B_PBFGS1 * s1 - y1)==0.


  partitioned_matrix_PSR1 = epm_from_epv(partitioned_gradient_x0)
  partitioned_matrix_PSE = copy(partitioned_matrix_PSR1)
  partitioned_linear_operator_PLBFGS = eplom_lbfgs_from_epv(partitioned_gradient_x0)
  partitioned_linear_operator_PLSE = eplom_lose_from_epv(partitioned_gradient_x0)

  B_PSR1 = update(partitioned_matrix_PSR1, partitioned_gradient_difference, s1; name=:psr1)
  B_PSE = update(partitioned_matrix_PSE, partitioned_gradient_difference, s1; name=:pse) # the default update
  B_PSE = update(partitioned_matrix_PSE, partitioned_gradient_difference, s1; name=:pse) # the default update

  B_PLBFGS = update(partitioned_linear_operator_PLBFGS, partitioned_gradient_difference, s1)
  B_PLSE = update(partitioned_linear_operator_PLSE, partitioned_gradient_difference, s1)

  @test norm(B_PSR1 * s1 - y1)==0.
  @test norm(B_PSE * s1 - y1)==0.
  @test norm(B_PLBFGS * s1 - y1)==0.
  @test norm(B_PLSE * s1 - y1)==0.


  # There is also a PLSR1 approximation, but is not fullt working since there is some issues with LSR1Operator
  partitioned_linear_operator_PLSR1 = eplom_lsr1_from_epv(partitioned_gradient_x0)
  B_PLSR1 = update(partitioned_linear_operator_PLSR1, partitioned_gradient_difference, s1)
  # @test norm(B_PLSR1 * s1 - y1)==0. # the second element hessian approximation is not update,
  # the partitioned quasi-Newton approximation doesn't satisfiy the secan equation.

end