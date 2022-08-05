using PartitionedStructures.Utils

@testset "test utils" begin
  @testset "my_and" begin
    @test my_and(true, false) == my_and(false, true)
    @test my_and(true, false) == false
    @test my_and(false, false) == false
    @test my_and(true, true) == true
  end

  @testset "max_indices and min_indices" begin
    N = 5
    elt_vars = map(i-> [i:i+5;], 1:N)
    @test max_indices(elt_vars) == 10
    @test min_indices(elt_vars) == 1
  end

  @testset "BFGS" begin
    n = 10
    B = [(i == j ? 1.0 : 0.0) for i = 1:n, j = 1:n]
    B_x2 = (x -> 2 * x).([(i == j ? 1.0 : 0.0) for i = 1:n, j = 1:n])
    s = rand(n)
    y = rand(n)

    B1 = BFGS(s, y, B)

    @test B1 == transpose(B1)
    @test isapprox(B1 * s, y)

    B2 = BFGS(s, -s, B)
    @test B2 == B

    B2_x2 = BFGS(s, -s, B_x2)
    @test B2_x2 == B_x2

    B3 = BFGS(s, -s, B_x2; index = 5, reset = 4)
    @test B3 == B
  end

  @testset "SR1" begin
    n = 10
    B = [(i == j ? 1.0 : 0.0) for i = 1:n, j = 1:n]
    B_x2 = (x -> 2 * x).([(i == j ? 1.0 : 0.0) for i = 1:n, j = 1:n])
    s = ones(n)
    y = (x -> x / 2).(ones(n))

    B1 = SR1(s, y, B)
    @test B1 == transpose(B1)
    @test isapprox(B1 * s, y)

    B2 = SR1(s, s, B)
    @test B2 == B

    B2_x2 = SR1((x -> 1 / 2 * x).(s), s, B_x2)
    @test B2_x2 == B_x2

    B3 = SR1((x -> 1 / 2 * x).(s), s, B_x2; index = 5, reset = 4)
    @test B3 == B
  end

  @testset "SE" begin
    n = 10
    B = [(i == j ? 1.0 : 0.0) for i = 1:n, j = 1:n]
    
    s = ones(n)
    y = (x -> x / 2).(ones(n))

    @test SE(s, y, B) == BFGS(s, y, B)
    @test SE(s, -s, B) == SR1(s, -s, B)
  end

end
