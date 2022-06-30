using LinearAlgebra
using PartitionedStructures
using PartitionedStructures.M_part_v
using PartitionedStructures.Instances, PartitionedStructures.Link, PartitionedStructures.Utils
using PartitionedStructures.PartitionedQuasiNewton

@testset "PQN updates" begin
  n = 9
  epm_B1, epv_y1 = create_epv_epm(; n = n, nie = 5, overlapping = 1, mul_m = 5.0, mul_v = 100.0)
  epm_B2, epv_y2 = create_epv_epm(; n = n, nie = 3, overlapping = 0, mul_m = 5.0, mul_v = 100.0)
  s = ones(n)
  epm_B11 = PBFGS_update(epm_B1, epv_y1, s)
  epm_B12 = PSR1_update(epm_B1, epv_y1, s)

  @test Matrix(epm_B1) == transpose(Matrix(epm_B1))
  @test Matrix(epm_B11) != Matrix(epm_B1)
  @test Matrix(epm_B11) != Matrix(epm_B12)
  @test mapreduce((x -> x > 0), my_and, eigvals(Matrix(epm_B11))) #test positive eigensvalues

  @test Matrix(epm_B11) == transpose(Matrix(epm_B11))
  @test Matrix(epm_B12) == transpose(Matrix(epm_B12))

  @test_throws DimensionMismatch PBFGS_update(epm_B1, epv_y2, s)
  @test_throws DimensionMismatch PSR1_update(epm_B1, epv_y2, s)

  @testset "Convexity preservation test of PBFGS_update" begin
    n_test = 50
    for i = 1:n_test
      n = rand(20:100)
      nie = rand(2:Int(floor(n / 2)))
      over = 1
      while mod(n - nie, nie - over) != 0
        over += 1
      end
      epm_B1, epv_y1 = create_epv_epm(;
        n = n,
        nie = nie,
        overlapping = over,
        mul_m = rand() + 1,
        mul_v = rand() * 100,
      )
      s = 100 .* rand(n)
      epm_B11 = PBFGS_update(epm_B1, epv_y1, s)
      @test mapreduce((x -> x > 0), my_and, eigvals(Matrix(epm_B11))) #test positive eigensvalues
    end
  end
end
