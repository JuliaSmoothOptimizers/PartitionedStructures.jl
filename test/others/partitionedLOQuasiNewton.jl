using LinearAlgebra
using PartitionedStructures
using PartitionedStructures.Instances, PartitionedStructures.Link, PartitionedStructures.Utils
using PartitionedStructures.M_part_v, PartitionedStructures.PartitionedLOQuasiNewton

@testset "PLBFGS first test" begin
  n=10
  nie=4
  over=2
  (eplom_B,epv_y) = create_epv_eplom_bfgs(; n=n, nie=nie, overlapping=over)
  s = ones(n)
  B = Matrix(eplom_B)

  eplom_B1 = PLBFGS_update(eplom_B,epv_y,s)
  B1 = Matrix(eplom_B1)

  @test B == transpose(B)
  @test B1 == transpose(B1)
  @test B != B1
  @test mapreduce((x -> x>0), my_and, eigvals(B1)) #test positive eigensvalues
end 

@testset "Convexity preservation test of PLBFGS" begin 
  n_test=50
  for i in 1:n_test
     n = rand(20:100)
    nie = rand(2:Int(floor(n/2)))
    over = 1
    while mod(n-nie,nie-over) != 0 
      over +=1
    end 
    epm_B1,epv_y1 = create_epv_eplom_bfgs(;n=n,nie=nie,overlapping=over,range_mul_m=rand()+1, mul_v=rand()*100)
    s = 100 .* rand(n)
    epm_B11 = PLBFGS_update(epm_B1,epv_y1,s) 
    @test mapreduce((x -> x>0), my_and, eigvals(Matrix(epm_B11))) #test positive eigensvalues
    @test Matrix(epm_B11) == transpose(Matrix(epm_B11))
  end 
end

@testset "PLSR1 first test" begin
  n=10
  nie=4
  over=2
  (eplom_B,epv_y) = create_epv_eplom_sr1(; n=n, nie=nie, overlapping=over)
  s = ones(n)
  B = Matrix(eplom_B)

  eplom_B1 = PLSR1_update(eplom_B,epv_y,s)
  B1 = Matrix(eplom_B1)

  @test B == transpose(B)
	@test isapprox(B1, transpose(B1))
	@test B != B1

end 

@testset "Partitionned update test" begin
  n=10
  nie=4
  over=2
  eplom = PLBFGSR1_eplom(;n=n,nie=nie,overlapping=over)
  eplom_B,epv_y = create_epv_eplom(;n=n,nie=nie,overlapping=over)
  s = ones(n)
  B = Matrix(eplom_B)
  @test B == transpose(B)	
  
  eplom_B1 = Part_update(eplom_B, epv_y, s)
  B1 = Matrix(eplom_B1)
  @test isapprox(B1, transpose(B1))
end
