using PartitionedStructures
using PartitionedStructures.Link, PartitionedStructures.M_part_v, PartitionedStructures.PartitionedLOQuasiNewton

using PartitionedStructures.Utils

using LinearAlgebra

@testset "PLBFGS first test" begin
	n=10
	nie=4
	over=2
	(eplom_B,epv_y) = create_epv_eplom(; n=n, nie=nie, overlapping=over)
	s = ones(n)
	B = Matrix(eplom_B)

	eplom_B1 = PLBFGS(eplom_B,epv_y,s)
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
		epm_B1,epv_y1 = create_epv_eplom(;n=n,nie=nie,overlapping=over,range_mul_m=rand()+1, mul_v=rand()*100)
		s = 100 .* rand(n)
		epm_B11 = PLBFGS(epm_B1,epv_y1,s) 
		@test mapreduce((x -> x>0), my_and, eigvals(Matrix(epm_B11))) #test positive eigensvalues
	end 
end 
