using PartitionedStructures
using PartitionedStructures.Link, PartitionedStructures.M_part_v, PartitionedStructures.PartitionedQuasiNewton

using PartitionedStructures.Utils

using LinearAlgebra

n=9
epm_B1,epv_y1 = create_epv_epm(;n=n,nie=5,overlapping=1,mul_m=5., mul_v=100.)
epm_B2,epv_y2 = create_epv_epm(;n=n,nie=3,overlapping=0,mul_m=5., mul_v=100.)

s = ones(n)

epm_B11 = PBFGS(epm_B1,epv_y1,s) 
epm_B12 = PSR1(epm_B1,epv_y1,s) 


@test Matrix(epm_B1) == transpose(Matrix(epm_B1))
@test Matrix(epm_B11) != Matrix(epm_B1)
@test Matrix(epm_B11) != Matrix(epm_B12)
@test mapreduce((x -> x>0), my_and, eigvals(Matrix(epm_B11))) #test positive eigensvalues

@test Matrix(epm_B11) == transpose(Matrix(epm_B11))
@test Matrix(epm_B12) == transpose(Matrix(epm_B12))

@test_throws DimensionMismatch PBFGS(epm_B1,epv_y2,s)
@test_throws DimensionMismatch PSR1(epm_B1,epv_y2,s) 

@testset "Convexity preservation test of PBFGS" begin 
	n_test=50
	for i in 1:n_test
	 	n = rand(20:100)
		nie = rand(2:Int(floor(n/2)))
		over = 1
		while mod(n-nie,nie-over) != 0 
			over +=1
		end 
		epm_B1,epv_y1 = create_epv_epm(;n=n,nie=nie,overlapping=over,mul_m=rand()+1, mul_v=rand()*100)
		s = 100 .* rand(n)
		epm_B11 = PBFGS(epm_B1,epv_y1,s) 
		@test mapreduce((x -> x>0), my_and, eigvals(Matrix(epm_B11))) #test positive eigensvalues
	end 
end 


# using PartitionedStructures.ModElemental_pm, PartitionedStructures.ModElemental_pv

# n=17
# nie=7
# over=2
# (x -> x+1).([0:nie-over:n+1-nie;])
# filter(x -> x <= n-nie+1, vcat(1,(x -> x + (nie-over)).([1:nie-over:n-(nie-over);])))
# mod(n-(nie-over), nie-over) == over || error("wrong structure: mod(n-(nie-over), nie-over) == over must holds") 
# n=9
# nie=3
# over=0
# filter(x -> x <= n-nie+1, vcat(1,(x -> x + (nie-over)).([1:nie-over:n-(nie-over);])))
# mod(n-(nie-over), nie-over) == over || error("wrong structure: mod(n-(nie-over), nie-over) == over must holds") 
# n=8
# nie=7
# over=6
# filter(x -> x <= n-nie+1, vcat(1,(x -> x + (nie-over)).([1:nie-over:n-(nie-over);])))
# mod(n-(nie-over), nie-over) == mod(over, nie-over) || error("wrong structure: mod(n-(nie-over), nie-over) == mod(over, nie-over) must holds") 
# n=8
# nie=7
# over=2

# part_vec(;n=n,nie=nie,overlapping=over)
# part_mat(;n=n,nie=nie,overlapping=over)