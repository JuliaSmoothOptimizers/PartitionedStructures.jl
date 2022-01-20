using PartitionedStructures
using PartitionedStructures.ModElemental_elom_bfgs, PartitionedStructures.ModElemental_elom_sr1
using PartitionedStructures.ModElemental_plom_bfgs, PartitionedStructures.ModElemental_plom
using PartitionedStructures.Utils, PartitionedStructures.Link


using SparseArrays, LinearAlgebra, LinearOperators


@testset "test elemental element linear operator matrix" begin
	for index in 3:3:15
		for T in [Float16,Float32,Float64]
			nie=5
			indices = [index:1:index+nie-1;]
			Bie_bfgs = LinearOperators.LBFGSOperator(T, nie)
			Bie_sr1 = LinearOperators.LSR1Operator(T, nie)
			@test Elemental_elom_bfgs{T}(nie,indices,Bie_bfgs) == LBFGS_eelom(nie;T=T, index=index)
			@test Elemental_elom_sr1{T}(nie,indices,Bie_sr1) == LSR1_eelom(nie;T=T, index=index)
		end
	end

	a = LBFGS_eelom(5)
	A = Matrix(get_Bie(a))
	@test A == transpose(A)

	a = LSR1_eelom(5)
	A = Matrix(get_Bie(a))
	@test A == transpose(A)
end 



@testset "test elemental partitioned linear operator matrix" begin
	n=10
	nie=4
	over=2
	(eplom_B,epv_y) = create_epv_eplom_bfgs(; n=n, nie=nie, overlapping=over)
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