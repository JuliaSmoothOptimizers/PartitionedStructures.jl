using PartitionedStructures
using PartitionedStructures.ModElemental_elom
using PartitionedStructures.ModElemental_plom
using PartitionedStructures.Utils, PartitionedStructures.Link


using SparseArrays, LinearAlgebra, LinearOperators


@testset "test elemental element linear operator matrix" begin
	for index in 3:3:15
		for T in [Float16,Float32,Float64]
			nie=5
			indices = [index:1:index+nie-1;]
			Bie = LinearOperators.LBFGSOperator(T, nie)
			@test Elemental_elom{T}(nie,indices,Bie) == LBFGS_eelom(nie;T=T, index=index)
		end
	end

	a = LBFGS_eelom(5)
	A = Matrix(get_Bie(a))
	@test A == transpose(A)
end 



@testset "test elemental partitioned linear operator matrix" begin
	n=10
	nie=4
	over=2
	(eplom_B,epv_y) = create_epv_eplom(; n=n, nie=nie, overlapping=over)
	B = Matrix(eplom_B)
	@test B == transpose(B)

	@test mapreduce((x -> x>0), my_and, eigvals(B)) # test definite positiveness
end