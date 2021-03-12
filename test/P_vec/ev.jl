using PartitionnedStructures
using PartitionnedStructures.M_elemental_elt_vec
using PartitionnedStructures.M_internal_elt_vec
using PartitionnedStructures.M_elt_vec

@testset "Test ev" begin
	nᵢᴱ = 5
	nᵢᴵ = 2

	ev1 = new_elt_ev(nᵢᴱ)
	v1 = rand(nᵢᴱ)
	i1 = [1:nᵢᴱ:nᵢᴱ^2;]
	set_vec!(ev1,v1)
	set_indices!(ev1,i1)

	@test get_vec(ev1) == v1
	@test get_indices(ev1) == i1


	ev2 = new_int_ev(nᵢᴱ,nᵢᴵ)
	v2 = rand(nᵢᴵ)
	i2 = [1:nᵢᴱ:nᵢᴱ^2;]
	lc2 = rand(nᵢᴵ, nᵢᴱ)
	set_vec!(ev2,v2)
	set_indices!(ev2,i2)
	set_lin_comb!(ev2,lc2)

	@test get_vec(ev2) == v2
	@test get_indices(ev2) == i2
	@test get_lin_comb(ev2) == lc2
	
end 