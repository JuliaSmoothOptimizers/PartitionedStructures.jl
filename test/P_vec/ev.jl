using PartitionnedStructures
using PartitionnedStructures.M_elemental_elt_vec
using PartitionnedStructures.M_internal_elt_vec
using PartitionnedStructures.M_elt_vec

using SparseArrays

@testset "Test ev getter/setter" begin
	nᵢᴱ = 5
	nᵢᴵ = 2

	# elemental
	ev1 = new_elt_ev(nᵢᴱ)
	v1 = rand(nᵢᴱ)
	i1 = [1:nᵢᴱ:nᵢᴱ^2;]
	set_vec!(ev1,v1)
	set_indices!(ev1,i1)

	@test get_vec(ev1) == v1
	@test get_indices(ev1) == i1

	# internal
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


@testset "Test ev min/max indices" begin
	for i in 1:20
		nᵢ = i # n <= nᵢ <= 25
		N = 30

		elt_ev_set = [ones_elt_ev(nᵢ) for j in 1:N]

		n_max = max_indices(elt_ev_set)
		n_min = min_indices(elt_ev_set)
		@test n_max <= nᵢ^2
		@test n_min <= nᵢ^2
		@test n_max >= 0	
		@test n_min >= 0
	end 
end

@testset "interface SparseVector" begin
	n = 20
	sx = spzeros(Float64, n)
	[sx[i] = i for i in 1:2:7]

	ex = elt_ev_from_sparse_vec(sx)
	_sx = sparse_vec_from_ev(ex; n=n)
	_ex = elt_ev_from_sparse_vec(sx)

	@test sx == _sx	
	@test ex == _ex

	# hardcode
	sx = sparsevec([1:2:5;],[1:3;],n)
	ex = Elemental_elt_vec([1:3;],[1:2:5;],3)
	_sx = sparse_vec_from_ev(ex; n=n)
	_ex = elt_ev_from_sparse_vec(sx)
	__sx = sparse_vec_from_ev(_ex; n=n)
	__ex = elt_ev_from_sparse_vec(_sx)
	
	@test _sx == sx
	@test _ex == ex
end