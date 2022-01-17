# include("src/P_vec/elemental_ev.jl")
module M_internal_elt_vec

	using SparseArrays, LinearAlgebra

 	using ..M_elt_vec, ..ModElemental_ev, ..M_abstract_element_struct

	import Base.==

	# distinguish from internal_elt_vec
	# Be careful with the order of the indices
	mutable struct Internal_elt_vec{T} <: Elt_vec{T}
		vec :: Vector{T} # size nᵢᴵ
		indices :: Vector{Int} # size nᵢᴱ
		# lin_comb :: Array{T,2} # size nᵢᴵ× nᵢᴱ
		lin_comb :: SparseMatrixCSC{T,Int} # size nᵢᴵ× nᵢᴱ
		nie :: Int
		nii :: Int
		_tmp :: Vector{T} # size nᵢᴱ
	end
	
	@inline get_lin_comb(iev :: Internal_elt_vec{T}) where T = iev.lin_comb
	@inline get_nii(iev :: Internal_elt_vec{T}) where T = iev.nii
	@inline get_tmp(iev :: Internal_elt_vec{T}) where T = iev._tmp
	@inline get_tmp(iev :: Internal_elt_vec{T}, i :: Int) where T = iev._tmp[i]

	@inline set_lin_comb!(iev :: Internal_elt_vec{T}, lin_comb :: Array{T,2}) where T = iev.lin_comb = lin_comb
	@inline set_nii!(iev :: Internal_elt_vec{T}, nii::Int) where T = iev.nii = nii

	(==)(iev1 :: Internal_elt_vec{T}, iev2 :: Internal_elt_vec{T}) where T = (get_vec(iev1)==get_vec(iev2)) && (get_indices(iev1)==get_indices(iev2)) && (get_lin_comb(iev1) == get_lin_comb(iev2)) && (get_nie(iev1)==get_nie(iev2)) && (get_nii(iev1)==get_nii(iev2))


	@inline new_iev(nᵢᴱ:: Int, nᵢᴵ:: Int; T=Float64, n=nᵢᴱ^2, prop=0.5) = Internal_elt_vec(rand(T,nᵢᴵ),rand(1:n,nᵢᴱ), sprand(T,nᵢᴵ, nᵢᴱ, prop), nᵢᴱ, nᵢᴵ, rand(T,nᵢᴱ))
	@inline ones_iev(nᵢᴱ:: Int, nᵢᴵ:: Int; T=Float64, n=nᵢᴱ^2, prop=0.5) = Internal_elt_vec(ones(T,nᵢᴵ),rand(1:n,nᵢᴱ), sprand(T,nᵢᴵ, nᵢᴱ, prop), nᵢᴱ, nᵢᴵ, rand(T,nᵢᴱ))
	iev_from_sparse_vec(sv ::SparseVector{T,Y}) where {T,Y} = iev_from_eev(eev_from_sparse_vec(sv)) 

	function iev_from_eev(eev :: Elemental_elt_vec{T}) where T 
		nie = get_nie(eev)
		indices = get_indices(eev)
		vec = get_vec(eev)
		lin_com = spzeros(T,nie,nie) # identity matrix Matrix(I,n,n) didn't work
		[lin_com[i,i] = 1 for i in 1:nie] 
		_tmp = rand(T,nie)
		iev = Internal_elt_vec{T}(vec, indices, lin_com, nie, nie, _tmp)
		return iev
	end 

	build_tmp!(iev :: Internal_elt_vec{T}) where T = mul!(iev._tmp, transpose(iev.lin_comb), iev.vec) 


	export Internal_elt_vec	

	export get_lin_comb, get_nii, get_tmp
	export set_lin_comb!, set_nii

	export new_iev, ones_iev, iev_from_eev, iev_from_sparse_vec

	export build_tmp!
	

end