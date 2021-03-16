# include("src/P_vec/elemental_ev.jl")
module M_elemental_elt_vec

	using SparseArrays

 	using ..M_elt_vec, ..M_utils
	
	import Base.==	
	 
	# we assume that the values of vec are associate to indices.
	mutable struct Elemental_elt_vec{T} <: Elt_vec{T}
		vec :: Vector{T} # nᵢᴱ
		indices :: Vector{Int} # nᵢᴱ
		nie :: Int
	end

	(==)(eev1 :: Elemental_elt_vec{T}, eev2 :: Elemental_elt_vec{T}) where T = (get_indices(eev1) == get_indices(eev2)) && (get_vec(eev1) == get_vec(eev2)) && (get_nie(eev1) == get_nie(eev2))		

	@inline new_eev(nᵢ::Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(rand(T,nᵢ), rand(1:n,nᵢ), nᵢ)
	@inline ones_eev(nᵢ::Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(ones(T,nᵢ), rand(1:n,nᵢ), nᵢ)
	


	"""
		SparseArrays.SparseVector interface
	"""
	function eev_from_sparse_vec(v :: SparseVector{T,Y}) where {T, Y}
		(indices, vec) = findnz(v)
		nie = length(indices)
		eev = Elemental_elt_vec{T}(Vector{T}(vec), Vector{Int}(indices), nie)
		return eev
	end
	sparse_vec_from_eev(eev :: Elemental_elt_vec{T}; n::Int=maximum(get_indices(eev))) where T = sparsevec(get_indices(eev), get_vec(eev), n)
		
	
# type
	export Elemental_elt_vec
# comfort
	export new_eev, ones_eev
# on var
	export max_indices, min_indices
# interface with SparseArrays
	export eev_from_sparse_vec, sparse_vec_from_eev

end