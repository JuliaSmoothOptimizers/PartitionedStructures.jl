# include("src/P_vec/elemental_ev.jl")
module M_elemental_elt_vec

	using SparseArrays

 	using ..M_elt_vec, ..M_utils

	# we assume that the values of vec are associate to indices.
	mutable struct Elemental_elt_vec{T} <: Elt_vec{T}
		vec :: Vector{T} # nᵢᴱ
		indices :: Vector{Int} # nᵢᴱ
		nie :: Int
	end

	@inline new_elt_ev(nᵢ::Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(rand(T,nᵢ), rand(1:n,nᵢ), nᵢ)
	@inline ones_elt_ev(nᵢ::Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(ones(T,nᵢ), rand(1:n,nᵢ), nᵢ)
	
	# get the max/min index of variable from the {indiceᵢ}ᵢ
	max_indices(elt_ev_set :: Vector{Elemental_elt_vec{T}}) where T = maximum(maximum.(get_indices.(elt_ev_set)))
	min_indices(elt_ev_set :: Vector{Elemental_elt_vec{T}}) where T = minimum(minimum.(get_indices.(elt_ev_set)))


	"""
		SparseArrays.SparseVector interface
	"""
	function elt_ev_from_sparse_vec(v :: SparseVector{T,Y}) where {T, Y}
		(indices, vec) = findnz(v)
		nie = length(indices)
		ev = Elemental_elt_vec{T}(Vector{T}(vec), Vector{Int}(indices), nie)
		return ev
	end
	sparse_vec_from_ev(ev :: Elemental_elt_vec{T}; n::Int=maximum(get_indices(ev))) where T = sparsevec(get_indices(ev), get_vec(ev), n)
		
	import Base.==	
	# Use of sum(x::BitArray) == length(x) because of BitArray (ie: can't use mapreduce(my_and, Vector{Bool}))
	(==)(ev1 :: Elemental_elt_vec{T}, ev2 :: Elemental_elt_vec{T}) where T = (sum(get_indices(ev1) .== get_indices(ev2)) == get_nie(ev2)) && (sum( get_vec(ev1) .== get_vec(ev2))  == get_nie(ev2)) && (get_nie(ev1) == get_nie(ev2))		

	

# type
	export Elemental_elt_vec
# comfort
	export new_elt_ev, ones_elt_ev
# on var
	export max_indices, min_indices
# interface with SparseArrays
	export elt_ev_from_sparse_vec, sparse_vec_from_ev

end