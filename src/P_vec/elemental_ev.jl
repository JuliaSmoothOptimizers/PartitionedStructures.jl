module ModElemental_ev

	using SparseArrays, StatsBase

 	using ..M_elt_vec, ..Utils, ..M_abstract_element_struct
	
	import Base.==, Base.copy, Base.similar
	 
	# we assume that the values of vec are associate to indices.
	mutable struct Elemental_elt_vec{T} <: Elt_vec{T}
		vec :: Vector{T} # nᵢᴱ
		indices :: Vector{Int} # nᵢᴱ
		nie :: Int
	end

	# get_vec(eev :: Elemental_elt_vec{T}) where T = eev.vec # define in the abstract type

	@inline (==)(eev1 :: Elemental_elt_vec{T}, eev2 :: Elemental_elt_vec{T}) where T = (get_indices(eev1) == get_indices(eev2)) && (get_vec(eev1) == get_vec(eev2)) && (get_nie(eev1) == get_nie(eev2))		
	@inline similar(eev :: Elemental_elt_vec{T}) where T = Elemental_elt_vec{T}(Vector{T}(undef,get_nie(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))
	@inline copy(eev :: Elemental_elt_vec{T}) where T = Elemental_elt_vec{T}(Vector{T}(get_vec(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))

	@inline new_eev(nᵢ::Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(rand(T,nᵢ), sample(1:n,nᵢ,replace = false), nᵢ)
	@inline ones_eev(nᵢ::Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(ones(T,nᵢ), sample(1:n,nᵢ,replace = false), nᵢ)
	@inline specific_ones_eev(nie::Int,index::Int; T=Float64, mul::Float64=1.) = Elemental_elt_vec((xi -> mul*xi).(rand(T,nie)), [index:index+nie-1;], nie)
	
	@inline set_vec_eev!(eev :: Elemental_elt_vec{T}, i :: Int, val :: T) where T = eev.vec[i] = val
	@inline set_vec_eev!(eev :: Elemental_elt_vec{T}, vec :: Vector{T}) where T = eev.vec = vec


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
		
	function create_eev(elt_var::Vector{Int}; type=Float64)
	  nie = length(elt_var)
	  eev_value = rand(nie)
	  eev = Elemental_elt_vec{type}(eev_value, elt_var, nie)
	  return eev
	end 

# type
	export Elemental_elt_vec

	export set_vec_eev!
# comfort
	export new_eev, ones_eev, specific_ones_eev
# interface with SparseArrays
	export eev_from_sparse_vec, sparse_vec_from_eev, create_eev

end