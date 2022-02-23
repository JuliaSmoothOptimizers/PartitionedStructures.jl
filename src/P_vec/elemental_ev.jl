module ModElemental_ev

	using SparseArrays, StatsBase
 	using ..M_abstract_element_struct, ..M_elt_vec, ..Utils
	
	import Base.==, Base.copy, Base.similar
	
	export Elemental_elt_vec
	export ones_eev, new_eev, specific_ones_eev
	export create_eev, eev_from_sparse_vec, sparse_vec_from_eev
	 
	# we assume that the values of vec are associate to indices.
	mutable struct Elemental_elt_vec{T} <: Elt_vec{T}
		vec :: Vector{T} # length(vec) == nᵢᴱ
		indices :: Vector{Int} # length(indices) == nᵢᴱ
		nie :: Int
	end

	@inline (==)(eev1 :: Elemental_elt_vec{T}, eev2 :: Elemental_elt_vec{T}) where T = (get_indices(eev1) == get_indices(eev2)) && (get_vec(eev1) == get_vec(eev2)) && (get_nie(eev1) == get_nie(eev2))		
	@inline similar(eev :: Elemental_elt_vec{T}) where T = Elemental_elt_vec{T}(Vector{T}(undef,get_nie(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))
	@inline copy(eev :: Elemental_elt_vec{T}) where T = Elemental_elt_vec{T}(Vector{T}(get_vec(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))

	"""
			new_eev(nie; T, n)
	Create an elemental element vector, with `nie` randoms values placed at indices within the range `1:n`.
	"""
	@inline new_eev(nᵢ :: Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(rand(T,nᵢ), sample(1:n, nᵢ, replace = false), nᵢ)
	"""
			ones_eev(nie; T, n)
	Create an elemental element vector, with `nie` values at `1` placed at indices within the range `1:n`.
	"""
	@inline ones_eev(nᵢ :: Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(ones(T,nᵢ), sample(1:n, nᵢ, replace = false), nᵢ)
	"""
			specific_ones_eev(nie, index; T, mul)
	Create an elemental element vector, with `nie` randoms values multiplied by `mul` placed at indices in range `index:index+nie`, with .
	"""
	@inline specific_ones_eev(nie :: Int,index :: Int; T=Float64, mul :: Float64=1.) = Elemental_elt_vec((xi -> mul*xi).(rand(T, nie)), [index:index+nie-1;], nie)	

	"""
			eev_from_sparse_vec(sparse_vec)
	eev_from_sparse_vec is an interface with SparseArrays.SparseVector.
	The indices and the values of the elemental element vector are define using findnz(sparse_vec).
	"""
	function eev_from_sparse_vec(v :: SparseVector{T,Y}) where {T, Y}
		(indices, vec) = findnz(v)
		nie = length(indices)
		eev = Elemental_elt_vec{T}(Vector{T}(vec), Vector{Int}(indices), nie)
		return eev
	end

	"""
			sparse_vec_from_eev(eev)
	Create a sparse vector from the element element vector eev.
	"""
	sparse_vec_from_eev(eev :: Elemental_elt_vec{T}; n :: Int=maximum(get_indices(eev))) where T = sparsevec(get_indices(eev), get_vec(eev), n)
		
	"""
			create_eev(vector_indices)
	Create a random elemental element vector from `vector_indices`.
	"""
	function create_eev(elt_var :: Vector{Int}; type=Float64)
	  nie = length(elt_var)
	  eev_value = rand(type, nie)
	  eev = Elemental_elt_vec{type}(eev_value, elt_var, nie)
	  return eev
	end 

end