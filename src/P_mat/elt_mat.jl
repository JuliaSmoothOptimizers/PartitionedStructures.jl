module M_elt_mat


	abstract type Elt_mat{T} end

	# we assume at least the field nie, and indices
	get_indices(eem :: T ) where T <: Elt_mat = eem.indices
	get_indices(eem :: T, i::Int ) where T <: Elt_mat = eem.indices[i]
	get_nie(eem :: T ) where T <: Elt_mat = eem.nie
	
	set_indices!(eem :: T, indices :: Vector{Int}) where T <: Elt_mat = eem.indices = indices
	set_nie!(eem :: T, nie ::Int) where T <: Elt_mat = eem.nie = nie

	max_indices(em_set :: Vector{T}) where T <: Elt_mat = maximum(maximum.(get_indices.(em_set)))
	min_indices(em_set :: Vector{T}) where T <: Elt_mat = minimum(minimum.(get_indices.(em_set)))


	export Elt_mat 

	export get_indices, get_nie
	export set_indices!, set_nie!

	export max_indices, min_indices

end