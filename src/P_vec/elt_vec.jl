module M_elt_vec

	# we assume that each type T <: Elt_vec{T} possess at least a field vec and indices
	abstract type Elt_vec{T} end

	#generic getter/setter
	@inline get_vec(ev :: T) where T <: Elt_vec = ev.vec
	@inline get_indices(ev :: T) where T <: Elt_vec = ev.indices
	@inline get_vec(ev :: T, i::Int) where T <: Elt_vec = ev.vec[i]
	@inline get_indices(ev :: T, i::Int) where T <: Elt_vec = ev.indices[i]
	@inline get_nie(ev :: T) where T <: Elt_vec = ev.nie

	@inline set_vec!(ev :: T, vec :: Vector{Y}) where T <: Elt_vec where Y = ev.vec = vec
	@inline set_indices!(ev :: T, indices :: Vector{Int}) where T <: Elt_vec = ev.indices = indices
	@inline set_nie!(ev :: T, nie :: Int) where T <: Elt_vec = ev.nie = nie

	# get the max/min index of variable from the {indiceᵢ}ᵢ
	max_indices(elt_ev_set :: Vector{T}) where T <: Elt_vec = maximum(maximum.(get_indices.(elt_ev_set)))
	min_indices(elt_ev_set :: Vector{T}) where T <: Elt_vec = minimum(minimum.(get_indices.(elt_ev_set)))
		
	export Elt_vec
	export get_vec, get_indices, get_nie
	export set_vec!, set_indices!, set_nie!
	export max_indices, min_indices
end
