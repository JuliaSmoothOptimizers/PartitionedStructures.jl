module M_elt_vec

	using ..M_abstract_element_struct
	# we assume that each type T <: Elt_vec{T} possess at least a field vec and indices
	abstract type Elt_vec{T} <: Element_struct{T} end

	#generic getter/setter
	@inline get_vec(ev :: T) where T <: Elt_vec = ev.vec
	
	@inline get_vec(ev :: T, i::Int) where T <: Elt_vec = ev.vec[i]

	@inline set_vec!(ev :: T, vec :: Vector{Y}) where T <: Elt_vec where Y = ev.vec = vec

	
	export set_vec!, get_vec
	export Elt_vec

end
