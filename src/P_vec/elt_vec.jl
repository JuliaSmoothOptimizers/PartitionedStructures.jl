module M_elt_vec

	using ..M_abstract_element_struct
	
	export set_vec!, get_vec, set_minus_vec!, set_add_vec!
	export Elt_vec

	# we assume that each type T <: Elt_vec{T} possess at least a field vec and indices
	abstract type Elt_vec{T} <: Element_struct{T} end

	#generic getter/setter
	@inline get_vec(ev :: T) where T <: Elt_vec = ev.vec
	@inline get_vec(ev :: T, i::Int) where T <: Elt_vec = ev.vec[i]

	@inline set_vec!(ev :: T, vec :: Vector{Y}) where T <: Elt_vec where Y = ev.vec .= vec
	@inline set_minus_vec!(ev :: T) where T <: Elt_vec = set_vec!(ev, - get_vec(ev))
	@inline set_add_vec!(ev :: T, vec :: Vector{Y}) where {T <: Elt_vec, Y <:Number}= ev.vec .+= vec	
	# @inline set_add_vec!(ev :: T, vec :: Vector{Y}) where {T <: Elt_vec, Y <:Number} =	set_vec!(ev, vec .+ get_vec(ev))
	
end
