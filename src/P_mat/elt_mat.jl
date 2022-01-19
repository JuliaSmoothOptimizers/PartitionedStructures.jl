module M_elt_mat

	using ..M_abstract_element_struct

	abstract type Elt_mat{T} <: Element_struct{T} end

	# we assume at least the field nie, and indices
	@inline get_mat(elt_mat :: T) where T <: Elt_mat = error("should not be called M_elt_mat")

	export Elt_mat 

	export get_mat
	
end