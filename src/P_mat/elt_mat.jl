module M_elt_mat

	using ..M_abstract_element_struct

	abstract type Elt_mat{T} <: Element_struct{T} end

	# we assume at least the field nie, indices and Bie
	@inline get_Bie(elt_mat :: T) where T <: Elt_mat = elt_mat.Bie

	export Elt_mat 
	export get_Bie
	
end