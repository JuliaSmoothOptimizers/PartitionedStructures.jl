module M_abstract_part_struct

	import Base.==

	abstract type Part_struct{T} end 

	get_n(ps :: T) where T <: Part_struct = ps.n
	get_N(ps :: T) where T <: Part_struct = ps.N

	(==)(ps1 :: T, ps2 :: T) where T <: Part_struct = get_n(ps1)==get_n(ps2) && get_N(ps1)==get_N(ps2)
	export Part_struct

	export get_n, get_N

end


module M_abstract_element_struct

	abstract type Element_struct{T} end

	@inline get_indices(elt :: T) where T <: Element_struct = elt.indices	
	@inline get_indices(elt :: T, i::Int) where T <: Element_struct = elt.indices[i]
	@inline get_nie(elt :: T) where T <: Element_struct = elt.nie
	
	@inline set_indices!(elt :: T, indices :: Vector{Int}) where T <: Element_struct = elt.indices = indices
	@inline set_nie!(elt :: T, nie :: Int) where T <: Element_struct = elt.nie = nie

	# get the max/min index of variable from the {indiceᵢ}ᵢ
	max_indices(elt_set :: Vector{T}) where T <: Element_struct = isempty(elt_set) ? 0 : maximum(maximum.(get_indices.(elt_set))) 
	min_indices(elt_set :: Vector{T}) where T <: Element_struct = isempty(elt_set) ? 0 : minimum(minimum.(get_indices.(elt_set)))

	export Element_struct

	export get_indices, get_nie
	export set_indices!, set_nie!

	export max_indices, min_indices

end