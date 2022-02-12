
module M_abstract_part_struct
	using ..M_abstract_element_struct
	import Base.==

	export Part_struct
	export get_n, get_N, get_component_list
	export check_epv_epm, full_check_epv_epm
	export initialize_component_list!, get_ee_struct

	abstract type Part_struct{T} end 

	get_n(ps :: T) where T <: Part_struct = ps.n
	get_N(ps :: T) where T <: Part_struct = ps.N
	get_component_list(ps :: T) where T <: Part_struct = ps.component_list
	get_component_list(ps :: T, i::Int) where T <: Part_struct = ps.component_list[i]

	(==)(ps1 :: T, ps2 :: T) where T <: Part_struct = get_n(ps1)==get_n(ps2) && get_N(ps1)==get_N(ps2)

	@inline check_epv_epm(epm :: Y, epv :: Z) where Y <: Part_struct where Z <: Part_struct = get_N(epm) == get_N(epv) && get_n(epm) == get_n(epv)
	@inline full_check_epv_epm(ep1 :: Y, ep2 :: Z) where Y <: Part_struct where Z <: Part_struct = check_epv_epm(ep1,ep2) && get_component_list(ep1) == get_component_list(ep2)

	initialize_component_list!(ps::T) where T <: Part_struct = @error("should not be called")
	get_ee_struct(ps::T) where T <: Part_struct = @error("should not be called")
end