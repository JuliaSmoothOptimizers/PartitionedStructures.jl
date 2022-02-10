module M_part_v

	using ..M_abstract_part_struct

	# we assume that each type T <: Part_v{T} possess at least a field N and n 
	abstract type Part_v{T} <: Part_struct{T} end 
		
	@inline get_v(pv :: T ) where T <: Part_v = pv.v
		
	@inline set_v!(pv :: T, v :: Vector{Y} ) where T <: Part_v{Y} where Y = pv.v = v
	@inline set_v!(pv :: T, i :: Int, value :: Y ) where T <: Part_v{Y} where Y = pv.v[i] = value
	@inline add_v!(pv :: T, i :: Int, value :: Y ) where T <: Part_v{Y} where Y = pv.v[i] += value

	@inline reset_v!(pv :: T ) where T <: Part_v{Y} where Y = pv.v .= (Y)(0)

	"""
			build_v(pv)
	Build the vector v from the partitionned vector pv.
	Call specialised method depending the type of the element vector inside pv
	For now if there is mix of elemental and internal element vectors it must be previously transform as internal partitioned vector.
	"""
	@inline build_v(pv :: T) where T <: Part_v = begin build_v!(pv); return get_v(pv) end
	@inline build_v!(pv :: T) where T <: Part_v = error("M_part_v.build_v!() should not be call")

	export Part_v
	export get_N, get_n, get_v
	export set_N!, set_n!, set_v!, add_v!, reset_v!
	export build_v!, build_v
end