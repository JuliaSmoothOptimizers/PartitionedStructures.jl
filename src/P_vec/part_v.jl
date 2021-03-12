module M_part_v

	# we assume that each type T <: Part_v{T} possess at least a field N and n 
	abstract type Part_v{T} end 

	@inline get_N(p :: T ) where T <: Part_v = p.N
	@inline get_n(p :: T ) where T <: Part_v = p.n
	@inline get_v(p :: T ) where T <: Part_v = p.v

	@inline set_N!(p :: T, N :: Int ) where T <: Part_v = p.N = N
	@inline set_n!(p :: T, n :: Int ) where T <: Part_v = p.n = n
	@inline set_v!(p :: T, v :: Vector{Y} ) where T <: Part_v{Y} where Y = p.v = v
	@inline set_v!(p :: T, i :: Int, value :: Y ) where T <: Part_v{Y} where Y = p.v[i] = value
	@inline add_v!(p :: T, i :: Int, value :: Y ) where T <: Part_v{Y} where Y = p.v[i] += value

	@inline reset_v!(p :: T ) where T <: Part_v{Y} where Y = p.v .= zeros(Y,get_n(p))

	@inline build_v!(p :: T) where T <: Part_v = error("part_v should not be call")

	# using ..M_elt_vec
	# mutable struct Part_v{T, Y <: Vector{Elt_vec{T}}}
	# 	_N :: Int
	# 	n :: Int
	# 	set_ev :: Y
	# end
	
	export Part_v
	export get_N, get_n, get_v
	export set_N!, set_n!, set_v!, add_v!, reset_v!
	export build_v!
end