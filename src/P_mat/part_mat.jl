module M_part_mat

	abstract type Part_mat{T} end

	get_N(pm :: T ) where T <: Part_mat = pm.N
	get_n(pm :: T ) where T <: Part_mat = pm.n

	set_N!(pm :: T, N :: Int) where T <: Part_mat = pm.N = N
	set_n!(pm :: T, n :: Int ) where T <: Part_mat = pm.n = n

	export Part_mat

	export get_N, get_n
	export set_N!, set_n!
end