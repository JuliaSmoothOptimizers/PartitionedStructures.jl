module M_part_mat

	using ..M_abstract_part_struct

	abstract type Part_mat{T} <: Part_struct{T} end

	# @inline get_N(pm :: T ) where T <: Part_mat = pm.N
	# @inline get_n(pm :: T ) where T <: Part_mat = pm.n
	@inline get_permutation(pm :: T) where T <: Part_mat = pm.permutation
	@inline get_spm(pm :: T) where T <: Part_mat = @error("sould not be called")
	@inline set_spm!(pm :: T) where T <: Part_mat = @error("sould not be called")

	@inline set_N!(pm :: T, N :: Int) where T <: Part_mat = pm.N = N
	@inline set_n!(pm :: T, n :: Int ) where T <: Part_mat = pm.n = n
	@inline set_permutation!(pm :: T, p :: Vector{Int}) where T <: Part_mat = pm.permutation = perm

	PBFGS(pm :: T, s, y) where T <: Part_mat = @error("PFBGS non défini")
	PLBFGS(pm :: T, s, y) where T <: Part_mat = @error("PFBGS non défini")


	export Part_mat

	export get_N, get_n, get_permutation
	export get_spm, set_spm!
	export set_N!, set_n!, set_permutation!
end