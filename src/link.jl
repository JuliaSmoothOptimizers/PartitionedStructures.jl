module M_link
	using ..M_part_mat, ..M_part_v, ..M_elemental_pv, ..M_elemental_pm

	@inline check_pev_pem(epm :: Elemental_pm{T}, epv :: Elemental_pv{T}) where T = get_N(epm) == get_N(epv) && get_n(epm) == get_n(epv)






	function create_epv_epm(;n=9,nie=5,overlapping=1,mul=5.)
		epm = part_mat(;n=n,nie=nie,overlapping=overlapping,mul=mul)
		epv = part_vec(;n=n,nie=nie,overlapping=overlapping)
		return (epm,epv)
	end 







	export check_pev_pem
	export create_epv_epm
end 