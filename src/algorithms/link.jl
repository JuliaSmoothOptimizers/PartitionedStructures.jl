module Link
	using ..M_part_mat, ..M_part_v, ..ModElemental_pv, ..ModElemental_pm

	@inline check_epv_epm(epm :: Elemental_pm{T}, epv :: Elemental_pv{T}) where T = get_N(epm) == get_N(epv) && get_n(epm) == get_n(epv)
	#todo finish full_check_epv_epm en comparant les indices
	@inline full_check_epv_epm(epm :: Elemental_pm{T}, epv :: Elemental_pv{T}) where T = check_epv_epm(epm,epv) && ModElemental_pm.get_component_list(epm) == ModElemental_pv.get_component_list(epv)



	function mul_epm_epv(epm :: Elemental_pm{T}, epv :: Elemental_pv{T}) where T
		check_epv_epm(epm,epv) || error("Structure differ epm/epv")
		epv_tmp = similar(epv)
		mul_epm_epv!(epv_tmp, epm, epv)
		return epv_tmp
	end

	#todo version trop courte
	function mul_epm_epv!(epv_tmp :: Elemental_pv{T}, epm :: Elemental_pm{T}, epv :: Elemental_pv{T}) where T
		
		set_spm!()
	end 

	function mul_epm_vector(epm :: Elemental_pm{T}, x :: Vector{T}) where T 
		set_spm!(epm)
		spm = get_spm(epm)
		return spm*x
	end 	

	function create_epv_epm(;n=9,nie=5,overlapping=1,mul_m=5., mul_v=100.)
		epm = part_mat(;n=n,nie=nie,overlapping=overlapping,mul=mul_m)
		epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
		return (epm,epv)
	end 


	function create_epv_epm_rand(;n=9,nie=5,overlapping=1,range_mul_m=nie:2*nie, mul_v=100.)
		epm = part_mat(;n=n,nie=nie,overlapping=overlapping,mul=rand(range_mul_m))
		epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
		return (epm,epv)
	end 




	export check_epv_epm, full_check_epv_epm
	export mul_epm_epv, mul_epm_epv!, mul_epm_vector
	export create_epv_epm, create_epv_epm_rand
end 