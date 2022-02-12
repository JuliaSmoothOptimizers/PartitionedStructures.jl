module Link
	using ..M_part_mat, ..M_part_v
	using ..ModElemental_pv, ..ModElemental_pm, ..ModElemental_plom_bfgs, ..ModElemental_plom
	using ..ModElemental_ev
	using ..M_abstract_part_struct, ..M_abstract_element_struct

	export mul_epm_epv, mul_epm_epv!, mul_epm_vector, mul_epm_vector!
	export create_epv_epm, create_epv_epm_rand, create_epv_eplom_bfgs, create_epv_eplom, epv_from_epm, epv_from_eplom
		
	function epv_from_epm(epm :: T) where T <: Part_mat{Y} where Y <: Number
		N = get_N(epm)
		n = get_n(epm)
		eev_set = Vector{Elemental_elt_vec{Y}}(undef,N)
		for i in 1:N
			eesi = get_ee_struct(epm,i)
			indices = get_indices(eesi)
			nie = get_nie(eesi)
			eev_set[i] = Elemental_elt_vec{Y}(rand(Y,nie), indices, nie)
		end 
		component_list = M_abstract_part_struct.get_component_list(epm)
		v = rand(Y,n)
		perm = [1:n;]
		epv = Elemental_pv{Y}(N, n, eev_set, v, component_list,perm)
		return epv
	end
	
	function mul_epm_vector(epm :: T, x :: Vector{Y}) where T <: Part_mat{Y} where Y <: Number
		epv = epv_from_epm(epm)
		mul_epm_vector(epm, epv, x)
	end 
	function mul_epm_vector(epm :: T, epv :: Elemental_pv{Y}, x :: Vector{Y}) where T <: Part_mat{Y} where Y <: Number
		g = similar(x)
		mul_epm_vector!(g, epm, epv, x)
		return g
	end 	
	function mul_epm_vector!(res :: Vector{Y}, epm :: T, x :: Vector{Y}) where T <: Part_mat{Y} where Y <: Number
		epv = epv_from_epm(epm)
		mul_epm_vector!(res,epm,epv,x)
	end

	function mul_epm_vector!(res :: Vector{Y}, epm :: T, epv :: Elemental_pv{Y}, x :: Vector{Y}) where T <: Part_mat{Y} where Y <: Number
		epv_from_v!(epv,x)
		mul_epm_epv!(epv,epm,epv)
		build_v!(epv)
		res .= get_v(epv)
	end 

	function mul_epm_epv(epm :: T, epv :: Elemental_pv{Y}) where T <: Part_mat{Y} where Y <: Number
		epv_res = similar(epv)
		mul_epm_epv!(epv_res, epm, epv)
		return epv_res
	end

	function mul_epm_epv!(epv_res :: Elemental_pv{Y}, epm :: T, epv :: Elemental_pv{Y}) where T <: Part_mat{Y} where Y <: Number
		full_check_epv_epm(epm,epv) || error("Structure differ epm/epv")
		N = get_N(epm)
		for i in 1:N
			Bie = get_ee_struct_Bie(epm,i)
			vie = get_eev_value(epv, i)
			set_eev!(epv_res, i,Bie*vie)
		end
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

	function create_epv_eplom_bfgs(;n=9,nie=5,overlapping=1,range_mul_m=nie:2*nie, mul_v=100.)
		eplom = PLBFGS_eplom(;n=n,nie=nie,overlapping=overlapping)
		epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
		return (eplom,epv)
	end 

	function create_epv_eplom(;n=9,nie=5,overlapping=1,range_mul_m=nie:2*nie, mul_v=100.)
		eplom = PLBFGSR1_eplom(;n=n,nie=nie,overlapping=overlapping)
		epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
		return (eplom,epv)
	end

end 