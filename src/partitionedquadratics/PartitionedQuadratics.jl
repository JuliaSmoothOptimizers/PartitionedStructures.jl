module ModPartitionedQuadratics

	using ..M_part_mat, ..M_part_v
	using ..ModElemental_ev, ..ModElemental_em
	using ..ModElemental_pm, ..ModElemental_pv
	using ..ModElemental_plom, ..ModElemental_plom_bfgs
	using ..M_abstract_element_struct, ..M_abstract_part_struct

	function elemental_quadratic(f :: T, g :: Elemental_ev, s :: Elemental_ev, m :: Elemental_em)
		mq = f + dot(get_vec(g), get_vec(s)) + 1/2 * dot(get_vec(s), get_Bie(m) * get_vec(s))
		return mq 
	end 

	function quadratic(pf :: Vector{T}, epm :: Part_mat{T}, epv_g :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}) where T <: Number
		full_check_epv_epm(epm,epv_g) || error("Structure differ epm/epv")
		full_check_epv_epm(epm,epv_s) || error("Structure differ epm/epv")
		N = get_N(epm)
		length(pf) = N || error("Structure differ epm/epv")    
    n = get_n(epm)    
		acc = 0
    for i in 1:N
      Bi = get_ee_struct(epm,i)
			gi = get_ee_struct(epv_g,i)
			si = get_ee_struct(epv_g,i)
      mi = elemental_quadratic(pf[i],gi,si,Bi)
			acc += mi
    end 
		return acc
  end
  


end 