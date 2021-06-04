module M_1_parallel

using ..M_part_mat, ..M_part_v, ..M_elemental_pv, ..M_elemental_pm
using ..M_link

	


	function first_parallel(epm_A :: Elemental_pm{T}, epv_b :: Elemental_pv{T}) where T	
		epv_x = similar(epv_b)
		first_parallel!(epm_A, epv_b, epv_x)
	end 

	function first_parallel!(epm_A :: Elemental_pm{T}, epv_b :: Elemental_pv{T}, epv_x :: Elemental_pv{T}) where T
		check_pev_pem(epm_A, epv_b)
		N = get_N(epm_A)
		for i in [1:N;]
			set_eev!(epv_x, i, get_eem_set_hie(epm_A,i)\get_eev_value(epv_b,i))
		end 
	end 

	export first_parallel, first_parallel!

end 