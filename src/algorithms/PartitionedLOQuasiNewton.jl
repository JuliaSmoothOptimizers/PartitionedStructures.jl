module PartitionedLOQuasiNewton

using ..M_abstract_part_struct, ..M_elt_vec
using ..Utils
using ..ModElemental_ev, ..ModElemental_pv
using ..ModElemental_plom, ..ModElemental_elom


	""" 
			PLBFGS(eplom_B, s, epv_y)
	Define the partitioned BFGS update of the partioned matrix eplom_B, given the step s and the element gradient difference epv_y
	"""
	PLBFGS(eplom_B :: Elemental_plom{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}) where T = begin epm_copy = copy(eplom_B); PLBFGS!(epm_copy,epv_y,s); return epm_copy end 
	PLBFGS!(eplom_B :: Elemental_plom{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}) where T = begin epv_s = epv_from_v(s, epv_y); PLBFGS!(eplom_B, epv_y, epv_s) end
	function PLBFGS!(eplom_B :: Elemental_plom{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}) where T 
		full_check_epv_epm(eplom_B,epv_y) || @error("differents partitioned structures between eplom_B and epv_y")
		full_check_epv_epm(eplom_B,epv_s) || @error("differents partitioned structures between eplom_B and epv_s")
		N = get_N(eplom_B)
		for i in 1:N
			Bi = get_Bie(get_eelom_set(eplom_B, i))
			si = get_vec(get_eev(epv_s,i))
			yi = get_vec(get_eev(epv_y,i))
			push!(Bi, si, yi)
		end 
	end


	export PLBFGS, PLBFGS!

end 