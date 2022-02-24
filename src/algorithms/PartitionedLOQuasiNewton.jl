module PartitionedLOQuasiNewton

	using ..M_abstract_part_struct, ..M_elt_vec, ..M_part_mat, ..M_elt_mat
	using ..Utils
	using ..ModElemental_ev, ..ModElemental_pv
	using ..ModElemental_plom, ..ModElemental_plom_bfgs, ..ModElemental_elom_bfgs
	
	export PLBFGS_update, PLBFGS_update!
	export Part_update, Part_update!

	""" 
			PLBFGS_update(eplom_B, s, epv_y)
	Define the partitioned BFGS update of the partioned matrix eplom_B, given the step s and the element gradient difference epv_y
	"""
	PLBFGS_update(eplom_B :: Elemental_plom_bfgs{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}) where T = begin epm_copy = copy(eplom_B); PLBFGS_update!(epm_copy,epv_y,s); return epm_copy end 
	PLBFGS_update!(eplom_B :: Elemental_plom_bfgs{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}) where T = begin epv_s = epv_from_v(s, epv_y); PLBFGS_update!(eplom_B, epv_y, epv_s) end
	function PLBFGS_update!(eplom_B :: Elemental_plom_bfgs{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}) where T 
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


	"""
			Part_update(eplom_B, epv_y, s)
	Perform the partitionned update of eplom_B.
	eplom_B is build from LBFGS or LSR1 elemental element matrices.
	The update performed on eachh element matrix correspond to the linear operator associated.
	"""
	Part_update(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}) where Y <: Part_LO_mat{T} where T = begin epm_copy = copy(eplom_B); Part_update!(epm_copy,epv_y,s); return epm_copy end 
	Part_update!(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}) where Y <: Part_LO_mat{T} where T = begin epv_s = epv_from_v(s, epv_y); Part_update!(eplom_B, epv_y, epv_s) end
	function Part_update!(eplom_B :: Y, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}) where Y <: Part_LO_mat{T} where T 
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

end 