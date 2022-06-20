module PartitionedQuasiNewton

using LinearAlgebra
using ..M_abstract_part_struct, ..M_elt_vec, ..M_elt_mat
using ..Utils, ..Link
using ..ModElemental_em, ..ModElemental_ev
using ..ModElemental_pm, ..ModElemental_pv

export PBFGS_update, PBFGS_update!
export PSR1_update, PSR1_update!
export PSE_update, PSE_update!

""" 
		B = PBFGS_update(epm_B, epv_y, s)

Performs the partitioned BFGS update onto the partitioned matrix `epm_B`, given the step `s` and the element gradient difference `epv_y`
"""
function PBFGS_update(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epm_copy = copy(epm_B)
	PBFGS_update!(epm_copy,epv_y,s; kwargs...)
	return epm_copy 
end

function PBFGS_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epv_s = epv_from_v(s, epv_y)
	PBFGS_update!(epm_B, epv_y, epv_s; kwargs...) 
	return epm_B
end

""" 
		PBFGS_update!(epm_B, epv_y, epv_s)

Performs the partitioned BFGS update onto the partitioned matrix `epm_B`, given the element steps `epv_s` and the element gradient differences `epv_y`.
"""
function PBFGS_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; verbose=true, kwargs...) where T 
	full_check_epv_epm(epm_B,epv_y) || @error("differents partitioned structures between epm_B and epv_y")
	full_check_epv_epm(epm_B,epv_s) || @error("differents partitioned structures between epm_B and epv_s")
	N = get_N(epm_B)
	for i in 1:N
		eemi = get_eem_set(epm_B, i)
		Bi = get_Bie(eemi)
		si = get_vec(get_eev(epv_s,i))
		yi = get_vec(get_eev(epv_y,i))
		index = get_index(eemi)
		update = BFGS!(si, yi, Bi, Bi; index=index, kwargs...) # return 0 or 1
		cem = get_cem(eemi)
		update_counter_elt_mat!(cem, update)
	end
	verbose && (str = string_counters_iter(epm_B))
	verbose && (print("\n PBFGS"*str))
	return epm_B
end

""" 
		B = PSR1_update(epm_B, epv_y, s)

Performs the partitioned SR1 update of the elemental partitioned matrix `epm_B`, given the step s and the element gradient difference epv_y
"""
function PSR1_update(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epm_copy = copy(epm_B)
	PSR1_update!(epm_copy,epv_y,s; kwargs...)
	return epm_copy 
end 

function PSR1_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epv_s = epv_from_v(s, epv_y)
	PSR1_update!(epm_B, epv_y, epv_s; kwargs...)
	return epm_B
end

""" 
		PSR1_update!(epm_B, epv_y, s)

Performs the partitioned-SR1 update of the elemental partitioned matrix `epm_B`, given the element steps `epv_s` and the element gradient differences `epv_y`.
"""
function PSR1_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; verbose=true, kwargs...) where T 
	full_check_epv_epm(epm_B,epv_y) || @error("differents partitioned structures between epm_B and epv_y")
	full_check_epv_epm(epm_B,epv_s) || @error("differents partitioned structures between epm_B and epv_s")
	N = get_N(epm_B)
	for i in 1:N
		eemi = get_eem_set(epm_B, i)
		Bi = get_Bie(eemi)
		si = get_vec(get_eev(epv_s,i))
		yi = get_vec(get_eev(epv_y,i))			
		index = get_index(eemi)
		update = SR1!(si, yi, Bi, Bi; index=index, kwargs...) # return 0 or 1
		cem = get_cem(eemi)
		update_counter_elt_mat!(cem, update)
	end 
	verbose && (str = string_counters_iter(epm_B))
	verbose && (print("\n PSR1"*str))
	return epm_B
end

""" 
		B = PSE_update(epm_B, epv_y, s)
		
Performs the partitioned update PSE of the elemental partitioned matrix `epm_B`, given the step s and the element gradient difference epv_y
"""
function PSE_update(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epm_copy = copy(epm_B)
	PSE_update!(epm_copy,epv_y,s; kwargs...)
	return epm_copy
end 
function PSE_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epv_s = epv_from_v(s, epv_y)
	PSE_update!(epm_B, epv_y, epv_s; kwargs...)
	return epm_B
end

""" 
		PSE_update!(epm_B, epv_y, epv_s)
		
Performs the partitioned update PSE of the elemental partitioned matrix `epm_B`, given the element steps `epv_s` and the element gradient differences `epv_y`.
"""
function PSE_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; verbose=true, kwargs...) where T 
	full_check_epv_epm(epm_B,epv_y) || @error("differents partitioned structures between epm_B and epv_y")
	full_check_epv_epm(epm_B,epv_s) || @error("differents partitioned structures between epm_B and epv_s")
	N = get_N(epm_B)
	acc = 0
	for i in 1:N
		eemi = get_eem_set(epm_B, i)
		Bi = get_Bie(eemi)
		si = get_vec(get_eev(epv_s,i))
		yi = get_vec(get_eev(epv_y,i))
		index = get_index(eemi)
		update = SE!(si, yi, Bi, Bi; index=index, kwargs...) # return 0 or 1
		cem = get_cem(eemi)
		update_counter_elt_mat!(cem, update)
	end 
	verbose && (str = string_counters_iter(epm_B))
	verbose && (print(str))
	return epm_B
end

end 