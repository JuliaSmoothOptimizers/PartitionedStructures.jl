module PartitionedLOQuasiNewton
using LinearOperators, LinearAlgebra

using ..M_abstract_part_struct, ..M_elt_vec, ..M_part_mat, ..M_elt_mat
using ..M_abstract_element_struct
using ..Utils, ..Link
using ..ModElemental_ev, ..ModElemental_pv
using ..ModElemental_elom_bfgs, ..ModElemental_elom_sr1, ..ModElemental_plom
using ..ModElemental_plom_bfgs, ..ModElemental_plom_sr1

export PLBFGS_update, PLBFGS_update!
export PLSR1_update, PLSR1_update!
export PLSE_update, PLSE_update!
export Part_update, Part_update!

""" 
    B = PLBFGS_update(eplom_B, s, epv_y)
    
Performs the partitioned-LBFGS update onto the limited memory partitioned operator `eplom_B`, given the step `s` and the element gradient differences `epv_y`.
"""
function PLBFGS_update(eplom_B :: Elemental_plom_bfgs{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epm_copy = copy(eplom_B)
	PLBFGS_update!(epm_copy,epv_y,s; kwargs...)
	return epm_copy 
end 

function PLBFGS_update!(eplom_B :: Elemental_plom_bfgs{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epv_s = epv_from_v(s, epv_y)
	PLBFGS_update!(eplom_B, epv_y, epv_s; kwargs...)
	return eplom_B
end

""" 
    PLBFGS_update!(eplom_B, s, epv_y)
    
Performs the partitioned-LBFGS update onto the limited memory partitioned operator `eplom_B`, given the element steps `epv_s` and the element gradient differences `epv_y`.
"""
function PLBFGS_update!(eplom_B :: Elemental_plom_bfgs{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; verbose=true, reset=true, kwargs...) where T 
  full_check_epv_epm(eplom_B,epv_y) || @error("differents partitioned structures between eplom_B and epv_y")
  full_check_epv_epm(eplom_B,epv_s) || @error("differents partitioned structures between eplom_B and epv_s")
  N = get_N(eplom_B)
  for i in 1:N      
    eelomi = get_eelom_set(eplom_B, i)
    si = get_vec(get_eev(epv_s,i))
    yi = get_vec(get_eev(epv_y,i))
    index = get_index(eelomi)
    if (dot(si,yi) > eps(T))
      Bi = get_Bie(eelomi)
      push!(Bi, si, yi)		
      update = 1	
    elseif index < reset # Bi is not updated nor reset				
      update = 0
    else 
      reset_eelom_bfgs!(eelomi)
      update = -1
    end 
    cem = get_cem(eelomi)
    update_counter_elt_mat!(cem, update)
  end 
  verbose && (str = string_counters_iter(eplom_B))
  verbose && (print("\n PLBFGS"*str))
  return eplom_B
end

""" 
    B = PLSR1_update(eplom_B, s, epv_y)

Performs the partitioned-SR1 update onto the limited memory partitioned operator `eplom_B`, given the step `s` and the element gradient differences `epv_y`.
"""
function PLSR1_update(eplom_B :: Elemental_plom_sr1{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epm_copy = copy(eplom_B)
	PLSR1_update!(epm_copy,epv_y,s; kwargs...)
	return epm_copy
end 

function PLSR1_update!(eplom_B :: Elemental_plom_sr1{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T
	epv_s = epv_from_v(s, epv_y)
	PLSR1_update!(eplom_B, epv_y, epv_s; kwargs...)
	return eplom_B
end

""" 
    PLSR1_update!(eplom_B, s, epv_y)

Performs the partitioned-SR1 update onto the limited memory partitioned operator `eplom_B`, given the element steps `epv_s` and the element gradient differences `epv_y`.
"""
function PLSR1_update!(eplom_B :: Elemental_plom_sr1{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; ω = 1e-6, verbose=true, reset=4, kwargs...) where T 
  full_check_epv_epm(eplom_B,epv_y) || @error("differents partitioned structures between eplom_B and epv_y")
  full_check_epv_epm(eplom_B,epv_s) || @error("differents partitioned structures between eplom_B and epv_s")
  N = get_N(eplom_B)
  for i in 1:N      
    eelomi = get_eelom_set(eplom_B, i)
    si = get_vec(get_eev(epv_s,i))
    yi = get_vec(get_eev(epv_y,i))
    Bi = get_Bie(eelomi)
    ri = yi .- Bi*si
    index = get_index(eelomi)
    if abs(dot(si,ri)) > ω * norm(si,2) * norm(ri,2)
      push!(Bi, si, yi)
      update = 1
    elseif index < reset # Bi is not updated nor reset				
      update = 0
    else 
      reset_eelom_sr1!(eelomi)
      update = -1
    end
    cem = get_cem(eelomi)
    update_counter_elt_mat!(cem, update)
  end 
  verbose && (str = string_counters_iter(eplom_B))
  verbose && (print("\n PLSR1"*str))
  return eplom_B
end

"""
    B = Part_update(eplom_B, epv_y, s)

Performs a quasi-Newton partitionned update of `eplom_B`, given the step `s` and the element gradient differences `epv_y`.
Each element linear operator from `eplom_B` is either a `LBFGSOperator` or `LSR1Operator`.
The update performs on each element the quasi-Newton to the linear operator associated.
"""
function Part_update(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}) where Y <: Part_LO_mat{T} where T
	epm_copy = copy(eplom_B)
	Part_update!(epm_copy,epv_y,s)
	return epm_copy 
end 

function Part_update!(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}) where Y <: Part_LO_mat{T} where T
	epv_s = epv_from_v(s, epv_y)
	Part_update!(eplom_B, epv_y, epv_s)
	return eplom_B
end

"""
    Part_update!(eplom_B, epv_y, s)

Performs a quasi-Newton partitionned update of `eplom_B`, given the element steps `epv_s` and the element gradient differences `epv_y`.
Each element linear operator from `eplom_B` is either a `LBFGSOperator` or `LSR1Operator`.
The update performs on each element the quasi-Newton to the linear operator associated.
"""
function Part_update!(eplom_B :: Y, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; kwargs...) where Y <: Part_LO_mat{T} where T 
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
    B = PLSE_update(eplom_B, epv_y, s)

Performs the partitionned update PLSE of `eplom_B`, which is made of LBFGS or LSR1 elemental element matrices, given the step `s` and the element gradient differences `epv_y`.
The update try to apply a LBFGS update to every Bᵢ, but if the curvature condition yᵢᵀs > 0 is not satisfied it applies LSR1 in place of LBFGS.
"""
function PLSE_update(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where Y <: Part_LO_mat{T} where T
	epm_copy = copy(eplom_B)
	PLSE_update!(epm_copy,epv_y,s; kwargs...)
	return epm_copy
end 

function PLSE_update!(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where Y <: Part_LO_mat{T} where T
	epv_s = epv_from_v(s, epv_y)
	PLSE_update!(eplom_B, epv_y, epv_s; kwargs...)
end

"""
    PLSE_update!(eplom_B, epv_y, epv_s)

Performs the partitionned update PLSE of `eplom_B`, which is made of LBFGS or LSR1 elemental element matrices, given the element steps `epv_s` and the element gradient differences `epv_y`.
The update try to apply a LBFGS update to every Bᵢ, but if the curvature condition yᵢᵀs > 0 is not satisfied it applies LSR1 in place of LBFGS.
"""
function PLSE_update!(eplom_B :: Y, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; ω = 1e-6, verbose=true, reset=4, kwargs...) where Y <: Part_LO_mat{T} where T 
  full_check_epv_epm(eplom_B,epv_y) || @error("differents partitioned structures between eplom_B and epv_y")
  full_check_epv_epm(eplom_B,epv_s) || @error("differents partitioned structures between eplom_B and epv_s")
  N = get_N(eplom_B)
  acc_lbfgs = 0
  acc_lsr1 = 0
  acc_untouched = 0
  acc_reset = 0
  for i in 1:N
    eelom = get_eelom_set(eplom_B, i)
    Bi = get_Bie(eelom)
    si = get_vec(get_eev(epv_s,i))
    yi = get_vec(get_eev(epv_y,i))			
    ri = yi .- Bi*si
    index = get_index(eelom)
    if isa(Bi, LBFGSOperator{T})
      if dot(si, yi) > eps(T)  # curvature condition
        push!(Bi, si, yi)			
        acc_lbfgs += 1 
      elseif abs(dot(si,ri)) > ω * norm(si,2) * norm(ri,2) # numerical condition of LSR1
        indices = get_indices(eelom)
        eelom = init_eelom_LSR1(indices; T=T)
        Bi = get_Bie(eelom)
        set_eelom_set!(eplom_B, i, eelom)
        push!(Bi, si, yi)	
        acc_lsr1 += 1
      elseif index < reset
        acc_untouched += 1
      else 
        reset_eelom_bfgs!(eelom)
        acc_reset += 1
      end
    else # isa(Bi, LSR1Operator{T})				
      if abs(dot(si,ri)) > ω * norm(si,2) * norm(ri,2)
        push!(Bi, si, yi)			
        acc_lsr1 += 1	
      elseif index < reset
        acc_untouched += 1
      else
        indices = get_indices(eelom)
        eelom = init_eelom_LBFGS(indices; T=T)
        acc_reset += 1
      end 
    end
  end 
  verbose && println("PLSE : LBFGS updates $(acc_lbfgs)/$(N), LSR1 $(acc_lsr1)/$(N), untouched $(acc_untouched)/$(N), reset $(acc_reset)/$(N)")
  return eplom_B
end

end 