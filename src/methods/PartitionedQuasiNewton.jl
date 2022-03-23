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
      PBFGS_update(epm_B, s, epv_y)
  Define the partitioned BFGS update of the partioned matrix epm_B, given the step s and the element gradient difference epv_y
  """
  PBFGS_update(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T = begin epm_copy = copy(epm_B); PBFGS_update!(epm_copy,epv_y,s; kwargs...); return epm_copy end 
  PBFGS_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T = begin epv_s = epv_from_v(s, epv_y); PBFGS_update!(epm_B, epv_y, epv_s; kwargs...) end
  function PBFGS_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; kwargs...) where T 
    full_check_epv_epm(epm_B,epv_y) || @error("differents partitioned structures between epm_B and epv_y")
    full_check_epv_epm(epm_B,epv_s) || @error("differents partitioned structures between epm_B and epv_s")
    N = get_N(epm_B)
    acc = 0
    for i in 1:N
      Bi = get_Bie(get_eem_set(epm_B, i))
      si = get_vec(get_eev(epv_s,i))
      yi = get_vec(get_eev(epv_y,i))
      acc += BFGS!(si, yi, Bi, Bi; kwargs...)
    end 
    println("PBFGS, update $(acc)/$(N) elements")
  end

  """ 
      PSR1_update(epm_B, s, epv_y)
  Define the partitioned SR1 update of the elemental partioned matrix epm_B, given the step s and the element gradient difference epv_y
  """
  PSR1_update(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}) where T = begin epm_copy = copy(epm_B); PSR1_update!(epm_copy,epv_y,s); return epm_copy end 
  PSR1_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}) where T = begin epv_s = epv_from_v(s, epv_y); PSR1_update!(epm_B, epv_y, epv_s) end
  function PSR1_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}) where T 
    full_check_epv_epm(epm_B,epv_y) || @error("differents partitioned structures between epm_B and epv_y")
    full_check_epv_epm(epm_B,epv_s) || @error("differents partitioned structures between epm_B and epv_s")
    N = get_N(epm_B)
		acc = 0
    for i in 1:N
      Bi = get_Bie(get_eem_set(epm_B, i))
      si = get_vec(get_eev(epv_s,i))
      yi = get_vec(get_eev(epv_y,i))
      acc += SR1!(si, yi, Bi, Bi)
    end 
		println("PSR1, update $(acc)/$(N) elements")
  end

  """ 
      PSE_update(epm_B, s, epv_y)
  Define the partitioned SR1 update of the elemental partioned matrix epm_B, given the step s and the element gradient difference epv_y
  """
  PSE_update(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}) where T = begin epm_copy = copy(epm_B); PSE_update!(epm_copy,epv_y,s); return epm_copy end 
  PSE_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}) where T = begin epv_s = epv_from_v(s, epv_y); PSE_update!(epm_B, epv_y, epv_s) end
  function PSE_update!(epm_B :: Elemental_pm{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}) where T 
    full_check_epv_epm(epm_B,epv_y) || @error("differents partitioned structures between epm_B and epv_y")
    full_check_epv_epm(epm_B,epv_s) || @error("differents partitioned structures between epm_B and epv_s")
    N = get_N(epm_B)
		acc = 0
    for i in 1:N
      Bi = get_Bie(get_eem_set(epm_B, i))
      si = get_vec(get_eev(epv_s,i))
      yi = get_vec(get_eev(epv_y,i))
      acc += SE!(si, yi, Bi, Bi)
    end 
		println("PSE, update $(acc)/$(N) elements")
  end

end 