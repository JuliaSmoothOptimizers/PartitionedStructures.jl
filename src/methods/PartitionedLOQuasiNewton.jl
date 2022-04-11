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
      PLBFGS_update(eplom_B, s, epv_y)
  Define the partitioned LBFGS update of the partioned matrix eplom_B, given the step s and the element gradient difference epv_y
  """
  PLBFGS_update(eplom_B :: Elemental_plom_bfgs{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T = begin epm_copy = copy(eplom_B); PLBFGS_update!(epm_copy,epv_y,s; kwargs...); return epm_copy end 
  PLBFGS_update!(eplom_B :: Elemental_plom_bfgs{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T = begin epv_s = epv_from_v(s, epv_y); PLBFGS_update!(eplom_B, epv_y, epv_s; kwargs...) end
  function PLBFGS_update!(eplom_B :: Elemental_plom_bfgs{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; verbose=true, kwargs...) where T 
    full_check_epv_epm(eplom_B,epv_y) || @error("differents partitioned structures between eplom_B and epv_y")
    full_check_epv_epm(eplom_B,epv_s) || @error("differents partitioned structures between eplom_B and epv_s")
    N = get_N(eplom_B)
    for i in 1:N      
			eelomi = get_eelom_set(eplom_B, i)
      si = get_vec(get_eev(epv_s,i))
      yi = get_vec(get_eev(epv_y,i))			
			if (dot(si,yi) > eps(T))
				Bi = get_Bie(eelomi)
      	push!(Bi, si, yi)		
				update = 1	
			else 
				reset_eelom_bfgs!(eelomi)
				update = -1
			end 
			cem = get_cem(eelomi)
			update_counter_elt_mat!(cem, update)
    end 
		str = string_counters_iter(eplom_B)
		println(str)
		return eplom_B
	end

	""" 
      PLSR1_update(eplom_B, s, epv_y)
  Define the partitioned LSR1 update of the partioned matrix eplom_B, given the step s and the element gradient difference epv_y
  """
  PLSR1_update(eplom_B :: Elemental_plom_sr1{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T = begin epm_copy = copy(eplom_B); PLSR1_update!(epm_copy,epv_y,s; kwargs...); return epm_copy end 
  PLSR1_update!(eplom_B :: Elemental_plom_sr1{T}, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where T = begin epv_s = epv_from_v(s, epv_y); PLSR1_update!(eplom_B, epv_y, epv_s; kwargs...) end
  function PLSR1_update!(eplom_B :: Elemental_plom_sr1{T}, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; ω = 1e-6, verbose=true, kwargs...) where T 
    full_check_epv_epm(eplom_B,epv_y) || @error("differents partitioned structures between eplom_B and epv_y")
    full_check_epv_epm(eplom_B,epv_s) || @error("differents partitioned structures between eplom_B and epv_s")
    N = get_N(eplom_B)
    for i in 1:N      
			eelomi = get_eelom_set(eplom_B, i)
      si = get_vec(get_eev(epv_s,i))
      yi = get_vec(get_eev(epv_y,i))
			Bi = get_Bie(eelomi)
			ri = yi .- Bi*si
    	if abs(dot(si,ri)) > ω * norm(si,2) * norm(ri,2)
      	push!(Bi, si, yi)
				update = 1
			else 
				reset_eelom_sr1!(eelomi)
				update = -1
			end
			cem = get_cem(eelomi)
			update_counter_elt_mat!(cem, update)
    end 
		verbose && (str = string_counters_iter(eplom_B))
		verbose && (println(str))
		return eplom_B
  end

  """
      Part_update(eplom_B, epv_y, s)
  Perform the partitionned update of eplom_B.
  eplom_B is build from LBFGS or LSR1 elemental element matrices.
  The update performed on eachh element matrix correspond to the linear operator associated.
  """
  Part_update(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}) where Y <: Part_LO_mat{T} where T = begin epm_copy = copy(eplom_B); Part_update!(epm_copy,epv_y,s); return epm_copy end 
  Part_update!(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}) where Y <: Part_LO_mat{T} where T = begin epv_s = epv_from_v(s, epv_y); Part_update!(eplom_B, epv_y, epv_s) end
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
      PLSE_update(eplom_B, epv_y, s)
  Perform the partitionned update of eplom_B.
  eplom_B is build from LBFGS or LSR1 elemental element matrices.
  The update performed on eachh element matrix correspond to the linear operator associated.
  """
  PLSE_update(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where Y <: Part_LO_mat{T} where T = begin epm_copy = copy(eplom_B); PLSE_update!(epm_copy,epv_y,s; kwargs...); return epm_copy end 
  PLSE_update!(eplom_B :: Y, epv_y :: Elemental_pv{T}, s :: Vector{T}; kwargs...) where Y <: Part_LO_mat{T} where T = begin epv_s = epv_from_v(s, epv_y); PLSE_update!(eplom_B, epv_y, epv_s; kwargs...) end
  function PLSE_update!(eplom_B :: Y, epv_y :: Elemental_pv{T}, epv_s :: Elemental_pv{T}; ω = 1e-6, verbose=true, kwargs...) where Y <: Part_LO_mat{T} where T 
    full_check_epv_epm(eplom_B,epv_y) || @error("differents partitioned structures between eplom_B and epv_y")
    full_check_epv_epm(eplom_B,epv_s) || @error("differents partitioned structures between eplom_B and epv_s")
    N = get_N(eplom_B)
		acc_lbfgs = 0
		acc_lsr1 = 0
		acc_reset = 0
    for i in 1:N
			eelom = get_eelom_set(eplom_B, i)
      Bi = get_Bie(eelom)
      si = get_vec(get_eev(epv_s,i))
      yi = get_vec(get_eev(epv_y,i))			
			ri = yi .- Bi*si
			if isa(Bi, LBFGSOperator{T})
				if dot(si, yi) > eps(T)  # curvature condition
					push!(Bi, si, yi)			
					acc_lbfgs += 1 
				elseif abs(dot(si,ri)) > ω * norm(si,2) * norm(ri,2)
					indices = get_indices(eelom)
					eelom = init_eelom_LSR1(indices; T=T)
					Bi = get_Bie(eelom)
					set_eelom_set!(eplom_B, i, eelom)
					push!(Bi, si, yi)	
					acc_lsr1 += 1
				else 
					reset_eelom_bfgs!(eelom)
					acc_reset += 1
				end
			else # isa(Bi, LSR1Operator{T})				
				if abs(dot(si,ri)) > ω * norm(si,2) * norm(ri,2)
					push!(Bi, si, yi)			
					acc_lsr1 += 1	
				else
					indices = get_indices(eelom)
					eelom = init_eelom_LBFGS(indices; T=T)
					acc_reset += 1
				end 
			end
		end 
		verbose && println("LBFGS updates $(acc_lbfgs)/$(N), LSR1 $(acc_lsr1)/$(N) ")
		return eplom_B
  end

end 