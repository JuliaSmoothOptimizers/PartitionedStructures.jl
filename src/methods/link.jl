module Link
  using LinearAlgebra, SparseArrays

	using ..M_elt_mat
  using ..M_part_mat, ..M_part_v
  using ..ModElemental_em, ..ModElemental_ev
  using ..ModElemental_pv, ..ModElemental_plom_bfgs, ..ModElemental_plom_sr1, ..ModElemental_plom, ..ModElemental_pm  
  using ..M_abstract_element_struct, ..M_abstract_part_struct

  export eplom_lbfgs_from_epv, eplom_lsr1_from_epv, eplom_lose_from_epv, epm_from_epv
	export epv_from_eplom, epv_from_epm
  export mul_epm_epv, mul_epm_epv!, mul_epm_vector, mul_epm_vector!
	export string_counters_iter, string_counters_total
	
  @inline epv_from_eplom(eplom) = epv_from_epm(eplom)
  """
      epv_from_epm(epm)
  Create an elemental partitioned vector with the same partitioned structure than `epm`.
  """
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
  
  """
      epm_from_epv(epm)
  Create an elemental partitioned matrix with the same partitioned structure than `epv`.
  Each elemental element matrix is set with an identity matrix.
  """
  function epm_from_epv(epv :: T) where T <: Elemental_pv{Y} where Y <: Number
		N = get_N(epv)
    n = get_n(epv)
    eelom_indices_set = Vector{Vector{Int}}(undef,N)
    for i in 1:N
      eesi = get_ee_struct(epv,i)
      indices = get_indices(eesi)
      eelom_indices_set[i] = indices    
    end 
    epm = identity_epm(eelom_indices_set, N, n; T=Y)
    return epm
  end

  """
      eplom_lbfgs_from_epv(epm)
  Create an elemental partitioned linear operator matrix with the same partitioned structure than `epv`.
  Each elemental element linear operator is set with an LBFGS operator.
  """
  function eplom_lbfgs_from_epv(epv :: T) where T <: Elemental_pv{Y} where Y <: Number
    N = get_N(epv)
    n = get_n(epv)
    eelom_indices_set = Vector{Vector{Int}}(undef,N)
    for i in 1:N
      eesi = get_ee_struct(epv,i)
      indices = get_indices(eesi)
      eelom_indices_set[i] = indices    
    end 
    eplom = identity_eplom_LBFGS(eelom_indices_set, N, n; T=Y)
    return eplom
  end

  """
			eplom_lsr1_from_epv(epm)
  Create an elemental partitioned linear operator matrix with the same partitioned structure than `epv`.
  Each elemental element linear operator is set with an LSR1 operator.
  """
  function eplom_lsr1_from_epv(epv :: T) where T <: Elemental_pv{Y} where Y <: Number
    N = get_N(epv)
    n = get_n(epv)
    eelom_indices_set = Vector{Vector{Int}}(undef,N)
    for i in 1:N
      eesi = get_ee_struct(epv,i)
      indices = get_indices(eesi)
      eelom_indices_set[i] = indices    
    end 
    eplom = identity_eplom_LSR1(eelom_indices_set, N, n; T=Y)
    return eplom
  end
	
	"""
			eplom_lose_from_epv(epm)
  Create an elemental partitioned linear operator matrix with the same partitioned structure than `epv`.
  Each elemental element linear operator is set with an LSR1 operator.
  """
  function eplom_lose_from_epv(epv :: Elemental_pv{T}) where T <: Number
    N = get_N(epv)
    n = get_n(epv)
    eelom_indices_set = Vector{Vector{Int}}(undef,N)
    for i in 1:N
      eesi = get_ee_struct(epv,i)
      indices = get_indices(eesi)
      eelom_indices_set[i] = indices    
    end 
    eplom = identity_eplom_LOSE(eelom_indices_set, N, n; T=T)
    return eplom
  end

  """
      mul_epm_vector(epm, x)
  Compute the product between the elemental partitioned matrix `epm` and the vector `x`.
  """
  function mul_epm_vector(epm :: T, x :: Vector{Y}) where T <: Part_mat{Y} where Y <: Number
    epv = epv_from_epm(epm)
    mul_epm_vector(epm, epv, x)
  end 
  function mul_epm_vector(epm :: T, epv :: Elemental_pv{Y}, x :: Vector{Y}) where T <: Part_mat{Y} where Y <: Number
    res = similar(x)
    mul_epm_vector!(res, epm, epv, x)
    return res
  end

  """
      mul_epm_vector!(res, epm, x)
  Compute the product between the elemental partitioned matrix `epm` and the vector `x`.
  The result is stored in the vector `res`.
  """	
  function mul_epm_vector!(res :: Vector{Y}, epm :: T, x :: Vector{Y}) where T <: Part_mat{Y} where Y <: Number
    epv = epv_from_epm(epm)
    mul_epm_vector!(res,epm,epv,x)
  end

  """
      mul_epm_vector!(res, epm, epv, x)
  Compute the product between the elemental partitioned matrix `epm` and the vector `x`.
  The result is stored in the vector `res`.
  """	
  function mul_epm_vector!(res :: Vector{Y}, epm :: T, epv :: Elemental_pv{Y}, x :: Vector{Y}) where T <: Part_mat{Y} where Y <: Number
    epv_from_v!(epv,x)
    mul_epm_epv!(epv,epm,epv)
    build_v!(epv)
    res .= get_v(epv)
  end 

  """
      mul_epm_epv(epm, epv)
  Compute the product between the elemental partitioned matrix `epm` and the elemental partitioned vecto `epv`.	
  """		
  function mul_epm_epv(epm :: T, epv :: Elemental_pv{Y}) where T <: Part_mat{Y} where Y <: Number
    epv_res = similar(epv)
    mul_epm_epv!(epv_res, epm, epv)
    return epv_res
  end

  """
      mul_epm_epv!(epv_res, epm, epv)
  Compute the product between the elemental partitioned matrix `epm` and the elemental partitioned vecto `epv`.	
  The result is stored in the elemental partitioned vector `epv_res`.
  """		
  function mul_epm_epv!(epv_res :: Elemental_pv{Y}, epm :: T, epv :: Elemental_pv{Y}) where T <: Part_mat{Y} where Y <: Number
    full_check_epv_epm(epm,epv) || error("Structure differ epm/epv")
    N = get_N(epm)
    for i in 1:N
      Bie = get_ee_struct_Bie(epm,i)
      vie = get_eev_value(epv, i)
      set_eev!(epv_res, i,Bie*vie)
    end
  end 

	function string_counters_iter(pm :: T) where T <: Part_mat
		epm_vectors = get_ee_struct(pm)
		counters = (epm -> epm.counter).(epm_vectors)
		update = 0
		untouch = 0
		reset = 0
		for counter in counters
			(up, un, re) = iter_info(counter)
			update += up
			untouch += un
			reset += re
		end 
		println("Partitioned update ", T, " update: ", update, ", untouch: ", untouch, ", reset: ", reset)
	end

	function string_counters_total(pm :: T) where T <: Part_mat
		epm_vectors = get_ee_struct(pm)
		counters = (epm -> epm.counter).(epm_vectors)
		update = 0
		untouch = 0
		reset = 0
		for counter in counters
			(up, un, re) =  total_info(counter)
			update += up
			untouch += un
			reset += re
		end 
		println("Partitioned update ", T, " update: ", update, ", untouch: ", untouch, ", reset: ", reset)
	end


end 