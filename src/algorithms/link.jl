module Link
  using ..M_part_mat, ..M_part_v
  using ..ModElemental_pv, ..ModElemental_plom_bfgs, ..ModElemental_plom, ..ModElemental_pm
  using ..ModElemental_ev
  using ..M_abstract_element_struct, ..M_abstract_part_struct

  export epv_from_epm
  export mul_epm_epv, mul_epm_epv!, mul_epm_vector, mul_epm_vector!
  export create_epv_eplom_bfgs, create_epv_eplom, create_epv_epm, create_epv_epm_rand
    
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

  """
      create_epv_epm(;n=n,nie=nie,overlpapping=overlapping, mul_m=mul_m, mul_v=mul_v)
  Create an elemental partitioned vector and a elemental partitioned matrix with the same partitioned structure defined by `n,nie,overlapping,mul_l,mul_v`.
  """
  function create_epv_epm(;n=9,nie=5,overlapping=1,mul_m=5., mul_v=100.)
    epm = part_mat(;n=n,nie=nie,overlapping=overlapping,mul=mul_m)
    epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
    return (epm,epv)
  end 

  """
      create_epv_epm_rand(;n=n,nie=nie,overlpapping=overlapping, mul_m=mul_m, mul_v=mul_v)
  Create a random elemental partitioned vector and a random elemental partitioned matrix with the same partitioned structure defined by `n,nie,overlapping,mul_l,mul_v`.
  """
  function create_epv_epm_rand(;n=9,nie=5,overlapping=1,range_mul_m=nie:2*nie, mul_v=100.)
    epm = part_mat(;n=n,nie=nie,overlapping=overlapping,mul=rand(range_mul_m))
    epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
    return (epm,epv)
  end 

  """
      create_epv_eplom_bfgs(;n=n,nie=nie,overlpapping=overlapping, mul_m=mul_m, mul_v=mul_v)
  Create a elemental partitioned vector and a random elemental partitioned matrix using linear operators LBFGS with the same partitioned structure defined by `n,nie,overlapping,mul_l,mul_v`.
  """
  function create_epv_eplom_bfgs(;n=9,nie=5,overlapping=1,range_mul_m=nie:2*nie, mul_v=100.)
    eplom = PLBFGS_eplom(;n=n,nie=nie,overlapping=overlapping)
    epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
    return (eplom,epv)
  end 

  """
      create_epv_epm_rand(;n=n,nie=nie,overlpapping=overlapping, mul_m=mul_m, mul_v=mul_v)
  Create a elemental partitioned vector and a random elemental partitioned matrix using linear operators LBFGS/LSR1 with the same partitioned structure defined by `n,nie,overlapping,mul_l,mul_v`.
  """
  function create_epv_eplom(;n=9,nie=5,overlapping=1,range_mul_m=nie:2*nie, mul_v=100.)
    eplom = PLBFGSR1_eplom(;n=n,nie=nie,overlapping=overlapping)
    epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
    return (eplom,epv)
  end

end 