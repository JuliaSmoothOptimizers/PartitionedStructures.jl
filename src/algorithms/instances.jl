module Instances

	using ..M_part_mat, ..M_part_v
  using ..ModElemental_pv, ..ModElemental_plom_bfgs, ..ModElemental_plom, ..ModElemental_pm
  using ..ModElemental_ev
  using ..M_abstract_element_struct, ..M_abstract_part_struct

	export create_epv_eplom_bfgs, create_epv_eplom, create_epv_epm, create_epv_epm_rand

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