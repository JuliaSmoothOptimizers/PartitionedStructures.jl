module PartitionedStructures

  using LinearAlgebra
  
  include("utils.jl")

  #include related to structure definition
  # Define the abstract element structure and abstract partitionned structure
  include("ab_elt_struct.jl")
  include("ab_ps_struct.jl")

  # Define the partitionned vector and the partitionned matrices
  include("P_vec/_include.jl")
  include("P_mat/_include.jl")

  #include related to factorization of partitionned matrices
	include("methods/_include.jl")
  include("factorizations/_include.jl")

  using .M_abstract_part_struct, .M_abstract_element_struct
  using .M_elt_vec, .M_elt_mat
  using .M_part_v, .M_part_mat
  using .ModElemental_ev, .ModElemental_em, .ModElemental_elom_bfgs
  using .ModElemental_pv, .ModElemental_pm, .ModElemental_plom_bfgs, .ModElemental_plom_sr1
  using .PartitionedQuasiNewton, .PartitionedLOQuasiNewton
  using .Link, .Instances, .PartMatInterface

  export Part_mat
  export Elemental_pv, Elemental_pm, Elemental_plom_bfgs
  export Elemental_elt_vec, Elemental_em, Elemental_elom_bfgs
  export create_eev, create_id_eem
  export identity_epm, identity_eplom_LBFGS, identity_eplom_LSR1
  export create_epv, get_eev, epv_from_v!, minus_epv!, add_epv!, epv_from_epv! # ModElemental_pv
  export get_v, build_v!
  export get_eev_value
  export full_check_epv_epm
  export PBFGS_update!, PBFGS_update, PLBFGS_update, PLBFGS_update!, PLSR1_update, PLSR1_update!
	export update, update!
  export epm_from_epv, eplom_lbfgs_from_epv, create_epv_eplom, epv_from_eplom, epv_from_epm
	export mul_epm_vector, mul_epm_vector!, mul_epm_epv
  export initialize_component_list!

end # module
