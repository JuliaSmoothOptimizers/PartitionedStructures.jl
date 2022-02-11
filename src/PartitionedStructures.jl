module PartitionedStructures

  using LinearAlgebra
  
  include("utils.jl")

  #include related to structure definition
  # Define the abstract element structure and abstract partitionned structure
  # include("ab_struct.jl")
  include("ab_elt_struct.jl")
  include("ab_ps_struct.jl")

  # Define the partitionned vector and the partitionned matrices
  include("P_vec/_include.jl")
  include("P_mat/_include.jl")

  #include related to factorization of partitionned matrices
  include("algorithms/_include.jl")

	using .M_abstract_part_struct
  using .ModElemental_pv, .ModElemental_ev, .ModElemental_em, .ModElemental_pm
  using .M_elt_vec, .M_part_v
	using .PartitionedQuasiNewton
	using .Link

  export Elemental_pv, Elemental_pm
  export create_epv, get_eev, epv_from_v!, minus_epv!, add_epv! # ModElemental_pv
  export get_v, build_v!
  export get_eev_value
	export full_check_epv_epm
	export PBFGS_update!, PBFGS_update
	export epv_from_epm, mul_epm_vector, mul_epm_vector!, mul_epm_epv

end # module
