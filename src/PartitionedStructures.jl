module PartitionedStructures

using LinearAlgebra

include("acronyms.jl")

include("utils.jl")

# include files related to abstract partitioned/element structures.
include("ab_elt_struct.jl")
include("ab_ps_struct.jl")

# include files related to the partitioned-vector and the partitionned matrices
include("P_vec/_include.jl")
include("P_mat/_include.jl")

# include files related to:
# - partitioned qusi-Newton update
# - partitioned or element instances
# - how element/partitioned structures interact between themselves
include("methods/_include.jl")


include("partitioned_vectors.jl")


# use the submodule of PartitionedStructures.jl
using .Utils
using .M_abstract_part_struct, .M_abstract_element_struct
using .M_elt_vec, .M_elt_mat
using .M_part_v, .M_part_mat
using .ModElemental_ev, .ModElemental_em, .ModElemental_elo_bfgs, .ModElemental_elo_sr1
using .ModElemental_pv,
  .ModElemental_pm, .ModElemental_plo_bfgs, .ModElemental_plo_sr1, .ModElemental_plo
using .PartitionedQuasiNewton, .PartitionedLOQuasiNewton
using .Link, .Instances, .PartMatInterface

using ..PartitionedVectors
export PartitionedVector
# export the main methods of every submodule

# structures and functions related to element-structures
export Elemental_elt_vec, Elemental_em, Elemental_elo_bfgs, Elemental_elo_sr1
export create_eev, create_id_eem
export create_epv, get_eev, epv_from_v!, minus_epv!, add_epv!, epv_from_epv! # ModElemental_pv
export set_epv!
export get_v, build_v!
export get_eev_value

# structures and functions related to partitioned structures
export Part_mat
export Elemental_pv, Elemental_pm, Elemental_plo_bfgs
export identity_epm, identity_eplo_LBFGS, identity_eplo_LSR1, identity_eplo_LOSE
export initialize_component_list!

# Method linking the partitioned structures
export full_check_epv_epm
export epm_from_epv, eplo_lbfgs_from_epv, eplo_lose_from_epv, eplo_lsr1_from_epv
export create_epv_eplo, epv_from_eplo, epv_from_epm
export mul_epm_vector, mul_epm_vector!, mul_epm_epv, mul_epm_epv!
export Counter_elt_mat, string_counters_iter, string_counters_total
export prod_part_vectors

# partitioned quasi-Newton methods
export update, update!
export PBFGS_update!, PBFGS_update
export PSR1_update!, PSR1_update
export PSE_update, PSE_update!
export PCS_update, PCS_update!
export PLBFGS_update, PLBFGS_update!
export PLSR1_update, PLSR1_update!
export PLSE_update, PLSE_update!

# for the tutorial
export BFGS, SR1

end
