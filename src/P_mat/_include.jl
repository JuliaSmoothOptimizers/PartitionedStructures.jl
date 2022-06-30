# Define the abstract type of element-matrix
include("elt_mat.jl")
# Define the element-matrices using the elemental variable
include("elemental_em.jl")
# Define the abstract type of partitionned matrix
include("part_mat.jl")
# Define the partitionned matrices using the elemental element-matrices
include("elemental_pm.jl")

include("elemental_elo_BFGS.jl")
include("elemental_elo_SR1.jl")

include("elemental_plo_BFGS.jl")
include("elemental_plo_SR1.jl")
include("elemental_plo.jl")