# Define the abstract type of element matrix
include("elt_mat.jl")
# Define the element matrices using the elemental variable
include("elemental_em.jl")
# Define the abstract type of partitionned matrix
include("part_mat.jl")
# Define the partitionned matrices using the elemental element matrices
include("elemental_pm.jl")

include("elemental_elom_BFGS.jl")
include("elemental_elom_SR1.jl")

include("elemental_plom_BFGS.jl")
include("elemental_plom_SR1.jl")
include("elemental_plom.jl")
