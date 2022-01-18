# Define the abstract type of element matrix
include("elt_mat.jl")
# Define the element matrices using the elemental variable
include("elemental_em.jl")
# Define the abstract type of partitionned matrix
include("part_mat.jl")
# Define the partitionned matrices using the elemental element matrices
include("elemental_pm.jl")

include("elemental_elm.jl")
# include("elemental_plm.jl")
