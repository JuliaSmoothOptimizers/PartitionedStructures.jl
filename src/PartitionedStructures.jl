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

include("link.jl")

#include related to factorization of partitionned matrices
include("factorization/_include.jl")


end # module
