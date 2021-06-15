module PartitionnedStructures

using LinearAlgebra



include("utils.jl")

#include related to structure definition
include("ab_struct.jl")
include("P_vec/_include.jl")
include("P_mat/_include.jl")

include("link.jl")

#include related to factorization
include("factorization/_include.jl")


end # module
