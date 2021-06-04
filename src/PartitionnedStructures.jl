module PartitionnedStructures

using LinearAlgebra



include("utils.jl")

include("ab_p_struct.jl")
include("P_vec/_include.jl")
include("P_mat/_include.jl")

include("link.jl")

include("frontale.jl")
include("1_parallel.jl")


end # module
