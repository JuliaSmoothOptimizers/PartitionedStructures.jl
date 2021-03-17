using PartitionnedStructures
using Test

# include("test/runtests.jl")
last = true
not_last = false  

not_last && include("P_vec/_include.jl")
last && include("P_mat/_include.jl")