using PartitionnedStructures
using Test
using BenchmarkTools, ProfileView

# include("test/runtests.jl")
last = true
not_last = true  

not_last && include("P_vec/_include.jl")
not_last && include("P_mat/_include.jl")
last && include("frontale.jl")
