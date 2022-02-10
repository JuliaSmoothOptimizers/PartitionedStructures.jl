using LinearAlgebra: Matrix
using PartitionedStructures
using Test

last = true
not_last = true  

not_last && include("P_vec/_include.jl")
not_last && include("P_mat/_include.jl")
last && include("others/_include.jl")

last && include("factorization/_include.jl")
