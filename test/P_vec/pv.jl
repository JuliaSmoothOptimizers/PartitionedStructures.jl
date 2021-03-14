using PartitionnedStructures
using PartitionnedStructures.M_part_v
using PartitionnedStructures.M_elemental_pv
using PartitionnedStructures.M_internal_pv

using BenchmarkTools, ProfileView, Test, LinearAlgebra


N = 100
k = 4
pev_1 = ones_kchained_epv(N,k)
build_v!(pev_1)
v = get_v(pev_1)
@test sum(v) == N*k
# @code_warntype ones_kchained_epv(N,k)
# b_pev_1 = @benchmark build_v!(pev_1)
# @code_warntype build_v!(pev_1)


N = 30
nᵢ = 50
pev = rand_epv(N,nᵢ)
v1 = build_v!(pev)
v2 = build_v!(pev)
@test v1 == v2

piv = rand_ipv(N,nᵢ)


# b = @benchmark build_v!(pev)
# ProfileView.@profview (@benchmark build_v!(pev))
# @code_warntype build_v!(pev)

