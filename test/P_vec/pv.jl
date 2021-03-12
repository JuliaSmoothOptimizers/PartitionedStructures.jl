using PartitionnedStructures
using PartitionnedStructures.M_part_v
using PartitionnedStructures.M_elemental_pv
using PartitionnedStructures.M_internal_pv


pev = new_elemental_pv(30,50)
v1 = build_v!(pev)
v2 = build_v!(pev)
@test v1 == v2

piv = new_internal_pv(30,50)



using BenchmarkTools, ProfileView

b = @benchmark build_v!(pev)
ProfileView.@profview (@benchmark build_v!(pev))
@code_warntype build_v!(pev)

# function f_view(a, _n)
# 	component_view = rand(1:n,_n)
# 	_view = view(a, component_view) 
# 	_view= (1000 .* ones(_n))
# 	a[1] = 1000
# end 

# function f_view2(a, _n)
# 	component_view = rand(1:n,_n)
# 	_view = view(a, component_view)
# 	_view .= (1000 .* ones(_n))
# 	a[1] = 1000
# end 

# n = 10000
# a = rand(n)
# b = rand(n)
# _n = 50

# b1 = @benchmark f_view(a,_n)
# b2 = @benchmark f_view(b,_n)