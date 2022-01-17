using PartitionedStructures
using PartitionedStructures.M_elemental_pm
using PartitionedStructures.M_part_mat
using PartitionedStructures.M_1_parallel, PartitionedStructures.M_2_parallel, PartitionedStructures.M_3_parallel, PartitionedStructures.M_okoubi_koko
using PartitionedStructures.M_link
using PartitionedStructures.M_part_v

using LinearAlgebra





# (epm,epv) = create_epv_epm()
(epm,epv) = create_epv_epm(;n=13)
# (epm,epv) = create_epv_epm(;n=21)
# (epm,epv) = create_epv_epm(;n=1033)
# (epm,epv) = create_epv_epm(;n=1033)
# (epm,epv) = create_epv_epm_rand(;n=1033)
m = Matrix(epm)
v = Vector(epv)
f(s) = m*s-v

s_par3 = third_parallel(epm,epv)

s = m\v
s_par1 = first_parallel(epm,epv)
s_par2 = second_parallel(epm,epv)
s_okou = okoubi(epm,epv)

println("résultats:")
# @show s, s_okou, s_par1, s_par2, s_par3
@show f(s); @show f(s_okou); @show f(s_par1); @show f(s_par2); @show f(s_par3);
 
@show norm(f(s)), norm(f(s_okou)), norm(f(s_par1)), norm(f(s_par2)), norm(f(s_par3))
# @show norm(m*s-v), norm(m*s_okou-v), norm(m*s_par1-v), norm(m*s_par2-v)

@show norm(s-s_okou), norm(s-s_par1), norm(s-s_par2), norm(s-s_par3)
@show norm(s), norm(s_okou), norm(s_par1), norm(s_par2), norm(s_par3)

# using Ipopt, ADNLPModels, NLPModelsIpopt 
# x_base = ones(5)
# f(x) = sum( (y -> y^2).(x_base.-x) )
# x0 = zeros(5)
# n=5

# nlp = ADNLPModel(x -> f(x), x0)
# res = ipopt(nlp)








# model = Model(with_optimizer(Ipopt.Optimizer))
# @variable(model, x[1:n])
# register(model, :f, 1, f; autodiff = true)
# @NLobjective(model, Min, f(x) )
# optimize!(model)
# value.(x)