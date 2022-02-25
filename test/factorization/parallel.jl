using LinearAlgebra
using PartitionedStructures
using PartitionedStructures.ModElemental_pm
using PartitionedStructures.M_part_mat
using PartitionedStructures.M_1_parallel, PartitionedStructures.M_2_parallel, PartitionedStructures.M_3_parallel, PartitionedStructures.M_okoubi_koko
using PartitionedStructures.Link
using PartitionedStructures.M_part_v

# (epm,epv) = create_epv_epm()
(epm,epv) = create_epv_epm(;n=17, nie =7, overlapping=2)
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

# println("résults:")
# @show f(s); @show f(s_okou); @show f(s_par1); @show f(s_par2); @show f(s_par3); 
@show norm(f(s)), norm(f(s_okou)), norm(f(s_par1)), norm(f(s_par2)), norm(f(s_par3))

@show norm(s-s_okou), norm(s-s_par1), norm(s-s_par2), norm(s-s_par3)
@show norm(s), norm(s_okou), norm(s_par1), norm(s_par2), norm(s_par3)
