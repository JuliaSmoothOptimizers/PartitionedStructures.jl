using LinearAlgebra: Matrix
using PartitionnedStructures
using PartitionnedStructures.M_elemental_pm
using PartitionnedStructures.M_part_mat

using PartitionnedStructures.M_link
using PartitionnedStructures.M_part_v

using PartitionnedStructures.M_okoubi_koko

using LinearAlgebra



(epm,epv) = create_epv_epm()
build_v!(epv)
x = okoubi(epm,epv)

A = Matrix(epm)
b = get_v(epv)

@show A * x - b, norm(A * x - b)