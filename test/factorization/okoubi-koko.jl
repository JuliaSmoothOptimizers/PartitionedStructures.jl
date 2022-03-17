using LinearAlgebra
using LinearAlgebra: Matrix
using PartitionedStructures
using PartitionedStructures.ModElemental_pm
using PartitionedStructures.M_part_mat
using PartitionedStructures.Instances, PartitionedStructures.Link
using PartitionedStructures.M_part_v
using PartitionedStructures.M_okoubi_koko

(epm,epv) = create_epv_epm()
build_v!(epv)
x = okoubi(epm,epv)

A = Matrix(epm)
b = get_v(epv)

# @show A * x - b, norm(A * x - b)