using PartitionnedStructures
using PartitionnedStructures.M_elemental_pm
using PartitionnedStructures.M_part_mat
using PartitionnedStructures.M_1_parallel
using PartitionnedStructures.M_link

using LinearAlgebra



N=11
n=10
nie=3
(epm,epv) = create_epv_epm()

first_parallel(epm,epv)