using PartitionnedStructures.M_elt_vec
using PartitionnedStructures.M_elemental_em
using PartitionnedStructures.M_elemental_pm

using SparseArrays

N = 4
n = 10
nie = 3
pm1 = identity_pm(N,n; nie=nie)
pm2 = ones_pm(N,n; nie=nie)

@test Matrix(pm2.spm) == transpose(Matrix(pm2.spm))

# a = sparse([1:2:5;],[2:2:6;],ones(3))