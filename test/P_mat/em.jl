using PartitionnedStructures.M_elt_vec
using PartitionnedStructures.M_elemental_em

T = Float64
nie = 5
n = 20
eem1 = identity_eem(nie;T=T,n=n)
eem2 = ones_eem(nie;T=T,n=n)