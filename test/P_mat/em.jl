using StatsBase
using PartitionedStructures.M_elt_mat, PartitionedStructures.ModElemental_em

@testset "eem" begin 
  T = Float64
  nie = 5
  n = 20
  eem1 = identity_eem(nie;T=T,n=n)
  eem2 = ones_eem(nie;T=T,n=n)

  cp_eem1 = copy(eem1)
  p = [1:n;]
  p_view = Vector(view(p, ModElemental_em.get_indices(eem1)))
  permute!(eem1,p_view)
  @test cp_eem1 == eem1

  p = sample(1:n, n, replace=false)
  p_view = Vector(view(p, (rand(1:n,nie))))
  permute!(eem1,p_view)
  @test cp_eem1 != eem1
end 
