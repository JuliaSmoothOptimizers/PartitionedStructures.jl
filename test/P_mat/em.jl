using StatsBase
using PartitionedStructures.M_elt_mat, PartitionedStructures.ModElemental_em

@testset "eem" begin
  T = Float64
  nie = 5
  n = 20
  eem1 = identity_eem(nie; T = T, n = n, bool = true)
  eem2 = ones_eem(nie; T = T, n = n, bool = false)
  @test eem1.convex == true
  @test eem2.convex == false

  cp_eem1 = copy(eem1)
  p = [1:n;]
  p_view = Vector(view(p, ModElemental_em.get_indices(eem1)))
  permute!(eem1, p_view)
  @test cp_eem1 == eem1

  p = sample(1:n, n, replace = false)
  p_view = Vector(view(p, (rand(1:n, nie))))
  permute!(eem1, p_view)
  @test cp_eem1 != eem1
end

@testset "Counters" begin
  T = Float64
  nie = 5
  n = 20
  eem = identity_eem(nie; T, n, bool = true)

  cpt_eem = get_counter_elt_mat(eem)
  cpt = Counter_elt_mat()
  cpt2 = similar(cpt)

  @test cpt == cpt2
  @test cpt == cpt_eem

  @test iter_info(cpt) == (0, 0, 0)
  @test total_info(cpt) == (0, 0, 0)
  
  update_counter_elt_mat!(cpt_eem, 1)
  update_counter_elt_mat!(cpt, 0)
  update_counter_elt_mat!(cpt2, -1)

  @test cpt != cpt2
  @test cpt != cpt_eem

  @test iter_info(cpt) == (0, 1, 0)
  @test total_info(cpt) == (0, 1, 0)
  
  @test iter_info(cpt_eem) == (1, 0, 0)
  @test total_info(cpt_eem) == (1, 0, 0)
  
  @test iter_info(cpt2) == (0, 0, 1)
  @test total_info(cpt2) == (0, 0, 1)
  
end
