using LinearAlgebra
using LDLFactorizations, PrettyTables

using PartitionedStructures
using PartitionedStructures.M_part_mat
using PartitionedStructures.ModElemental_pm
using PartitionedStructures.M_frontale

@testset "Frontal method" begin
  @testset "random matrices" begin
    for n = 10:30, N = Int(floor(n / 4)):Int(floor(3 * n / 4)), nie = 2:Int(floor(sqrt(n)))
      pm = ones_epm_and_id(N, n; nie = nie) # create bloc matrix without null diagonal term
      sp_pm = SparseMatrixCSC(pm)
      m = Matrix(pm)
      @test Matrix(sp_pm) == m

      LLT = cholesky(m)
      L_chol = LLT.L

      frontale!(pm)
      L_frontale = Matrix(tril(ModElemental_pm.get_L(pm)))
      m_frontale = L_frontale * L_frontale'

      @test norm(L_chol - L_frontale) ≤ 1e-6
      @test norm(m_frontale - m) ≤ 1e-6
    end
  end

  @testset "one shot" begin
    n = 100
    nie = 5
    pm = n_i_SPS(n; nie = nie) # create a tridiag dominant matrix
    sp_pm = SparseMatrixCSC(pm) # the sparse matrix from the bloc matrix
    m = Matrix(sp_pm) # Matrix format for nice print

    LLT = cholesky(m)
    frontale!(pm)

    L_chol = LLT.L
    L_frontale = Matrix(tril(ModElemental_pm.get_L(pm)))
    m_frontale = L_frontale * L_frontale'

    @test norm(L_chol - L_frontale) ≤ 1e-6
    @test norm(m_frontale - m) ≤ 1e-6
  end
end
