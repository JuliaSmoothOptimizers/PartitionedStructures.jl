using PartitionnedStructures.M_elt_vec
using PartitionnedStructures.M_elemental_em
using PartitionnedStructures.M_elemental_pm

using SparseArrays


@testset "first tests on epm" begin 
	N = 4
	n = 10
	nie = 3
	pm1 = identity_epm(N,n; nie=nie)
	pm2 = ones_epm(N,n; nie=nie)

	@test Matrix(pm2.spm) == transpose(Matrix(pm2.spm))
	@test sum(Matrix(pm2.spm)) == N * nie^2
	@test sum(Matrix(pm1.spm)) == N * nie
	# a = sparse([1:2:5;],[2:2:6;],ones(3))

	@test (@allocated reset_spm!(pm1)) == 0
	@test (@allocated set_spm!(pm1)) == 0
end 

N = 100
n = 200
nie = 5
pm1 = identity_epm(N,n; nie=nie)
pm2 = ones_epm(N,n; nie=nie)

b1 = @benchmark reset_spm!(pm1)
b2 = @benchmark set_spm!(pm1)

# @code_warntype set_spm!(pm1)
# ProfileView.@profview (@benchmark set_spm!(pm1))


# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     36.400 μs (0.00% GC)
#   median time:      36.999 μs (0.00% GC)
#   mean time:        39.761 μs (0.00% GC)
#   maximum time:     181.901 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1