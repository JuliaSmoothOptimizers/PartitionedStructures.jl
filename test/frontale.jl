using PartitionnedStructures
using PartitionnedStructures.M_elemental_pm
using PartitionnedStructures.M_part_mat
using PartitionnedStructures.M_frontale

using LinearAlgebra, LDLFactorizations


#=
	Test de la méthode multifrontale sur des matrices générées aléatoirement
=#
not_last && @testset "test frontal method" begin 
	for n in 10:50, N in Int(floor(n/4)):Int(floor(3*n/4)), nie in 2:Int(floor(sqrt(n)))	
		pm = ones_epm_and_id(N,n; nie=nie) # create bloc matrix without null diagonal term
		sp_pm = get_spm(pm)
		m = Matrix(sp_pm)

		LLT = cholesky(m)
		L_chol = LLT.L

		frontale!(pm)
		L_frontale = Matrix(tril(get_L(pm)))
		m_frontale = L_frontale * L_frontale'

		@test norm(L_chol - L_frontale) ≤	1e-6 
		@test norm(m_frontale - m ) ≤ 1e-6
	end 
end 



# Exemple pouvant être plus facilement manipuler
# attention aux grandes dimensions le pattern de ones_epm_and_id(N,n,nie) n'est pas bon de manière générale.
N = 15
n = 30
nie = 5
pm = ones_epm_and_id(N,n; nie=nie) # create bloc matrix without null diagonal term
sp_pm = get_spm(pm) # the sparse matrix from the bloc matrix
m = Matrix(sp_pm) # Matrix format for nice print

LLT = cholesky(m)
L_chol = LLT.L

frontale!(pm)
L_frontale = Matrix(tril(get_L(pm)))
m_frontale = L_frontale * L_frontale'

@test norm(L_chol - L_frontale) ≤	1e-6 
@test norm(m_frontale - m ) ≤ 1e-6

bench_chol = @benchmark cholesky(m)
bench_frontale = @benchmark frontale!(pm)

# afficher m et m_frontale