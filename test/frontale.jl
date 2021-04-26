using PartitionnedStructures
using PartitionnedStructures.M_elemental_pm
using PartitionnedStructures.M_part_mat
using PartitionnedStructures.M_frontale

using LinearAlgebra

using LDLFactorizations, PrettyTables

#=
	Test de la méthode multifrontale sur des matrices générées aléatoirement
=#
not_last && @testset "test frontal method" begin 
	for n in 10:30, N in Int(floor(n/4)):Int(floor(3*n/4)), nie in 2:Int(floor(sqrt(n)))	
		pm = ones_epm_and_id(N,n; nie=nie) # create bloc matrix without null diagonal term
		sp_pm = SparseMatrixCSC(pm)
		m = Matrix(pm)
		@test Matrix(sp_pm) == m

		LLT = cholesky(m)
		L_chol = LLT.L

		frontale!(pm)
		L_frontale = Matrix(tril(get_L(pm)))
		m_frontale = L_frontale * L_frontale'

		@test norm(L_chol - L_frontale) ≤	1e-6 
		@test norm(m_frontale - m ) ≤ 1e-6
		
	end 
end 

not_last && @testset "génération de matrice" begin 
	N = 5 
	n = 10 #by default must be a mulitple of 5
	id_epm = identity_epm(N,n)
	id_m = Matrix(id_epm)

	ones_pm = ones_epm(N,n)
	one_m = Matrix(ones_pm)

	ones_id_pm = ones_epm_and_id(N,n)
	one_id_m = Matrix(ones_id_pm)

	n_i_sep_pm = n_i_sep(n)
	sep_m = Matrix(n_i_sep_pm)

	n_i_sps_pm = n_i_SPS(n; overlapping=1)
	sps_sp_m = SparseMatrixCSC(n_i_sps_pm)	
	sps_m = Matrix(n_i_sps_pm)
end 





# Exemple pouvant être plus facilement manipuler
# attention aux grandes dimensions le pattern de ones_epm_and_id(N,n,nie) n'est pas bon de manière générale.

n = 100
nie = 5
# N = 15
# pm = ones_epm_and_id(N,n; nie=nie) # create bloc matrix without null diagonal term
pm = n_i_SPS(n; nie=nie) # create a tridiag dominant matrix
sp_pm = SparseMatrixCSC(pm) # the sparse matrix from the bloc matrix
m = Matrix(sp_pm) # Matrix format for nice print

LLT = cholesky(m)
frontale!(pm)

L_chol = LLT.L
L_frontale = Matrix(tril(get_L(pm)))
m_frontale = L_frontale * L_frontale'

@test norm(L_chol - L_frontale) ≤	1e-6 
@test norm(m_frontale - m ) ≤ 1e-6

# bench_chol = @benchmark cholesky(m)
# bench_frontale = @benchmark frontale!(pm)
# bench_sparse = @benchmark ldl(sp_pm)

# ProfileView.@profview (@benchmark frontale!(pm))

# @code_warntype frontale!(pm) 
# afficher m et m_frontale

N = 5 
n = 10 #by default must be a mulitple of 5
id_epm = identity_epm(N,n)
id_m = Matrix(id_epm)

ones_pm = ones_epm(N,n)
one_m = Matrix(ones_pm)

ones_id_pm = ones_epm_and_id(N,n)
one_id_m = Matrix(ones_id_pm)

n_i_sep_pm = n_i_sep(n)
sep_m = Matrix(n_i_sep_pm)

n_i_sps_pm = n_i_SPS(n; overlapping=1)
sps_sp_m = SparseMatrixCSC(n_i_sps_pm)
ldl(sps_sp_m)
sps_m = Matrix(n_i_sps_pm)


names = []
vector_n = [100,200,500,1000,5000,10000]
vector_ni = [5,10,20]
vector_chevauchement = [1,2,3]

time_chol = []
time_sparse = []
time_frontale = []

memory_chol = []
memory_sparse = []
memory_frontale = []

allocs_chol = []
allocs_sparse = []
allocs_frontale = []

for (id_n,n) in enumerate(vector_n)
	vector_N = Vector{Int}([n/10:n/10:2*n;])
	for N in vector_N 
		for ni in vector_ni
			for chevauchement in vector_chevauchement
				pm = n_i_SPS(n; nie=nie, overlapping=chevauchement) # create a overlapping bloc diagonale matrix
				sp_m = SparseMatrixCSC(pm) # the sparse matrix from the partitioned matrix
				m = Matrix(sp_pm) # dense one

				name = "$(n)_"*"$(N)_"*"$(ni)_"*"$(chevauchement)"
				println(name)
				push!(names, name)

				bench_chol = @timed cholesky(m)
				push!(memory_chol, bench_chol.bytes)
				push!(allocs_chol, bench_chol.gcstats.malloc)
				push!(time_chol, bench_chol.time)

				bench_sparse = @timed ldl(sp_m)
				push!(memory_sparse, bench_sparse.bytes)
				push!(allocs_sparse, bench_sparse.gcstats.malloc)
				push!(time_sparse, bench_sparse.time)

				bench_frontale = @timed frontale!(pm)
				push!(memory_frontale, bench_frontale.bytes)
				push!(allocs_frontale, bench_frontale.gcstats.malloc)
				push!(time_frontale, bench_frontale.time)
			end 			
		end 
	end 
end 

data = Matrix(undef,length(names),10)

for (id,name) in enumerate(names)
	data[id,1] = name

	data[id,2] = time_chol[id]
	data[id,3] = time_sparse[id]
	data[id,4] = time_frontale[id]

	data[id,5] = memory_chol[id]
	data[id,6] = memory_sparse[id]
	data[id,7] = memory_frontale[id]

	data[id,8] = allocs_chol[id]
	data[id,9] = allocs_sparse[id]
	data[id,10] = allocs_frontale[id]
end 

pretty_table(data, ["name","t chol","t sparse", "t frontale","memory chol", "memory sparse", "memory frontale", "allocs chol", "allocs sparse", "allocs frontale"])