module ModElemental_pm
# Symmetric bloc elemental partitioned matrix

	using SparseArrays
	using LoopVectorization

	using ..M_part_mat
	using ..M_elt_mat, ..ModElemental_em, ..M_abstract_element_struct

	import Base.==, Base.copy, Base.similar
	import ..M_part_mat.set_spm!
	# , ..M_part_mat.get_spm

	import Base.Matrix, SparseArrays.SparseMatrixCSC
	
	
	
	
	mutable struct Elemental_pm{T} <: Part_mat{T}
		N :: Int
		n :: Int
		eem_set :: Vector{Elemental_em{T}}
		spm :: SparseMatrixCSC{T,Int}
		L :: SparseMatrixCSC{T,Int}
		component_list :: Vector{Vector{Int}}
		permutation :: Vector{Int} # n-size vector 
	end

	#getter/setter
	@inline get_eem_set(epm :: Elemental_pm{T}) where T = epm.eem_set
	@inline get_eem_set(epm :: Elemental_pm{T}, i::Int) where T = @inbounds epm.eem_set[i]
	@inline get_eem_sub_set(epm :: Elemental_pm{T}, indices::Vector{Int}) where T = epm.eem_set[indices]
	@inline get_eem_set_Bie(epm :: Elemental_pm{T}, i::Int) where T = get_Bie(get_eem_set(epm,i))

	# @inline get_spm(epm :: Elemental_pm{T}) where T = epm.spm
	# @inline get_spm(epm :: Elemental_pm{T}, i :: Int, j :: Int) where T = @inbounds epm.spm[i,j]
	@inline get_L(epm :: Elemental_pm{T}) where T = epm.L
	@inline get_L(epm :: Elemental_pm{T}, i :: Int, j :: Int) where T = @inbounds epm.L[i,j]
	@inline get_component_list(epm :: Elemental_pm{T}) where T = epm.component_list
	@inline get_component_list(epm :: Elemental_pm{T},i::Int) where T = @inbounds epm.component_list[i]

	@inline set_L!(epm :: Elemental_pm{T}, i :: Int, j :: Int, v :: T) where T = @inbounds epm.L[i,j] = v
	@inline set_L_to_spm!(epm :: Elemental_pm{T}) where T = epm.L = copy(epm.spm)
	
	
	
	@inline (==)(epm1 :: Elemental_pm{T}, epm2 :: Elemental_pm{T}) where T = (get_N(epm1) == get_N(epm2)) && (get_n(epm1) == get_n(epm2)) && (get_eem_set(epm1).== get_eem_set(epm2)) && (get_permutation(epm1) == get_permutation(epm2))
	@inline copy(epm :: Elemental_pm{T}) where T = Elemental_pm{T}(copy(get_N(epm)),copy(get_n(epm)),copy.(get_eem_set(epm)),copy(get_spm(epm)), copy(get_L(epm)),copy(get_component_list(epm)),copy(get_permutation(epm)))
	@inline similar(epm :: Elemental_pm{T}) where T = Elemental_pm{T}(copy(get_N(epm)),copy(get_n(epm)),similar.(get_eem_set(epm)),similar(get_spm(epm)), similar(get_L(epm)),copy(get_component_list(epm)),copy(get_permutation(epm)))

	"""
		identity_epm(N,n; type, nie)
	Create a a partitionned matrix of N nie-identity blocs whose positions are randoms
	"""
	function identity_epm(N :: Int, n ::Int; T=Float64, nie::Int=5)		
		eem_set = map(i -> identity_eem(nie;T=T,n=n), [1:N;])
		spm = spzeros(T,n,n)
		L = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		no_perm = [1:n;]
		epm = Elemental_pm{T}(N,n,eem_set,spm,L,component_list,no_perm)
		initialize_component_list!(epm)
		# set_spm!(epm)
		return epm
	end 

	"""
		ones_epm(N,n; type, nie)
	Create a a partitionned matrix of N ones(nie,nie) blocs whose positions are randoms.
	Be careful the partitionned matrix created may be singular.
	"""
	function ones_epm(N :: Int, n ::Int; T=Float64, nie::Int=5)		
		eem_set = map(i -> ones_eem(nie;T=T,n=n), [1:N;])
		spm = spzeros(T,n,n)
		L = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		no_perm= [1:n;]
		epm = Elemental_pm{T}(N,n,eem_set,spm,L,component_list,no_perm)
		initialize_component_list!(epm)
		# set_spm!(epm)
		return epm
	end 

	"""
			ones_epm_and_id(N,n; type, nie)
	Create a a partitionned matrix of N ones(nie,nie) blocs whose positions are randoms and adding .
	Not singular.
	"""
	function ones_epm_and_id(N :: Int, n ::Int; T=Float64, nie::Int=5)		
		eem_set1 = map(i -> ones_eem(nie;T=T,n=n), [1:N;])
		eem_set2 = map(i -> one_size_bloc(i;T=T), [1:n;])
		eem_set = vcat(eem_set1,eem_set2)
		spm = spzeros(T,n,n)
		L = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		no_perm= [1:n;]
		epm = Elemental_pm{T}(N+n,n,eem_set,spm,L,component_list,no_perm)
		initialize_component_list!(epm)
		# set_spm!(epm)
		return epm
	end 


	"""
			n_i_diag_dom(n)
	A nᵢ bloc separable matrix
	By default nᵢ = 5
	"""
	function n_i_sep(n ::Int; T=Float64, nie::Int=5, mul=5.)
		mod(n,nie) == 0 || error("n doit être multiple de nie")
		eem_set = map(i -> fixed_ones_eem(i,nie;T=T, mul=mul), [1:nie:n;])
		spm = spzeros(T,n,n)
		L = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		no_perm = [1:n;]
		N = Int(floor(n/nie))	
		epm = Elemental_pm{T}(N,n,eem_set,spm,L,component_list,no_perm)
		initialize_component_list!(epm)
		# set_spm!(epm)
		return epm
	end 

	"""
			n_i_SPS(n)
	A nᵢ bloc separable matrix
	By default nᵢ = 5
	"""
	function n_i_SPS(n ::Int; T=Float64, nie::Int=5, overlapping::Int=1, mul=5.)
		mod(n,nie) == 0 || error("n doit être multiple de nie")
		overlapping < nie || error("l'overlapping doit être plus faible que nie")
		eem_set1 = map(i -> fixed_ones_eem(i,nie;T=T,mul=mul), [1:nie:n;])
		# eem_set2 = map(i -> fixed_ones_eem(i,nie;T=T,mul=mul), [1+overlapping:nie:n-nie+overlapping;])
		eem_set2 = map(i -> fixed_ones_eem(i,2*overlapping;T=T,mul=mul), [nie-overlapping:nie:n-nie-overlapping;])
		eem_set = vcat(eem_set1,eem_set2)
		spm = spzeros(T,n,n)
		L = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		no_perm = [1:n;]
		N = length(eem_set)
		epm = Elemental_pm{T}(N,n,eem_set,spm,L,component_list,no_perm)
		initialize_component_list!(epm)
		# set_spm!(epm)
		return epm
	end 

	"""
			part_mat(n)
	A nᵢ partially bloc separable matrix, whith on the diagonal band regular with overlapping (=1 by default)
	By default nᵢ=5, overlapping=1, mul=5.
	"""
	function part_mat(;n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1, mul=5.)
		overlapping < nie || error("l'overlapping doit être plus faible que nie")
		mod(n-(nie-overlapping), nie-overlapping) == mod(overlapping, nie-overlapping) || error("wrong structure: mod(n-(nie-over), nie-over) == mod(over, nie-over) must holds")
		indices = filter(x -> x <= n-nie+1, vcat(1,(x -> x + (nie-overlapping)).([1:nie-overlapping:n-(nie-overlapping);])))
		eem_set = map(i -> fixed_ones_eem(i,nie;T=T,mul=mul), indices)
		spm = spzeros(T,n,n)
		L = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		no_perm = [1:n;]
		N = length(eem_set)
		epm = Elemental_pm{T}(N,n,eem_set,spm,L,component_list,no_perm)
		initialize_component_list!(epm)		
		# set_spm!(epm)
		return epm
	end 



	"""
		initialize_component_list!(epm)
	initialize_component_list! Build for each index i (∈ {1,...,n}) the list of the blocs using i.
	"""
	function initialize_component_list!(epm)
		N = get_N(epm)
		n = get_n(epm)
		for i in 1:N
			eemᵢ = get_eem_set(epm,i)
			_indices = get_indices(eemᵢ)
			for j in _indices 
				push!(get_component_list(epm,j),i)
			end 
		end 
	end 


	"""
		set_spm!(epm)
	Build the sparse matrix spm from the blocs epm.eem_set, according to the indinces.
	"""
	function set_spm!(epm :: Elemental_pm{T}) where T
		reset_spm!(epm) # epm.spm .= 0
		N = get_N(epm)
		n = get_n(epm)
		spm = get_spm(epm)
		for i in 1:N
			epmᵢ = get_eem_set(epm,i)
			nie = get_nie(epmᵢ)
			Bie = get_Bie(epmᵢ)
			for i in 1:nie, j in 1:nie
				val = Bie[i,j]
				real_i = get_indices(epmᵢ,i) # epmᵢ.indices[i]
				real_j = get_indices(epmᵢ,j) # epmᵢ.indices[j]
				spm[real_i, real_j] += val 
			end 
		end 
	end


	import Base.permute! 
	"""
			permute!(epm,p)
	apply the permutation p to the elemental partitionned matrix epm.
	The permutation is applied to each eem via indices.
	The current epm permutation is stored in epm.permutation
	"""
	function permute!(epm :: Elemental_pm{T}, p :: Vector{Int}) where T
		N = get_N(epm)
		n = get_n(epm)
		# permute on element matrix 
		for i in 1:N
			epmᵢ = get_eem_set(epm,i)
			indicesᵢ = get_indices(epmᵢ)
			e_perm = Vector(view(p,indicesᵢ))
			permute!(epmᵢ,e_perm)
		end 
		# permute on the permutation vector
		perm = get_permutation(epm)
		permute!(perm, p)
		# permute component list
		new_component_list = Vector{Vector{Int}}(undef,n)
		for i in 1:n
			new_component_list[i] = get_component_list(epm, p[i])
		end 
		# hard reset of the sparse matrix
		hard_reset_spm!(epm)
	end 

	"""
			correlated_var(epm,i) 
	correlated_var(epm,i) get the linked vars to i depending the structure of epm.
	"""
	function correlated_var(epm :: Elemental_pm{T}, i :: Int) where T
		component_list = get_component_list(epm)
		bloc_list = component_list[i]
		indices_list = Vector{Int}(undef,0)
		for (id_j,j) in enumerate(bloc_list)
			eemᵢ = get_eem_set(epm,j)
			_indices = get_indices(eemᵢ)
			append!(indices_list, _indices)
		end
		var_list = vcat(indices_list...)
		unique!(var_list)
		return var_list
	end 

	
	function Base.Matrix(epm :: Elemental_pm{T}) where T
		set_spm!(epm)
		sp_pm = get_spm(epm)
		m = Matrix(sp_pm)
		return m
	end 

	SparseArrays.SparseMatrixCSC(epm :: Elemental_pm{T}) where T = begin set_spm!(epm); get_spm(epm) end


	export Elemental_pm

	export get_eem_set, get_spm, get_L, get_component_list, get_eem_set_Bie, get_eem_sub_set
	export set_L!, set_L_to_spm!

	export initialize_component_list!, correlated_var
	export reset_spm!, set_spm!, set_L_to_spm!

	export identity_epm, ones_epm, ones_epm_and_id, n_i_sep, n_i_SPS, part_mat
end