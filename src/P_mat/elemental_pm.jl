module M_elemental_pm
# Symmetric bloc elemental partitioned matrix

	using SparseArrays

	using ..M_part_mat
	using ..M_elt_mat, ..M_elemental_em
	
	
	mutable struct Elemental_pm{T} <: Part_mat{T}
		N :: Int
		n :: Int
		eem_set :: Vector{Elemental_em{T}}
		spm :: SparseMatrixCSC{T,Int}
		component_list :: Vector{Vector{Int}}
	end

	#getter/setter
	get_eem_set(epm :: Elemental_pm{T}) where T = epm.eem_set
	get_eem_set(epm :: Elemental_pm{T}, i::Int) where T = epm.eem_set[i]

	get_spm(epm :: Elemental_pm{T}) where T = epm.spm
	get_component_list(epm :: Elemental_pm{T}) where T = epm.component_list
	get_component_list(epm :: Elemental_pm{T},i::Int) where T = epm.component_list[i]
	
	"""
		identity_epm(N,n; type, nie)
	Create a a partitionned matrix of N nie-identity blocs whose positions are randoms
	"""
	function identity_epm(N :: Int, n ::Int; T=Float64, nie::Int=5)		
		eem_set = map(i -> identity_eem(nie;T=T,n=n), [1:N;])
		spm = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		epm = Elemental_pm{T}(N,n,eem_set,spm,component_list)
		initialize_component_list!(epm)
		set_spm!(epm)
		return epm
	end 

	"""
		ones_epm(N,n; type, nie)
	Create a a partitionned matrix of N ones(nie,nie) blocs whose positions are randoms
	"""
	function ones_epm(N :: Int, n ::Int; T=Float64, nie::Int=5)		
		eem_set = map(i -> ones_eem(nie;T=T,n=n), [1:N;])
		spm = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		epm = Elemental_pm{T}(N,n,eem_set,spm,component_list)
		initialize_component_list!(epm)
		set_spm!(epm)
		return epm
	end 


	"""
		initialize_component_list!(epm)
	initialize_component_list! Build for each index i the list of the blocs using i.
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
		reset_spm!(epm)
	Reset the sparse matrix epm.spm
	"""
	# @inline reset_spm!(epm :: Elemental_pm{T}) where T = epm.spm .= (T)(0)
	@inline reset_spm!(epm :: Elemental_pm{T}) where T = epm.spm.nzval .= (T)(0) #.nzval delete the 1 alloc

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
			hie = get_hie(epmᵢ)
			for i in 1:nie, j in 1:nie
				val = hie[i,j]
				real_i = get_indices(epmᵢ,i) # epmᵢ.indices[i]
				real_j = get_indices(epmᵢ,j) # epmᵢ.indices[j]
				spm[real_i, real_j] += val 
			end 
		end 
	end

	export Elemental_pm

	export get_eem_set, get_spm, get_component_list
	export initialize_component_list!
	export reset_spm!, set_spm!
	export identity_epm, ones_epm
end