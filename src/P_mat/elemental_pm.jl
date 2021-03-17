module M_elemental_pm

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

	get_eem_set(epm :: Elemental_pm{T}) where T = epm.eem_set
	get_eem_set(epm :: Elemental_pm{T}, i::Int) where T = epm.eem_set[i]

	get_spm(epm :: Elemental_pm{T}) where T = epm.spm
	get_component_list(epm :: Elemental_pm{T}) where T = epm.component_list
	get_component_list(epm :: Elemental_pm{T},i::Int) where T = epm.component_list[i]
	
	function identity_pm(N :: Int, n ::Int; T=Float64, nie::Int=5)		
		eem_set = map(i -> identity_eem(nie;T=T,n=n), [1:N;])
		spm = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		epm = Elemental_pm{T}(N,n,eem_set,spm,component_list)
		initialize_component_list!(epm)
		set_spm!(epm)
		return epm
	end 

	function ones_pm(N :: Int, n ::Int; T=Float64, nie::Int=5)		
		eem_set = map(i -> ones_eem(nie;T=T,n=n), [1:N;])
		spm = spzeros(T,n,n)
		component_list = map(i -> Vector{Int}(undef,0), [1:n;])
		epm = Elemental_pm{T}(N,n,eem_set,spm,component_list)
		initialize_component_list!(epm)
		set_spm!(epm)
		return epm
	end 

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

	reset_spm!(epm :: Elemental_pm{T}) where T = epm.spm .= (T)(0)
	function set_spm!(epm :: Elemental_pm{T}) where T
		reset_spm!(epm)
		N = get_N(epm)
		n = get_n(epm)
		spm = get_spm(epm)
		for i in 1:N
			epmᵢ = get_eem_set(epm,i)
			nie = get_nie(epmᵢ)
			_indices = get_indices(epmᵢ)
			hie = get_hie(epmᵢ)
			for i in 1:nie, j in 1:nie
				val = hie[i,j]
				real_i = _indices[i]
				real_j = _indices[j]
				spm[real_i,real_j] += val 
			end 
		end 
	end

	export Elemental_pm

	export get_eem_set, get_spm, get_component_list
	export initialize_component_list!
	export reset_spm!, set_spm!
	export identity_pm, ones_pm
end