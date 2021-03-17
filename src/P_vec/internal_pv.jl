module M_internal_pv

	using ..M_elt_vec, ..M_internal_elt_vec # element modules 
	using ..M_part_v, ..M_elemental_pv	# partitoned modules
	
	using LinearAlgebra

	mutable struct Internal_pv{T} <: Part_v{T}
		N :: Int
		n :: Int
		iev_set :: Vector{Internal_elt_vec{T}}
		v :: Vector{T}
	end

	@inline get_iev_set(ipv :: Internal_pv{T}) where T = ipv.iev_set
	@inline get_iev(ipv :: Internal_pv{T}, i :: Int) where T = ipv.iev_set[i] # i <= N

	function ipv_from_epv(epv :: Elemental_pv{T}) where T 
		N = get_N(epv)
		n = get_n(epv)
		iev_set = iev_from_eev.(get_eev_set(epv))
		v = get_v(epv)
		ipv = Internal_pv{T}(N, n, iev_set, v)
		return ipv
	end
	
	function create_ipv(iev_set :: Vector{Internal_elt_vec{T}}; n=max_indices(iev_set)  ) where T
		N = length(iev_set)
		v = zeros(T,n)
		ipv = Internal_pv{T}(N, n, iev_set, v)
		return ipv
	end	


	"""
			new_internal_pv(N,n;nᵢ,T)
	Define an internal partitionned vector of N elemental nᵢ-sized vector simulating a n-sized T-vector.
	"""
	function rand_ipv(N :: Int,n :: Int; nᵢ=3, T=Float64)		
		iev_set = Vector{Internal_elt_vec{T}}(undef,N)
		for i in 1:N
			nᵢᴱ = rand(max(nᵢ-1,0):nᵢ+1)
			nᵢᴵ = rand(max(nᵢᴱ-3,1):nᵢᴱ+1)
			iev_set[i] = new_iev(nᵢᴱ, nᵢᴵ; T=T, n=n)
		end 
		v = rand(T,n)
		return Internal_pv{T}(N, n, iev_set, v)
	end 

	# Warning: the order of ipv.indices is crucial to get the expected result.
	# the order of ipv.lin_comb, ipv.vec, ipv.indices must be synchronise	
	function M_part_v.build_v!(ipv :: Internal_pv{T}) where T
		reset_v!(ipv)
		N = get_N(ipv)		
		for i in 1:N
			ievᵢ = get_iev(ipv,i)
			nᵢᴱ = get_nie(ievᵢ)
			for j in 1:nᵢᴱ
				_id_j = get_indices(ievᵢ,j)				
				build_tmp!(ievᵢ)
				val = get_tmp(ievᵢ,j)
				add_v!(ipv, _id_j, val)
			end 
		end
		# return get_v(ipv)
	end




export Internal_pv

export get_iev, get_iev_set
export rand_ipv, create_ipv, ipv_from_epv
	
end