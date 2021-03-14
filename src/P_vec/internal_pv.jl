module M_internal_pv

	using ..M_internal_elt_vec
	using ..M_part_v

	mutable struct Internal_pv{T} <: Part_v{T}
		N :: Int
		n :: Int
		int_ev_set :: Vector{Internal_elt_vec{T}}
		v :: Vector{T}
	end
	

	
	"""
	new_internal_pv(N,n;nᵢ,T)
	Define an internal partitionned vector of N elemental nᵢ-sized vector simulating a n-sized T-vector.
	"""
	function rand_ipv(N :: Int,n :: Int; nᵢ=3, T=Float64)		
		int_ev_set = Vector{Internal_elt_vec{T}}(undef,N)
		for i in 1:N
			nᵢᴱ = rand(max(nᵢ-1,0):nᵢ+1)
			nᵢᴵ = rand(max(nᵢᴱ-3,1):nᵢᴱ+1)
			int_ev_set[i] = new_int_ev(nᵢᴱ, nᵢᴵ; T=T, n=n)
		end 
		v = rand(T,n)
		return Internal_pv{T}(N, n, int_ev_set, v)
	end 

export Internal_pv

export rand_ipv
	
end