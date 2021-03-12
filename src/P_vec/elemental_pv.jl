module M_elemental_pv

	using ..M_elt_vec
	using ..M_elemental_elt_vec	
	using ..M_part_v

	mutable struct Elemental_pv{T} <: Part_v{T}
		N :: Int
		n :: Int
		elt_ev_set :: Vector{Elemental_elt_vec{T}}
		v :: Vector{T}
	end
	@inline get_elt_ev_set(pv :: Elemental_pv{T}) where T = pv.elt_ev_set
	@inline get_elt_ev(pv :: Elemental_pv{T}, i :: Int) where T = pv.elt_ev_set[i]

	"""
		build_v!(pv)
	Build from pv the vector v 
	"""
	function M_part_v.build_v!(pv :: Elemental_pv{T}) where T
		# length(get_elt_ev_set(pv)) == get_N(pv) || @warn "N ≂ length(elt_ev_set)"
		reset_v!(pv)
		N = get_N(pv)
		for i in 1:N
			evᵢ = get_elt_ev(pv,i)
			nᵢᴱ = get_nie(evᵢ)			
			for j in 1:nᵢᴱ
				add_v!(pv, get_indices(evᵢ,j), get_vec(evᵢ,j))
				# view(get_v(pv), get_indices(evᵢ,j)) .+= get_vec(evᵢ,j)
			end 
		end 
	end 


	"""
			new_elemental_pv(N,n;nᵢ,T)
	Define an elemental partitionned vector of N elemental nᵢ-sized vector simulating a n-sized T-vector.
	"""
	function new_elemental_pv(N :: Int,n :: Int; nᵢ=3, T=Float64)		
		elt_ev_set = Vector{Elemental_elt_vec{T}}(undef,N)
		for i in 1:N
			elt_ev_set[i] = new_elt_ev(nᵢ;T=T,n=n)
		end 
		v = rand(T,n)
		return Elemental_pv{T}(N, n, elt_ev_set, v)
	end 

export Elemental_pv

export new_elemental_pv

end