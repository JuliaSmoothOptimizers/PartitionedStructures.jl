module M_elemental_pv

	using ..M_elt_vec
	using ..M_elemental_elt_vec	
	using ..M_part_v

	using SparseArrays

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
	Build from pv the vector v according to the information of each {evᵢ}ᵢ
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
			create_elemental_pv(elt_ev_set)
	create easily a pv from elt_ev_set (confort)
	"""
	create_elemental_pv(sp_set :: Vector{SparseVector{T}}) where T = create_elemental_pv(elt_ev_from_sparse_vec.(sp_set))
	function create_elemental_pv(elt_ev_set :: Vector{Elemental_elt_vec{T}}) where T
		N = length(elt_ev_set)
		n = get_var_from_indices(elt_ev_set)
		v = zeros(T,n)
		Elemental_pv{T}(N, n, elt_ev_set, v)
	end	

#=
	Tests structures fonctions 
=#

	"""
			new_elemental_pv(N,n;nᵢ,T)
	Define an elemental partitionned vector of N elemental nᵢ-sized vector simulating a n-sized T-vector.
	"""
	function rand_epv(N :: Int,n :: Int; nᵢ=3, T=Float64)
		elt_ev_set = [new_elt_ev(nᵢ;T=T,n=n) for i in 1:N]
		v = zeros(T,n)
		return Elemental_pv{T}(N, n, elt_ev_set, v)
	end 

	"""
			ones_kchained_epv(N, k; T)
	Construct a N-partitionned k-sized vector such as n = N+k.
	eltype(pv)=T
	"""
	function ones_kchained_epv(N :: Int, k :: Int; T=Float64)
		n = N+k
		nᵢ = k
		elt_ev_set = [ones_elt_ev(nᵢ;T=T,n=n) for i in 1:N]
		v = zeros(T,n)
		return Elemental_pv{T}(N, n, elt_ev_set, v)
	end


export Elemental_pv

export rand_epv, ones_kchained_epv
export create_elemental_pv

end