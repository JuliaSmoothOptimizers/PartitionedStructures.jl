module M_elemental_pv

	using ..M_elt_vec, ..M_elemental_elt_vec	# element modules 
	using ..M_part_v # partitoned modules

	using SparseArrays

	mutable struct Elemental_pv{T} <: Part_v{T}
		N :: Int
		n :: Int
		eev_set :: Vector{Elemental_elt_vec{T}}
		v :: Vector{T}
	end
	@inline get_eev_set(pv :: Elemental_pv{T}) where T = pv.eev_set
	@inline get_eev(pv :: Elemental_pv{T}, i :: Int) where T = pv.eev_set[i]


	"""
		build_v!(pv)
	Build from pv the vector v according to the information of each {evᵢ}ᵢ
	"""
	function M_part_v.build_v!(epv :: Elemental_pv{T}) where T
		reset_v!(epv)
		N = get_N(epv)
		for i in 1:N
			eevᵢ = get_eev(epv,i)
			nᵢᴱ = get_nie(eevᵢ)			
			for j in 1:nᵢᴱ
				add_v!(epv, get_indices(eevᵢ,j), get_vec(eevᵢ,j))
			end 
		end 
	end 

	
	"""
			create_elemental_pv(elt_ev_set)
	create easily a pv from elt_ev_set (confort)
	"""
	@inline create_epv(sp_set :: Vector{SparseVector{T,Y}}; kwargs...) where {T,Y} = create_epv(eev_from_sparse_vec.(sp_set); kwargs...)
	function create_epv(eev_set :: Vector{Elemental_elt_vec{T}}; n=max_indices(eev_set)  ) where T
		N = length(eev_set)
		v = zeros(T,n)
		Elemental_pv{T}(N, n, eev_set, v)
	end	


	import Base.==
	(==)(ep1 :: Elemental_pv{T},ep2 :: Elemental_pv{T}) where T = (get_N(ep1) == get_N(ep2)) && (get_n(ep1) == get_n(ep2)) && (get_eev_set(ep1) == get_eev_set(ep2))
#=
	Tests structures fonctions 
=#

	"""
			new_elemental_pv(N,n;nᵢ,T)
	Define an elemental partitionned vector of N elemental nᵢ-sized vector simulating a n-sized T-vector.
	"""
	function rand_epv(N :: Int,n :: Int; nᵢ=3, T=Float64)
		eev_set = [new_eev(nᵢ;T=T,n=n) for i in 1:N]
		v = zeros(T,n)
		return Elemental_pv{T}(N, n, eev_set, v)
	end 

	"""
			ones_kchained_epv(N, k; T)
	Construct a N-partitionned k-sized vector such as n = N+k.
	eltype(pv)=T
	"""
	function ones_kchained_epv(N :: Int, k :: Int; T=Float64)
		n = N+k
		nᵢ = k
		eev_set = [ones_eev(nᵢ;T=T,n=n) for i in 1:N]
		v = zeros(T,n)
		return Elemental_pv{T}(N, n, eev_set, v)
	end


export Elemental_pv

export get_eev_set, get_eev
export rand_epv, create_epv, ones_kchained_epv

end