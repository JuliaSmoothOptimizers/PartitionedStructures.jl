module M_elemental_pv

	using ..M_elt_vec, ..M_elemental_elt_vec	# element modules 
	using ..M_part_v # partitoned modules

	using SparseArrays
	
	import Base.==, Base.similar, Base.copy

	mutable struct Elemental_pv{T} <: Part_v{T}
		N :: Int
		n :: Int
		eev_set :: Vector{Elemental_elt_vec{T}}
		v :: Vector{T}
	end


	@inline get_eev_set(pv :: Elemental_pv{T}) where T = pv.eev_set
	@inline get_eev(pv :: Elemental_pv{T}, i :: Int) where T = pv.eev_set[i]
	@inline get_eev_value(pv :: Elemental_pv{T}, i :: Int) where T = get_vec(get_eev(pv,i))
	@inline set_eev!(pv :: Elemental_pv{T}, i :: Int, j::Int, val:: T ) where T = set_vec_eev!(get_eev(pv,i),j,val)
	@inline set_eev!(pv :: Elemental_pv{T}, i :: Int, vec ::Vector{T} ) where T = set_vec_eev!(get_eev(pv,i),vec)

	@inline (==)(ep1 :: Elemental_pv{T},ep2 :: Elemental_pv{T}) where T = (get_N(ep1) == get_N(ep2)) && (get_n(ep1) == get_n(ep2)) && (get_eev_set(ep1) == get_eev_set(ep2))
	@inline similar(ep :: Elemental_pv{T}) where T = Elemental_pv{T}(get_N(ep), get_n(ep), similar.(get_eev_set(ep)), Vector{T}(undef,get_n(ep)))
	@inline copy(ep :: Elemental_pv{T}) where T = Elemental_pv{T}(get_N(ep), get_n(ep), copy.(get_eev_set(ep)), Vector{T}(get_v(ep)))

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




#=
	Tests structures fonctions 
=#

	"""
			new_elemental_pv(N,n;nᵢ,T)
	Define an elemental partitionned vector of N elemental nᵢ-sized vector simulating a n-sized T-vector.
	"""
	function rand_epv(N :: Int,n :: Int; nie=3, T=Float64)
		eev_set = [new_eev(nie;T=T,n=n) for i in 1:N]
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

	function part_vec(;n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1)
		overlapping < nie || error("l'overlapping doit être plus faible que nie")
		mod(n-overlapping,nie-overlapping) == 0 || error("n-(nie-overlapping) doit être multiple de nie-overlapping")

		eev_set = map(i -> ones_eev(nie; T, n=n), [1:nie-overlapping:n-(nie-overlapping);])
		N = length(eev_set)
		v = Vector{T}(undef,n)
		epv = Elemental_pv{T}(N,n,eev_set,v)		
		return epv
	end


export Elemental_pv

export get_eev_set, get_eev, get_eev_value
export set_eev!
export rand_epv, create_epv, ones_kchained_epv, part_vec

end