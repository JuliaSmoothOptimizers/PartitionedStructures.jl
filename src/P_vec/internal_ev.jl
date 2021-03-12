# include("src/P_vec/elemental_ev.jl")
module M_internal_elt_vec

 	using ..M_elt_vec

	# distinguish from internal_elt_vec
	mutable struct Internal_elt_vec{T} <: Elt_vec{T}
		vec :: Vector{T} # size nᵢᴵ
		indices :: Vector{Int} # size nᵢᴱ
		lin_comb :: Array{T,2} # size nᵢᴵ× nᵢᴱ
		nie :: Int
		nii :: Int		
	end
	
	@inline get_lin_comb(ev :: Internal_elt_vec{T}) where T = ev.lin_comb

	@inline set_lin_comb!(ev :: Internal_elt_vec{T}, lin_comb :: Array{T,2}) where T = ev.lin_comb = lin_comb

	@inline new_int_ev(nᵢᴱ::Int, nᵢᴵ:: Int; T=Float64, n=nᵢᴱ^2) = Internal_elt_vec(rand(T,nᵢᴵ),rand(1:n,nᵢᴱ), rand(T,nᵢᴵ, nᵢᴱ), nᵢᴱ, nᵢᴵ)
	

	export Internal_elt_vec	
	export get_lin_comb	
	export set_lin_comb!

	export new_int_ev

	# export get_vec, get_indices
	# export set_vec!, set_indices!
end