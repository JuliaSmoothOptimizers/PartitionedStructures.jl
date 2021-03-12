# include("src/P_vec/elemental_ev.jl")
module M_elemental_elt_vec

 	using ..M_elt_vec

	# we assume that the values of vec are associate to indices.
	mutable struct Elemental_elt_vec{T} <: Elt_vec{T}
		vec :: Vector{T} # nᵢᴱ
		indices :: Vector{Int} # nᵢᴱ
		nie :: Int
	end

	@inline new_elt_ev(nᵢ::Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(rand(T,nᵢ),rand(1:n,nᵢ),nᵢ)
	
	export Elemental_elt_vec

	export new_elt_ev

end