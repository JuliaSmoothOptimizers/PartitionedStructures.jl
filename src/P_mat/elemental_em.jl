module M_elemental_em

	using ..M_elt_mat

	mutable struct Elemental_em{T} <: Elt_mat{T}
		nie :: Int # nᵢᴱ
		indices :: Vector{Int} # size nᵢᴱ
		hie :: Matrix{T} # size nᵢᴱ × nᵢᴱ
	end

	get_hie(epm ::Elemental_em{T}) where T = epm.hie

	function identity_eem(nie :: Int; T=Float64, n=nie^2) 
		indices = rand(1:n, nie)
		hie = zeros(T,nie,nie)
		[hie[i,i]=1 for i in 1:nie]		
		eem = Elemental_em{T}(nie,indices,hie)
		return eem
	end 

	function ones_eem(nie :: Int; T=Float64, n=nie^2) 
		indices = rand(1:n, nie)
		hie = ones(T,nie,nie)		
		eem = Elemental_em{T}(nie,indices,hie)
		return eem
	end 

	export Elemental_em

	export get_hie
	export identity_eem, ones_eem

end 