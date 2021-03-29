module M_elemental_em

	using LinearAlgebra
	
	using ..M_elt_mat

	mutable struct Elemental_em{T} <: Elt_mat{T}
		nie :: Int # nᵢᴱ
		indices :: Vector{Int} # size nᵢᴱ
		hie :: Symmetric{T,Matrix{T}} # size nᵢᴱ × nᵢᴱ
	end

	get_hie(eem :: Elemental_em{T}) where T = eem.hie

	import Base.==
	(==)(eem1 :: Elemental_em{T}, eem2 :: Elemental_em{T}) where T = (get_nie(eem1)== get_nie(eem2)) && (get_hie(eem1)== get_hie(eem2)) && (get_indices(eem1)== get_indices(eem2))

	import Base.copy
	copy(eem :: Elemental_em{T}) where T = Elemental_em{T}(copy(get_nie(eem)), copy(get_indices(eem)), copy(get_hie(eem)))

	# function creating elemental element matrix 

	function identity_eem(nie :: Int; T=Float64, n=nie^2) 
		indices = rand(1:n, nie)
		hie = zeros(T,nie,nie)
		[hie[i,i]=1 for i in 1:nie]		
		eem = Elemental_em{T}(nie,indices,Symmetric(hie))
		return eem
	end 

	function ones_eem(nie :: Int; T=Float64, n=nie^2) 
		indices = rand(1:n, nie)
		hie = ones(T,nie,nie)		
		eem = Elemental_em{T}(nie,indices,Symmetric(hie))
		return eem
	end 

	function fixed_ones_eem(i::Int, nie :: Int; T=Float64) 
		indices = [i:(i+nie);]
		hie = ones(T,nie,nie)		
		[hie[i,i] = 5 for i in 1:nie]
		eem = Elemental_em{T}(nie,indices,Symmetric(hie))
		return eem
	end 

	one_size_bloc(index :: Int; T=Float64) = Elemental_em{T}(1,[index], Symmetric(ones(1,1)))


	import Base.permute!
	permute!(eem :: Elemental_em{T}, p :: Vector{Int}) where T = eem.indices .= p


	export Elemental_em

	export get_hie
	export identity_eem, ones_eem, fixed_ones_eem, one_size_bloc

end 