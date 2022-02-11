module ModElemental_em

	using LinearAlgebra
	using ..M_elt_mat, ..M_abstract_element_struct

	import Base.==, Base.copy, Base.similar

	export Elemental_em
	export get_Bie
	export identity_eem, ones_eem, fixed_ones_eem, one_size_bloc, create_id_eem

	mutable struct Elemental_em{T} <: Elt_mat{T}
		nie :: Int # nᵢᴱ
		indices :: Vector{Int} # size nᵢᴱ
		Bie :: Symmetric{T,Matrix{T}} # size nᵢᴱ × nᵢᴱ
	end

	
	@inline (==)(eem1 :: Elemental_em{T}, eem2 :: Elemental_em{T}) where T = (get_nie(eem1)== get_nie(eem2)) && (get_Bie(eem1)== get_Bie(eem2)) && (get_indices(eem1)== get_indices(eem2))
	@inline copy(eem :: Elemental_em{T}) where T = Elemental_em{T}(copy(get_nie(eem)), copy(get_indices(eem)), copy(get_Bie(eem)))
	@inline similar(eem :: Elemental_em{T}) where T = Elemental_em{T}(copy(get_nie(eem)), copy(get_indices(eem)), similar(get_Bie(eem)))

	# function creating elemental element matrix 

	function create_id_eem(elt_var::Vector{Int}; type=Float64)
	  nie = length(elt_var)
	  Bie = zeros(type,nie,nie)
	  [Bie[i,i]=1 for i in 1:nie]  
	  eem = Elemental_em{type}(nie, elt_var, Symmetric(Bie))
	  return eem
	end

	function identity_eem(nie :: Int; T=Float64, n=nie^2) 
		indices = rand(1:n, nie)
		Bie = zeros(T,nie,nie)
		[Bie[i,i]=1 for i in 1:nie]		
		eem = Elemental_em{T}(nie,indices,Symmetric(Bie))
		return eem
	end 

	function ones_eem(nie :: Int; T=Float64, n=nie^2) 
		indices = rand(1:n, nie)
		Bie = ones(T,nie,nie)		
		eem = Elemental_em{T}(nie,indices,Symmetric(Bie))
		return eem
	end 

	function fixed_ones_eem(i::Int, nie :: Int; T=Float64, mul=5.) 
		indices = [i:(i+nie-1);]
		Bie = ones(T,nie,nie)		
		[Bie[i,i] = mul for i in 1:nie]
		eem = Elemental_em{T}(nie,indices,Symmetric(Bie))
		return eem
	end 

	one_size_bloc(index :: Int; T=Float64) = Elemental_em{T}(1,[index], Symmetric(ones(1,1)))


	import Base.permute!
	permute!(eem :: Elemental_em{T}, p :: Vector{Int}) where T = eem.indices .= p

end 