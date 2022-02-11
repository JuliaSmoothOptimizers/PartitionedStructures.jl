module ModElemental_elom_sr1
	using LinearOperators

	using ..M_elt_mat, ..M_abstract_element_struct

	import Base.==, Base.copy, Base.similar

	# mutable struct Elemental_elom_sr1{Y} where Y <: AbstractLinearOperator{T} where T <: Number <: Elt_mat{T}
	mutable struct Elemental_elom_sr1{T} <: Elt_mat{T}
		nie :: Int # nᵢᴱ
		indices :: Vector{Int} # size nᵢᴱ
		Bie :: LinearOperators.LSR1Operator{T}  # LinearOperator
	end


	@inline (==)(eelom1 :: Elemental_elom_sr1{T}, eelom2 :: Elemental_elom_sr1{T}) where T = (get_nie(eelom1)== get_nie(eelom2)) && begin v=rand(get_nie(eelom1)); (get_Bie(eelom1) *v == get_Bie(eelom2)*v) end && (get_indices(eelom1)== get_indices(eelom2))
	@inline copy(eelom :: Elemental_elom_sr1{T}) where T = Elemental_elom_sr1{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), deepcopy(get_Bie(eelom)))
	@inline similar(eelom :: Elemental_elom_sr1{T}) where T = Elemental_elom_sr1{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), similar(get_Bie(eelom)))


	function LSR1_eelom_rand(nie :: Int; T=Float64, n=nie^2)
		indices = rand(1:n, nie) 		
		Bie = LinearOperators.LSR1Operator(T, nie)
		eelom = Elemental_elom_sr1{T}(nie,indices,Bie)
		return eelom
	end 

	function LSR1_eelom(nie :: Int; T=Float64, index=1)
		indices = [index:1:index+nie-1;]
		Bie = LinearOperators.LSR1Operator(T, nie)
		eelom = Elemental_elom_sr1{T}(nie,indices,Bie)
		return eelom
	end 

	export get_Bie
	export Elemental_elom_sr1, LSR1_eelom_rand, LSR1_eelom

end 