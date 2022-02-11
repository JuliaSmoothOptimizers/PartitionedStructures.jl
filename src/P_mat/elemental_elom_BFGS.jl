module ModElemental_elom_bfgs
	using LinearOperators
	using ..M_elt_mat, ..M_abstract_element_struct

	import Base.==, Base.copy, Base.similar

	export get_Bie
	export Elemental_elom_bfgs, LBFGS_eelom_rand, LBFGS_eelom, init_eelom_lbfgs

	# mutable struct Elemental_elom_bfgs{Y} where Y <: AbstractLinearOperator{T} where T <: Number <: Elt_mat{T}
	mutable struct Elemental_elom_bfgs{T} <: Elt_mat{T}
		nie :: Int # nᵢᴱ
		indices :: Vector{Int} # size nᵢᴱ
		Bie :: LinearOperators.LBFGSOperator{T}  # LinearOperator
	end

	@inline (==)(eelom1 :: Elemental_elom_bfgs{T}, eelom2 :: Elemental_elom_bfgs{T}) where T = (get_nie(eelom1)== get_nie(eelom2)) && begin v=rand(get_nie(eelom1)); (get_Bie(eelom1) *v == get_Bie(eelom2)*v) end && (get_indices(eelom1)== get_indices(eelom2))
	@inline copy(eelom :: Elemental_elom_bfgs{T}) where T = Elemental_elom_bfgs{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), deepcopy(get_Bie(eelom)))
	@inline similar(eelom :: Elemental_elom_bfgs{T}) where T = Elemental_elom_bfgs{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), similar(get_Bie(eelom)))

	function init_eelom_lbfgs(elt_var::Vector{Int}; type=Float64)
		nie = length(elt_var)
		Bie = LinearOperators.LBFGSOperator(type, nie)
		elom = Elemental_elom_bfgs{type}(nie, elt_var, Bie)
		return elom
	end 

	function LBFGS_eelom_rand(nie :: Int; T=Float64, n=nie^2)
		indices = rand(1:n, nie) 		
		Bie = LinearOperators.LBFGSOperator(T, nie)
		eelom = Elemental_elom_bfgs{T}(nie,indices,Bie)
		return eelom
	end 

	function LBFGS_eelom(nie :: Int; T=Float64, index=1)
		indices = [index:1:index+nie-1;]
		Bie = LinearOperators.LBFGSOperator(T, nie)
		eelom = Elemental_elom_bfgs{T}(nie,indices,Bie)
		return eelom
	end 

end 