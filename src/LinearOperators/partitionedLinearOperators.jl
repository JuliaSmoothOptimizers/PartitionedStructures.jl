module PartitionedLinearOperators

	using LinearOperators
	using ..M_part_mat

	abstract type AbstractPartitionedLinearOperators{T} <: AbstractLinearOperators{T}

	# Copier les champs de LinearOperator.
	# Attente de la réponse de Dominique

	mutable struct PBFGSLinearOperator{T} <: AbstractPartitionedLinearOperators{T}

	end

	mutable struct PLBFGSLinearOperator{T} <: AbstractPartitionedLinearOperators{T}

	end

end 
