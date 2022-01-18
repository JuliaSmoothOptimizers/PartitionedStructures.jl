module ModElemental_plm

using ModElemental_elm


mutable struct Elemental_plm{T} <: Part_mat{T}
	N :: Int
	n :: Int
	eem_set :: Vector{Elemental_elm{T}}
	spm :: SparseMatrixCSC{T,Int}
	L :: SparseMatrixCSC{T,Int}
	component_list :: Vector{Vector{Int}}
	permutation :: Vector{Int} # n-size vector 
end


end 