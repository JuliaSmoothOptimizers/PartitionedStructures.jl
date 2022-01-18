module ModElemental_elm
	using LinearOperators

	using PartitionedStructures.M_elt_mat

	# mutable struct Elemental_elm{Y} where Y <: AbstractLinearOperator{T} where T <: Number <: Elt_mat{T}
	mutable struct Elemental_elm{T} <: Elt_mat{T}
		nie :: Int # nᵢᴱ
		indices :: Vector{Int} # size nᵢᴱ
		Bie <: AbstractLinearOperator{T}  # LinearOperator
		t :: T
	end

	@inline get_Bie(eem :: Elemental_elm{T}) where T = eem.Bie
	@inline M_elt_mat.get_mat(eem :: Elemental_elm{T}) where T = get_Bie(eem)
		
	@inline (==)(eem1 :: Elemental_elm{T}, eem2 :: Elemental_elm{T}) where T = (get_nie(eem1)== get_nie(eem2)) && (get_Bie(eem1)== get_Bie(eem2)) && (get_indices(eem1)== get_indices(eem2))
	@inline copy(eem :: Elemental_elm{T}) where T = Elemental_elm{T}(copy(get_nie(eem)), copy(get_indices(eem)), copy(get_Bie(eem)))
	@inline similar(eem :: Elemental_elm{T}) where T = Elemental_elm{T}(copy(get_nie(eem)), copy(get_indices(eem)), similar(get_Bie(eem)))

end 