module ModElemental_elom_sr1

  using LinearOperators
  using ..M_elt_mat, ..M_abstract_element_struct

  import Base.==, Base.copy, Base.similar

  export Elemental_elom_sr1
  export init_eelom_LSR1, LSR1_eelom_rand, LSR1_eelom
	export reset_eelom_sr1!

  "Type that represents elemental element linear operator LSR1"
  mutable struct Elemental_elom_sr1{T} <: Elt_mat{T}
    nie :: Int # nᵢᴱ
    indices :: Vector{Int} # size nᵢᴱ
    Bie :: LinearOperators.LSR1Operator{T}
  end

  @inline (==)(eelom1 :: Elemental_elom_sr1{T}, eelom2 :: Elemental_elom_sr1{T}) where T = (get_nie(eelom1)== get_nie(eelom2)) && begin v=rand(get_nie(eelom1)); (get_Bie(eelom1) *v == get_Bie(eelom2)*v) end && (get_indices(eelom1)== get_indices(eelom2))
  @inline copy(eelom :: Elemental_elom_sr1{T}) where T = Elemental_elom_sr1{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), deepcopy(get_Bie(eelom)))
  @inline similar(eelom :: Elemental_elom_sr1{T}) where T = Elemental_elom_sr1{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), similar(get_Bie(eelom)))

  """
      init_eelom_LSR1(indices; T=T)
  Define a `Elemental_elom_sr1` of type `Elemental_elom_sr1` based from the vector `indices`.
  """
  function init_eelom_LSR1(elt_var :: Vector{Int}; T=Float64)
    nie = length(elt_var)
    Bie = LinearOperators.LSR1Operator(T, nie)
    elom = Elemental_elom_sr1{T}(nie, elt_var, Bie)
    return elom
  end 

  """
      LSR1_eelom_rand(nie, T=T, n=n)
  Create a `Elemental_elom_sr1` of type `T` with `nie` random indices within the range `1:n`.
  """
  function LSR1_eelom_rand(nie :: Int; T=Float64, n=nie^2)
    indices = rand(1:n, nie) 		
    Bie = LinearOperators.LSR1Operator(T, nie)
    eelom = Elemental_elom_sr1{T}(nie, indices, Bie)
    return eelom
  end 

  """
      LSR1_eelom(nie, T=T, index=index)
  Create a `Elemental_elom_sr1` of type `T` of size `nie`, the indices are in the range `index:index+nie-1`.
  """
  function LSR1_eelom(nie :: Int; T=Float64, index=1)
    indices = [index:1:index+nie-1;]
    Bie = LinearOperators.LSR1Operator(T, nie)
    eelom = Elemental_elom_sr1{T}(nie, indices, Bie)
    return eelom
  end 

	"""
			reset_eelom_sr1!(eelom)
	
	Reset the LSR1 linear operator of the elemental element linear operator matrix.
	"""
	function reset_eelom_sr1!(eelom::Elemental_elom_sr1{T}) where T <: Number
		eelom.Bie = LinearOperators.LSR1Operator(T, eelom.nie)
	end

end 