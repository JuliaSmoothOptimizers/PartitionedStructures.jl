module ModElemental_elom_bfgs
  
  using LinearOperators
  using ..M_abstract_element_struct, ..M_elt_mat

  import Base.==, Base.copy, Base.similar

  export Elemental_elom_bfgs
  export init_eelom_LBFGS, LBFGS_eelom, LBFGS_eelom_rand

  "Type that represents an elemental element linear operator LBFGS"
  mutable struct Elemental_elom_bfgs{T} <: Elt_mat{T}
    nie :: Int # nᵢᴱ
    indices :: Vector{Int} # size nᵢᴱ
    Bie :: LinearOperators.LBFGSOperator{T}
  end

  @inline (==)(eelom1 :: Elemental_elom_bfgs{T}, eelom2 :: Elemental_elom_bfgs{T}) where T = (get_nie(eelom1)== get_nie(eelom2)) && begin v=rand(get_nie(eelom1)); (get_Bie(eelom1) *v == get_Bie(eelom2)*v) end && (get_indices(eelom1)== get_indices(eelom2))
  @inline copy(eelom :: Elemental_elom_bfgs{T}) where T = Elemental_elom_bfgs{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), deepcopy(get_Bie(eelom)))
  @inline similar(eelom :: Elemental_elom_bfgs{T}) where T = Elemental_elom_bfgs{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), similar(get_Bie(eelom)))

  """
      init_eelom_LBFGS(indices; T=T)
  Define a `Elemental_elom_bfgs` of type `Elemental_elom_sr1` based from the vector `indices`.
  """	
  function init_eelom_LBFGS(elt_var :: Vector{Int}; T=Float64)
    nie = length(elt_var)
    Bie = LinearOperators.LBFGSOperator(T, nie)
    elom = Elemental_elom_bfgs{T}(nie, elt_var, Bie)
    return elom
  end 

  """
      LBFGS_eelom_rand(nie, T=T, n=n)
  Create a `Elemental_elom_bfgs` of type `T` with `nie` random indices within the range `1:n`.
  """
  function LBFGS_eelom_rand(nie :: Int; T=Float64, n=nie^2)
    indices = rand(1:n, nie) 		
    Bie = LinearOperators.LBFGSOperator(T, nie)
    eelom = Elemental_elom_bfgs{T}(nie, indices, Bie)
    return eelom
  end 

  """
      LBFGS_eelom(nie, T=T, index=index)
  Create a `Elemental_elom_bfgs` of type `T` of size `nie`, the indices are in the range `index:index+nie-1`.
  """
  function LBFGS_eelom(nie :: Int; T=Float64, index=1)
    indices = [index:1:index+nie-1;]
    Bie = LinearOperators.LBFGSOperator(T, nie)
    eelom = Elemental_elom_bfgs{T}(nie, indices, Bie)
    return eelom
  end 

end 