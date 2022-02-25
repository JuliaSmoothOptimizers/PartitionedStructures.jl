module ModElemental_em

  using LinearAlgebra
  using ..M_abstract_element_struct, ..M_elt_mat

  import Base.==, Base.copy, Base.permute!, Base.similar

  export Elemental_em
  export identity_eem, create_id_eem, fixed_ones_eem, ones_eem, one_size_bloc

  mutable struct Elemental_em{T} <: Elt_mat{T}
    nie :: Int # nᵢᴱ
    indices :: Vector{Int} # size nᵢᴱ
    Bie :: Symmetric{T, Matrix{T}} # size nᵢᴱ × nᵢᴱ
  end

  @inline (==)(eem1 :: Elemental_em{T}, eem2 :: Elemental_em{T}) where T = (get_nie(eem1)== get_nie(eem2)) && (get_Bie(eem1)== get_Bie(eem2)) && (get_indices(eem1)== get_indices(eem2))
  @inline copy(eem :: Elemental_em{T}) where T = Elemental_em{T}(copy(get_nie(eem)), copy(get_indices(eem)), copy(get_Bie(eem)))
  @inline similar(eem :: Elemental_em{T}) where T = Elemental_em{T}(copy(get_nie(eem)), copy(get_indices(eem)), similar(get_Bie(eem)))

  """
      create_id_eem(indices; T=T)
  Create an `nie` identity elemental element matrix of type `T` at the `indices`.
  """
  function create_id_eem(elt_var :: Vector{Int}; T=Float64)
    nie = length(elt_var)
    Bie = zeros(T, nie, nie)
    [Bie[i, i]=1 for i in 1:nie]  
    eem = Elemental_em{T}(nie, elt_var, Symmetric(Bie))
    return eem
  end

  """
      identity_eem(nie; T=T, n=n)
  Create an `nie` identity elemental element matrix of type `T` at random indices in the range `1:n`.
  """
  function identity_eem(nie :: Int; T=Float64, n=nie^2) 
    indices = rand(1:n, nie)
    Bie = zeros(T, nie, nie)
    [Bie[i, i]=1 for i in 1:nie]		
    eem = Elemental_em{T}(nie, indices, Symmetric(Bie))
    return eem
  end 

  """
      ones_eem(nie; T=T, n=n)
  Create an `nie` ones elemental element matrix of type `T` at random indices in the range `1:n`.
  """
  function ones_eem(nie :: Int; T=Float64, n=nie^2) 
    indices = rand(1:n, nie)
    Bie = ones(T, nie, nie)		
    eem = Elemental_em{T}(nie, indices, Symmetric(Bie))
    return eem
  end 

  """
      fixed_ones_eem(index, nie; type=T, mul=mul)
  Create an `nie` elemental element matrix of type `T` at indices `index:index+nie-1`.
  All element have the value `1` except the diagonal that have the value `mul`, it is use to define diagonal dominant matrix.
  """
  function fixed_ones_eem(i :: Int, nie :: Int; T=Float64, mul=5.) 
    indices = [i:(i+nie-1);]
    Bie = ones(T, nie, nie)		
    [Bie[i, i] = mul for i in 1:nie]
    eem = Elemental_em{T}(nie, indices, Symmetric(Bie))
    return eem
  end 

  """
      one_size_bloc(i)
  Define a elemental element matrix of type `T` of size one at the index `i`.
  """
  one_size_bloc(index :: Int; T=Float64) = Elemental_em{T}(1, [index], Symmetric(ones(1, 1)))

  permute!(eem :: Elemental_em{T}, p :: Vector{Int}) where T = eem.indices .= p

end 