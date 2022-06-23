module ModElemental_em

using LinearAlgebra
using ..M_abstract_element_struct, ..M_elt_mat

import Base.==, Base.copy, Base.permute!, Base.similar

export Elemental_em
export identity_eem, create_id_eem, fixed_ones_eem, ones_eem, one_size_bloc

"""
    Elemental_em{T} <: DenseEltMat{T}

Type that represents an elemental element matrix
"""
mutable struct Elemental_em{T} <: DenseEltMat{T}
  nie :: Int # nᵢᴱ
  indices :: Vector{Int} # size nᵢᴱ
  Bie :: Symmetric{T, Matrix{T}} # size nᵢᴱ × nᵢᴱ
  counter :: Counter_elt_mat
end

@inline (==)(eem1 :: Elemental_em{T}, eem2 :: Elemental_em{T}) where T = (get_nie(eem1)== get_nie(eem2)) && (get_Bie(eem1)== get_Bie(eem2)) && (get_indices(eem1)== get_indices(eem2))
@inline copy(eem :: Elemental_em{T}) where T = Elemental_em{T}(copy(get_nie(eem)), copy(get_indices(eem)), copy(get_Bie(eem)), copy(get_cem(eem)))
@inline similar(eem :: Elemental_em{T}) where T = Elemental_em{T}(copy(get_nie(eem)), copy(get_indices(eem)), similar(get_Bie(eem)), copy(get_cem(eem)))

"""
    create_id_eem(elt_var; T=T)

Creates a `nie` identity elemental element matrix of type `T` at the indices `elt_var`.
"""
function create_id_eem(elt_var :: Vector{Int}; T=Float64)
  nie = length(elt_var)
  Bie = zeros(T, nie, nie)
  [Bie[i, i]=1 for i in 1:nie]  
  counter = Counter_elt_mat()
  eem = Elemental_em{T}(nie, elt_var, Symmetric(Bie), counter)
  return eem
end

"""
    identity_eem(nie; T=T, n=n)

Creates a `nie` identity elemental element matrix of type `T` from `nie` random indices in the range `1:n`.
"""
function identity_eem(nie :: Int; T=Float64, n=nie^2) 
  indices = rand(1:n, nie)
  Bie = zeros(T, nie, nie)
  [Bie[i, i]=1 for i in 1:nie]		
  counter = Counter_elt_mat()
  eem = Elemental_em{T}(nie, indices, Symmetric(Bie), counter)
  return eem
end 

"""
    ones_eem(nie; T=T, n=n)

Creates a `nie` ones elemental element matrix of type `T` from `nie` random indices in the range `1:n`.
"""
function ones_eem(nie :: Int; T=Float64, n=nie^2) 
  indices = rand(1:n, nie)
  Bie = ones(T, nie, nie)		
  counter = Counter_elt_mat()
  eem = Elemental_em{T}(nie, indices, Symmetric(Bie), counter)
  return eem
end 

"""
    fixed_ones_eem(index, nie; type=T, mul=mul)

Creates a `nie` elemental element matrix of type `T` at indices `index:index+nie-1`.
All the components of the element matrix are set to `1` except the diagonal terms that are set to `mul`.
This method is used to define diagonal dominant element matrix.
"""
function fixed_ones_eem(i :: Int, nie :: Int; T=Float64, mul=5.) 
  indices = [i:(i+nie-1);]
  Bie = ones(T, nie, nie)		
  [Bie[i, i] = mul for i in 1:nie]
  counter = Counter_elt_mat()
  eem = Elemental_em{T}(nie, indices, Symmetric(Bie), counter)
  return eem
end 

"""
    one_size_bloc(index)

Defines an elemental element matrix of type `T` of size one at `index`.
"""
one_size_bloc(index :: Int; T=Float64) = Elemental_em{T}(1, [index], Symmetric(ones(1, 1)), Counter_elt_mat())

"""
    permute!(eem, p)

Set the indices of the element variables of `eem` to `p`.
Must be use with caution.
"""
permute!(eem :: Elemental_em{T}, p :: Vector{Int}) where T = eem.indices .= p

end 