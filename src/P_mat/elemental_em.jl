module ModElemental_em

using ..Acronyms
using LinearAlgebra
using ..M_abstract_element_struct, ..M_elt_mat

import Base.==, Base.copy, Base.permute!, Base.similar

export Elemental_em
export identity_eem, create_id_eem, fixed_ones_eem, ones_eem, one_size_bloc

"""
    Elemental_em{T} <: DenseEltMat{T}

Represent an elemental element-matrix.
It has fields:

* `indices`: indices of elemental variables;
* `nie`: elemental size (`=length(indices)`);
* `Bie`: the elemental matrix`::Symmetric{T, Matrix{T}}`;
* `counter`: counts how many update the elemental matrix goes through from its allocation;
* `convex`: if `Elemental_em` is by default update with BFGS or SR1.
"""
mutable struct Elemental_em{T} <: DenseEltMat{T}
  nie::Int # nᵢᴱ
  indices::Vector{Int} # size nᵢᴱ
  Bie::Symmetric{T, Matrix{T}} # size nᵢᴱ × nᵢᴱ
  counter::Counter_elt_mat
  convex::Bool
end

@inline (==)(eem1::Elemental_em{T}, eem2::Elemental_em{T}) where {T} =
  (get_nie(eem1) == get_nie(eem2)) &&
  (get_Bie(eem1) == get_Bie(eem2)) &&
  (get_indices(eem1) == get_indices(eem2))
@inline copy(eem::Elemental_em{T}) where {T} = Elemental_em{T}(
  copy(get_nie(eem)),
  copy(get_indices(eem)),
  copy(get_Bie(eem)),
  copy(get_cem(eem)),
  copy(get_convex(eem))
)
@inline similar(eem::Elemental_em{T}) where {T} = Elemental_em{T}(
  copy(get_nie(eem)),
  copy(get_indices(eem)),
  similar(get_Bie(eem)),
  copy(get_cem(eem)),
  copy(get_convex(eem))
)

"""
    eem = create_id_eem(elt_var::Vector{Int}; T=Float64)

Create a `nie` identity elemental element-matrix of type `T` based on the vector of the elemental variables `elt_var`.
"""
function create_id_eem(elt_var::Vector{Int}; T = Float64, bool=false)
  nie = length(elt_var)
  Bie = zeros(T, nie, nie)
  [Bie[i, i] = 1 for i = 1:nie]
  counter = Counter_elt_mat()
  eem = Elemental_em{T}(nie, elt_var, Symmetric(Bie), counter, bool)
  return eem
end

"""
    eem = identity_eem(nie::Int; T=Float64, n=nie^2)

Return a `nie` identity elemental element-matrix of type `T` from `nie` random indices in the range `1:n`.
"""
function identity_eem(nie::Int; T = Float64, n = nie^2, bool=false)
  indices = rand(1:n, nie)
  Bie = zeros(T, nie, nie)
  [Bie[i, i] = 1 for i = 1:nie]
  counter = Counter_elt_mat()
  eem = Elemental_em{T}(nie, indices, Symmetric(Bie), counter, bool)
  return eem
end

"""
    eem = ones_eem(nie::Int; T=Float64, n=nie^2)

Return a `nie` ones elemental element-matrix of type `T` from `nie` random indices in the range `1:n`.
"""
function ones_eem(nie::Int; T = Float64, n = nie^2, bool=false)
  indices = rand(1:n, nie)
  Bie = ones(T, nie, nie)
  counter = Counter_elt_mat()
  eem = Elemental_em{T}(nie, indices, Symmetric(Bie), counter, bool)
  return eem
end

"""
    eem = fixed_ones_eem(i::Int, nie::Int; T=Float64, mul=5.)

Create a `nie` elemental element-matrix of type `T` at indices `index:index+nie-1`.
All the components of the element-matrix are set to `1` except the diagonal terms that are set to `mul`.
This method is used to define diagonal dominant element-matrix.
"""
function fixed_ones_eem(i::Int, nie::Int; T = Float64, mul = 5.0, bool=false)
  indices = [i:(i + nie - 1);]
  Bie = ones(T, nie, nie)
  [Bie[i, i] = mul for i = 1:nie]
  counter = Counter_elt_mat()
  eem = Elemental_em{T}(nie, indices, Symmetric(Bie), counter, bool)
  return eem
end

"""
    eem = one_size_bloc(index::Int; T=Float64)

Return an elemental element-matrix of type `T` of size one at `index`.
"""
one_size_bloc(index::Int; T = Float64) =
  Elemental_em{T}(1, [index], Symmetric(ones(1, 1)), Counter_elt_mat())

"""
    permute!(eem::Elemental_em{T}, p::Vector{Int}) where T

Set the indices of the element variables of `eem` to `p`.
Must be use with caution.
"""
function permute!(eem::Elemental_em{T}, p::Vector{Int}) where {T}
  eem.indices .= p
  return eem
end

end
