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
* `Bie::Symmetric{T, Matrix{T}}`: the elemental matrix;
* `counter`: counts how many update the elemental matrix goes through from its allocation;
* `convex`: if `convex==true`, then `Elemental_em` default update is BFGS otherwise it is SR1;
* `linear`: if `linear==true`, then the element matrix contribution is null;
* `_Bsr`: a vector used during quasi-Newton update of the elemental matrix.
"""
mutable struct Elemental_em{T} <: DenseEltMat{T}
  nie::Int # nᵢᴱ
  indices::Vector{Int} # size nᵢᴱ
  Bie::Symmetric{T, Matrix{T}} # size nᵢᴱ × nᵢᴱ
  counter::Counter_elt_mat
  convex::Bool
  linear::Bool
  _Bsr::Vector{T} # size nᵢᴱ
end

@inline (==)(eem1::Elemental_em{T}, eem2::Elemental_em{T}) where {T} =
  (get_nie(eem1) == get_nie(eem2)) &&
  (get_Bie(eem1) == get_Bie(eem2)) &&
  (get_indices(eem1) == get_indices(eem2)) &&
  (get_convex(eem1) == get_convex(eem2)) &&
  (get_linear(eem1) == get_linear(eem2))
@inline copy(eem::Elemental_em{T}) where {T} = Elemental_em{T}(
  copy(get_nie(eem)),
  copy(get_indices(eem)),
  copy(get_Bie(eem)),
  Counter_elt_mat(),
  copy(get_convex(eem)),
  copy(get_linear(eem)),
  copy(get_Bsr(eem)),
)
@inline similar(eem::Elemental_em{T}) where {T} = Elemental_em{T}(
  copy(get_nie(eem)),
  copy(get_indices(eem)),
  similar(get_Bie(eem)),
  Counter_elt_mat(),
  copy(get_convex(eem)),
  copy(get_linear(eem)),
  similar(get_Bsr(eem)),
)

"""
    eem = create_id_eem(elt_var::Vector{Int}; T=Float64)

Create a `nie` identity elemental element-matrix of type `T` based on the vector of the elemental variables `elt_var`.
"""
function create_id_eem(elt_var::Vector{Int}; T = Float64, convex = false, linear = false)
  nie = length(elt_var)
  _nie = (!linear) * nie
  Bie = zeros(T, _nie, _nie)
  [Bie[i, i] = 1 for i = 1:_nie]
  counter = Counter_elt_mat()
  _Bsr = Vector{T}(undef, _nie)
  eem = Elemental_em{T}(nie, elt_var, Symmetric(Bie), counter, convex, linear, _Bsr)
  return eem
end

"""
    eem = identity_eem(nie::Int; T=Float64, n=nie^2)

Return a `nie` identity elemental element-matrix of type `T` from `nie` random indices in the range `1:n`.
"""
function identity_eem(nie::Int; T = Float64, n = nie^2, convex = false, linear = false)
  indices = rand(1:n, nie)
  _nie = (!linear) * nie
  Bie = zeros(T, _nie, _nie)
  [Bie[i, i] = 1 for i = 1:_nie]
  counter = Counter_elt_mat()
  _Bsr = Vector{T}(undef, _nie)
  eem = Elemental_em{T}(nie, indices, Symmetric(Bie), counter, convex, linear, _Bsr)
  return eem
end

"""
    eem = ones_eem(nie::Int; T=Float64, n=nie^2)

Return a `nie` ones elemental element-matrix of type `T` from `nie` random indices in the range `1:n`.
"""
function ones_eem(nie::Int; T = Float64, n = nie^2, convex = false, linear = false)
  indices = rand(1:n, nie)
  _nie = (!linear) * nie
  Bie = ones(T, _nie, _nie)
  counter = Counter_elt_mat()
  _Bsr = Vector{T}(undef, _nie)
  eem = Elemental_em{T}(nie, indices, Symmetric(Bie), counter, convex, linear, _Bsr)
  return eem
end

"""
    eem = fixed_ones_eem(i::Int, nie::Int; T=Float64, mul=5.)

Create a `nie` elemental element-matrix of type `T` at indices `index:index+nie-1`.
All the components of the element-matrix are set to `1` except the diagonal terms that are set to `mul`.
This method is used to define diagonal dominant element-matrix.
"""
function fixed_ones_eem(i::Int, nie::Int; T = Float64, mul = 5.0, convex = false, linear = false)
  indices = [i:(i + nie - 1);]
  _nie = (!linear) * nie
  Bie = ones(T, _nie, _nie)
  [Bie[i, i] = mul for i = 1:_nie]
  counter = Counter_elt_mat()
  _Bsr = Vector{T}(undef, _nie)
  eem = Elemental_em{T}(nie, indices, Symmetric(Bie), counter, convex, linear, _Bsr)
  return eem
end

"""
    eem = one_size_bloc(index::Int; T=Float64)

Return an elemental element-matrix of type `T` of size one at `index`.
"""
function one_size_bloc(index::Int; nie = 1, T = Float64, convex = false, linear = false)
  _nie = (!linear) * nie
  Elemental_em{T}(
    nie,
    [index],
    Symmetric(ones(_nie, _nie)),
    Counter_elt_mat(),
    convex,
    linear,
    Vector{T}(undef, _nie),
  )
end

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
