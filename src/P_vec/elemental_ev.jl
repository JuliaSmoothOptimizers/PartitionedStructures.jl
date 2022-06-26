module ModElemental_ev

using SparseArrays, StatsBase
using ..M_abstract_element_struct, ..M_elt_vec, ..Utils

import Base.==, Base.copy, Base.similar

export Elemental_elt_vec
export ones_eev, new_eev, specific_ones_eev
export create_eev, eev_from_sparse_vec, sparse_vec_from_eev

# we assume that the values of vec are associate to indices.
"""
    Elemental_elt_vec{T} <: Elt_vec{T}

Type that represents an elemental element-vector.
"""
mutable struct Elemental_elt_vec{T} <: Elt_vec{T}
  vec :: Vector{T} # length(vec) == nᵢᴱ
  indices :: Vector{Int} # length(indices) == nᵢᴱ
  nie :: Int
end

@inline (==)(eev1 :: Elemental_elt_vec{T}, eev2 :: Elemental_elt_vec{T}) where T = (get_indices(eev1) == get_indices(eev2)) && (get_vec(eev1) == get_vec(eev2)) && (get_nie(eev1) == get_nie(eev2))
@inline similar(eev :: Elemental_elt_vec{T}) where T = Elemental_elt_vec{T}(Vector{T}(undef,get_nie(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))
@inline copy(eev :: Elemental_elt_vec{T}) where T = Elemental_elt_vec{T}(Vector{T}(get_vec(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))

"""
    eem = new_eev(nie; T, n)

Create an elemental element-vector of size `nie`, with random values and whose the indices are within the range `1:n`.
"""
@inline new_eev(nᵢ :: Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(rand(T,nᵢ), sample(1:n, nᵢ, replace = false), nᵢ)

"""
    eem = ones_eev(nie; T, n)

Create an elemental element-vector of size `nie` with values set to `1` and whose the indices are within the range `1:n`.
"""
@inline ones_eev(nᵢ :: Int; T=Float64, n=nᵢ^2) = Elemental_elt_vec(ones(T,nᵢ), sample(1:n, nᵢ, replace = false), nᵢ)

"""
    eem = specific_ones_eev(nie, index; T, mul)

Create an elemental element-vector of size `nie`, of random values multiplied by `mul` and whose indices are in range `index:index+nie`.
"""
@inline specific_ones_eev(nie :: Int, index :: Int; T=Float64, mul :: Float64=1.) = Elemental_elt_vec((xi -> mul*xi).(rand(T, nie)), [index:index+nie-1;], nie)

"""
    eem = eev_from_sparse_vec(sparse_vec)

Define an elemental element-vector from a `SparseVector` `sparsevec`.
The indices and the values are define with `findnz(sparse_vec)`.
"""
function eev_from_sparse_vec(v :: SparseVector{T,Y}) where {T, Y}
  (indices, vec) = findnz(v)
  nie = length(indices)
  eev = Elemental_elt_vec{T}(Vector{T}(vec), Vector{Int}(indices), nie)
  return eev
end

"""
    sp_vec = sparse_vec_from_eev(eev)

Create a `SparseVector` from the element element-vector `eev`.
"""
sparse_vec_from_eev(eev :: Elemental_elt_vec{T}; n :: Int=maximum(get_indices(eev))) where T = sparsevec(get_indices(eev), get_vec(eev), n)

"""
    eem = create_eev(elt_var)

Create a random elemental element-vector `eem` from the elemental variables `elt_var`.
`eem` is set to random values.
"""
function create_eev(elt_var :: Vector{Int}; type=Float64)
  nie = length(elt_var)
  eev_value = rand(type, nie)
  eev = Elemental_elt_vec{type}(eev_value, elt_var, nie)
  return eev
end

end