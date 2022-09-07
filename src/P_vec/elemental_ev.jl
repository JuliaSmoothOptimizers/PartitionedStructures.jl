module ModElemental_ev

using ..Acronyms
using SparseArrays, StatsBase
using ..M_abstract_element_struct, ..M_elt_vec, ..Utils

import Base.==, Base.copy, Base.similar
import Base:-, +, *
import Base.broadcast!

export Elemental_elt_vec
export ones_eev, new_eev, specific_ones_eev
export create_eev, eev_from_sparse_vec, sparse_vec_from_eev

# we assume that the values of vec are associate to indices.
"""
    Elemental_elt_vec{T} <: Elt_vec{T}

Represent an elemental element-vector.
`indices` retains the indices of the elemental variables.
`nie` is the elemental size (`=length(indices)`).
`vec` is the current value of the elemental element vector.
"""
mutable struct Elemental_elt_vec{T} <: Elt_vec{T}
  vec::Vector{T} # length(vec)==nᵢᴱ
  indices::Vector{Int} # length(indices)==nᵢᴱ
  nie::Int
end

@inline (==)(eev1::Elemental_elt_vec{T}, eev2::Elemental_elt_vec{T}) where {T} =
  (get_indices(eev1) == get_indices(eev2)) &&
  (get_vec(eev1) == get_vec(eev2)) &&
  (get_nie(eev1) == get_nie(eev2))
@inline similar(eev::Elemental_elt_vec{T}) where {T} =
  Elemental_elt_vec{T}(Vector{T}(undef, get_nie(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))
@inline copy(eev::Elemental_elt_vec{T}) where {T} =
  Elemental_elt_vec{T}(Vector{T}(get_vec(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))

(-)(eev::Elemental_elt_vec{T}) where {T} = Elemental_elt_vec{T}(Vector{T}(- get_vec(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))
(-)(eev1::Elemental_elt_vec{T}, eev2::Elemental_elt_vec{T})  where {T} = Elemental_elt_vec{T}(Vector{T}(get_vec(eev1) - get_vec(eev2)), Vector{Int}(get_indices(eev1)), get_nie(eev1))
(+)(eev1::Elemental_elt_vec{T}, eev2::Elemental_elt_vec{T})  where {T} = Elemental_elt_vec{T}(Vector{T}(get_vec(eev1) + get_vec(eev2)), Vector{Int}(get_indices(eev1)), get_nie(eev1))
(*)(eev::Elemental_elt_vec{T}, val::Y) where {T, Y} = Elemental_elt_vec{T}(Vector{T}(val .* get_vec(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))
(*)(val::Y, eev::Elemental_elt_vec{T}) where {T, Y} = Elemental_elt_vec{T}(Vector{T}(val .* get_vec(eev)), Vector{Int}(get_indices(eev)), get_nie(eev))

# function broadcast!(f::Function, eev::Elemental_elt_vec{T}, As...) where {T}
#   _As = map(as -> as[get_indices(eev)], As)
#   broadcast!(f, get_vec(eev), _As...)
#   return eev
# end

"""
    eem = new_eev(nᵢ::Int; T=Float64, n=nᵢ^2)

Create an elemental element-vector of size `nie`, with random values and whose the indices are within the range `1:n`.
"""
@inline new_eev(nᵢ::Int; T = Float64, n = nᵢ^2) =
  Elemental_elt_vec(rand(T, nᵢ), sample(1:n, nᵢ, replace = false), nᵢ)

"""
    eem = ones_eev(nᵢ::Int; T=Float64, n=nᵢ^2)

Create an elemental element-vector of size `nie` with values set to `1` and whose the indices are within the range `1:n`.
"""
@inline ones_eev(nᵢ::Int; T = Float64, n = nᵢ^2) =
  Elemental_elt_vec(ones(T, nᵢ), sample(1:n, nᵢ, replace = false), nᵢ)

"""
    eem = specific_ones_eev(nie::Int, index::Int; T=Float64, mul::Float64=1.)

Create an elemental element-vector of size `nie`, of random values multiplied by `mul` and whose indices are in range `index:index+nie`.
"""
@inline specific_ones_eev(nie::Int, index::Int; T = Float64, mul::Float64 = 1.0) =
  Elemental_elt_vec((xi -> mul * xi).(rand(T, nie)), [index:(index + nie - 1);], nie)

"""
    eem = eev_from_sparse_vec(sparsevec::SparseVector{T,Y})

Define an elemental element-vector from a `sparsevec::SparseVector`.
The indices and the values are define with `findnz(sparse_vec)`.
"""
function eev_from_sparse_vec(v::SparseVector{T, Y}) where {T, Y}
  (indices, vec) = findnz(v)
  nie = length(indices)
  eev = Elemental_elt_vec{T}(Vector{T}(vec), Vector{Int}(indices), nie)
  return eev
end

"""
    sp_vec = sparse_vec_from_eev(eev::Elemental_elt_vec{T}; n::Int=maximum(get_indices(eev))) where T

Create a `SparseVector` from the element element-vector `eev`.
"""
sparse_vec_from_eev(eev::Elemental_elt_vec{T}; n::Int = maximum(get_indices(eev))) where {T} =
  sparsevec(get_indices(eev), get_vec(eev), n)

"""
    eev = create_eev(elt_var::Vector{Int}; type=Float64)

Create a random $(_eev) from the elemental variables `elt_var`.
`eev` is set to random values.
"""
function create_eev(elt_var::Vector{Int}; type = Float64)
  nie = length(elt_var)
  eev_value = rand(type, nie)
  eev = Elemental_elt_vec{type}(eev_value, elt_var, nie)
  return eev
end

end
