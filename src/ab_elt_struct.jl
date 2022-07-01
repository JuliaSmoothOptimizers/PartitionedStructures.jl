module M_abstract_element_struct

using ..Utils

export Element_struct
export get_indices, get_nie
export set_indices!, set_nie!
export max_indices, min_indices

""" Supertype of every element-structure, ex : `Elemental_elt_vec`, `Elemental_em`, `Elemental_elo_bfgs`, `Internal_elt_vec`..."""
abstract type Element_struct{T} end

"""
    indices = get_indices(elt::T) where T <: Element_struct
    indice = get_indices(elt::T, i::Int) where T <: Element_struct

Every element-structure is based on a variable subset of a partitioned-structure.
`get_indices(elt)` retrieves the variable set of an element `elt`.
`get_indices(elt, i)` retrieves the `i`-th variable associated to `elt`.
"""
@inline get_indices(elt::T) where {T <: Element_struct} = elt.indices
@inline get_indices(elt::T, i::Int) where {T <: Element_struct} = elt.indices[i]

"""
    nie = get_nie(elt::T) where T <: Element_struct

Return the elemental size of the element `elt.nie`.
"""
@inline get_nie(elt::T) where {T <: Element_struct} = elt.nie

"""
    set_indices!(elt::T, indices::Vector{Int}) where T <: Element_struct

Set the indices of the element `elt.indices` to `indices`.
"""
@inline set_indices!(elt::T, indices::Vector{Int}) where {T <: Element_struct} =
  elt.indices = indices

"""
    set_nie!(elt::T, nie::Int) where T <: Element_struct

Set the element size of `elt` to `nie`.
"""
@inline set_nie!(elt::T, nie::Int) where {T <: Element_struct} = elt.nie = nie

# get the max/min index of variable from the {indiceᵢ}ᵢ
Utils.max_indices(elt_set::Vector{T}) where {T <: Element_struct} =
  isempty(elt_set) ? 0 : maximum(maximum.(get_indices.(elt_set)))
Utils.min_indices(elt_set::Vector{T}) where {T <: Element_struct} =
  isempty(elt_set) ? 0 : minimum(minimum.(get_indices.(elt_set)))

end
