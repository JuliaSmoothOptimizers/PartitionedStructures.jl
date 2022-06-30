module M_elt_vec

using ..Acronyms
using ..M_abstract_element_struct

export Elt_vec
export get_vec, set_vec!
export set_add_vec!, set_minus_vec!

"""Supertype of element-vectors."""
abstract type Elt_vec{T} <: Element_struct{T} end

"""
    vec = get_vec(ev::T) where T <: Elt_vec
    vec_i = get_vec(ev::T, i::Int) where T <: Elt_vec

Return the vector `ev.vec` or `ev.vec[i]` from an element-vector.
"""
@inline get_vec(ev::T) where {T <: Elt_vec} = ev.vec
@inline get_vec(ev::T, i::Int) where {T <: Elt_vec} = ev.vec[i]

"""
    set_vec!(ev::T, vec::Vector{Y}) where {Y <: Number, T <: Elt_vec{Y}}
    set_vec!(ev::T, i::Int, val::Y) where {Y <: Number, T <: Elt_vec{Y}}

Set `ev.vec` to `vec` or `ev.vec[i] = val` of the element-vector `ev`.
"""
@inline set_vec!(ev::T, vec::Vector{Y}) where {Y <: Number, T <: Elt_vec{Y}} = ev.vec .= vec
@inline set_vec!(ev::T, i::Int, val::Y) where {Y <: Number, T <: Elt_vec{Y}} = ev.vec[i] = val

"""
    set_minus_vec!(ev::T) where T <: Elt_vec

Multiply by `-1` the vector inside the element-vector `ev`.
"""
@inline set_minus_vec!(ev::T) where {T <: Elt_vec} = set_vec!(ev, -get_vec(ev))

"""
    set_add_vec!(ev::T, vec::Vector{Y}) where {T <: Elt_vec, Y <: Number}

Add `vec` to the vector `ev.vec` of the element-vector `ev`.
"""
@inline set_add_vec!(ev::T, vec::Vector{Y}) where {T <: Elt_vec, Y <: Number} = ev.vec .+= vec

end
