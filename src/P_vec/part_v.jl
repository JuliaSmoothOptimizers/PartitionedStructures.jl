module M_part_v

using ..Acronyms
using ..M_abstract_part_struct

export Part_v
export get_v
export set_N!, set_n!, set_v!
export add_v!, build_v!, build_v, reset_v!

"""Supertype of every partitioned-vectors, ex : `Elemental_elt_vec`, `Internal_elt_vec`."""
abstract type Part_v{T} <: AbstractPartitionedStructure{T} end

"""
    v = get_v(pv::T) where T <: Part_v

Return the vector `pv.v` of the partitioned-vector `pv`.
"""
@inline get_v(pv::T) where {T <: Part_v} = pv.v

"""
    set_v!(pv::T, v::Vector{Y}) where {Y, T <: Part_v{Y}}
    set_v!(pv::T, i::Int, value::Y) where {Y, T <: Part_v{Y}}

Set the components of the vector `pv.v` (resp. `pv.v[i]`) from the partitioned-vector `pv` to the vector `v` (resp. `value`).
"""
@inline set_v!(pv::T, v::Vector{Y}) where {Y, T <: Part_v{Y}} = pv.v .= v
@inline set_v!(pv::T, i::Int, value::Y) where {Y, T <: Part_v{Y}} = pv.v[i] = value

"""
    add_v!(pv::T, i::Int, value::Y) where {Y, T <: Part_v{Y}}
    add_v!(pv::T, indices::Vector{Int}, values::Vector{Y}) where {Y, T <: Part_v{Y}}

Add `value` (resp `values`) to the vector of the partitioned-vector `pv.v` at the indice `i` (resp `indices`).
"""
@inline add_v!(pv::T, i::Int, value::Y) where {Y, T <: Part_v{Y}} = pv.v[i] += value
@inline add_v!(pv::T, indices::Vector{Int}, values::Vector{Y}) where {Y, T <: Part_v{Y}} =
  get_v(pv)[indices] .+= values

"""
    reset_v!(pv::T) where {Y, T <: Part_v{Y}}

Reset the vector embedded in the partitioned-vector `pv`, i.e. `pv.v .= (Y)(0)`.
"""
@inline reset_v!(pv::T) where {Y, T <: Part_v{Y}} = pv.v .= (Y)(0)

"""
    vec = build_v(pv::T) where T <: Part_v

Build the vector `v = pv.v` by accumulating inside `pv.v` the contributions of every element-vector of the partitioned-vector `pv`.
"""
function build_v(pv::T) where {T <: Part_v}
  build_v!(pv)
  return get_v(pv)
end

"""
    build_v!(pv::T) where T <: Part_v

Accumulate in `pv.v` the contributions of every element-vector of the partitioned-vector `pv`.
"""
@inline build_v!(pv::T) where {T <: Part_v} = error("M_part_v.build_v!() should not be call")

end
