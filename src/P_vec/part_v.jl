module M_part_v

using ..M_abstract_part_struct

export Part_v
export get_v
export set_N!, set_n!, set_v!
export add_v!, build_v!, build_v, reset_v!

"""Abstract type representing the partitioned-vectors."""
abstract type Part_v{T} <: Part_struct{T} end

@inline get_v(pv :: T ) where T <: Part_v = pv.v

"""
    set_v!(pv, vec)
    set_v!(pv, index, value)

Set the component of the vector from the partitioned-vector `pv.v[index]` to `value` .
Set the vector from the partitioned-vector `pv.v`to `vec`.
"""
@inline set_v!(pv :: T, v :: Vector{Y} ) where T <: Part_v{Y} where Y = pv.v = v
@inline set_v!(pv :: T, i :: Int, value :: Y ) where T <: Part_v{Y} where Y = pv.v[i] = value

"""
    add_v!(pv, i, value)
    add_v!(pv, indices, values)

Add `value` (resp `values`) to the vector of the partitioned-vector `pv.v` at the indice `i` (resp `indices`).
"""
@inline add_v!(pv :: T, i :: Int, value :: Y ) where T <: Part_v{Y} where Y = pv.v[i] += value
@inline add_v!(pv :: T, indices :: Vector{Int}, values :: Vector{Y}) where T <: Part_v{Y} where Y = get_v(pv)[indices] .+= values


@inline reset_v!(pv :: T ) where T <: Part_v{Y} where Y = pv.v .= (Y)(0)

"""
    vec = build_v(pv)

Build the vector `v = pv.v` by accumulating inside `pv.v` the contributions of the element-vectors of the partitioned-vector `pv`.
"""
@inline build_v(pv :: T) where T <: Part_v = begin build_v!(pv); return get_v(pv) end

"""
    build_v!(pv)

Accumulatie in `pv.v` the contributions of the element-vectors of the partitioned-vector `pv`.
"""
@inline build_v!(pv :: T) where T <: Part_v = error("M_part_v.build_v!() should not be call")

end