module M_abstract_part_struct

using ..M_abstract_element_struct
using ..Acronyms

import Base.==

export AbstractPartitionedStructure
export get_n, get_N, get_component_list
export check_epv_epm, full_check_epv_epm
export initialize_component_list!, get_ee_struct

"""Supertype of every partitioned-structure, ex: `Elemental_pv`, `Elemental_pm`, `Elemental_plo_bfgs`, `Internal_pv`..."""
abstract type AbstractPartitionedStructure{T} end

"""
    get_n(ps::T) where T <: AbstractPartitionedStructure

Return the total size of the $(_ps).
"""
get_n(ps::T) where {T <: AbstractPartitionedStructure} = ps.n

"""
    get_N(ps::T) where T <: AbstractPartitionedStructure

Return the number of element composing the $(_ps).
"""
get_N(ps::T) where {T <: AbstractPartitionedStructure} = ps.N

"""
    list = get_component_list(ps::T) where T <: AbstractPartitionedStructure
    ith_component = get_component_list(ps::T, i::Int) where T <: AbstractPartitionedStructure

Return either the list of every element-structure composing the $(_ps) or the `i`-th element-structure of `ps`.
"""
get_component_list(ps::T) where {T <: AbstractPartitionedStructure} = ps.component_list
get_component_list(ps::T, i::Int) where {T <: AbstractPartitionedStructure} = ps.component_list[i]

"""
    (==)(ps1::T, ps2::T) where T <: AbstractPartitionedStructure

Return true if both partitioned-structures are composed of the same amont of element-structures, and have the same size.
"""
(==)(ps1::T, ps2::T) where {T <: AbstractPartitionedStructure} =
  get_n(ps1) == get_n(ps2) && get_N(ps1) == get_N(ps2)

"""
    bool = check_epv_epm(epm::Y, epv::Z) where {Y <: AbstractPartitionedStructure, Z <: AbstractPartitionedStructure}

Similar to `==`, but it can compare different partitioned-structures, example: an `Elemental_pv` and an `Elemental_pm`.
`check_epv_epm` is a superficial test, see `full_check_epv_epm(epm, epv)` for a complete check of the partitioned-structure (i.e. if each element depends of the same variable subset).
"""
@inline check_epv_epm(
  epm::Y,
  epv::Z,
) where {Y <: AbstractPartitionedStructure, Z <: AbstractPartitionedStructure} =
  get_N(epm) == get_N(epv) && get_n(epm) == get_n(epv)

"""
    full_check_epv_epm(ep1::Y, ep2::Z) where {Y <: AbstractPartitionedStructure, Z <: AbstractPartitionedStructure}

Check if each element-structure of both partitioned-structures depend of the same subset of variables.
"""
@inline full_check_epv_epm(
  ep1::Y,
  ep2::Z,
) where {Y <: AbstractPartitionedStructure, Z <: AbstractPartitionedStructure} =
  check_epv_epm(ep1, ep2) && get_component_list(ep1) == get_component_list(ep2)

"""
    initialize_component_list!(ps::T) where T <: AbstractPartitionedStructure)

Build for each variable i (∈ {1,..., n}) the list of elements (⊆ {1,...,N}) being parametrised by `i`.
"""
initialize_component_list!(ps::T) where {T <: AbstractPartitionedStructure} =
  error("should not be called")

"""
    ee_vector = get_ee_struct(eps::AbstractPartitionedStructure{T}) where T
    ee = get_ee_struct(eps::AbstractPartitionedStructure{T}, i::Int) where T

Return a vector of every elemental elements fom $(_eps) or only its `i`-th elemental element.
"""
get_ee_struct(ps::T) where {T <: AbstractPartitionedStructure} = error("should not be called")

end
