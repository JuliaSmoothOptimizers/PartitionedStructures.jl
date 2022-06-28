
module M_abstract_part_struct
using ..M_abstract_element_struct
import Base.==

export Part_struct
export get_n, get_N, get_component_list
export check_epv_epm, full_check_epv_epm
export initialize_component_list!, get_ee_struct

"""Supertype of every partitioned-structure, ex: Elemental_pv, Elemental_pm, Elemental_plom_bfgs, Internal_pv, ..."""
abstract type Part_struct{T} end

"""
    get_n(ps :: T) where T <: Part_struct

Return the total size of the partitioned-structure `ps`.
"""
get_n(ps :: T) where T <: Part_struct = ps.n

"""
    get_N(ps :: T) where T <: Part_struct

Return the number of element composing the partitioned-structure `ps`.
"""
get_N(ps :: T) where T <: Part_struct = ps.N

"""
    list = get_component_list(ps :: T) where T <: Part_struct
    ith_component = get_component_list(ps :: T, i::Int) where T <: Part_struct

Return either the list of every element-structure composing the partitioned-structure `ps` or the `i`-th element-structure of `ps`.
"""
get_component_list(ps :: T) where T <: Part_struct = ps.component_list
get_component_list(ps :: T, i::Int) where T <: Part_struct = ps.component_list[i]

"""
    (==)(ps1 :: T, ps2 :: T) where T <: Part_struct

Return true if both partitioned-structures are composed of the same amont of elemnt-structures, and have the same size.
"""
(==)(ps1 :: T, ps2 :: T) where T <: Part_struct = get_n(ps1)==get_n(ps2) && get_N(ps1)==get_N(ps2)

"""
    bool = check_epv_epm(epm :: Y, epv :: Z) where {Y <: Part_struct, Z <: Part_struct}

Similar to `==`, but it can compare different partitioned-structures, example: an `Elemental_pv` and an `Elemental_pm`.
`check_epv_epm` is a superficial test, see `full_check_epv_epm(epm, epv)` for a complete check of the partitioned-structure (i.e. each element depending of the same variable subset).
"""
@inline check_epv_epm(epm :: Y, epv :: Z) where {Y <: Part_struct, Z <: Part_struct} = get_N(epm) == get_N(epv) && get_n(epm) == get_n(epv)

"""
    full_check_epv_epm(ep1 :: Y, ep2 :: Z) where {Y <: Part_struct, Z <: Part_struct}

Check if each element-structure of both partitioned-structures depend of the same subset of variables.
"""
@inline full_check_epv_epm(ep1 :: Y, ep2 :: Z) where {Y <: Part_struct, Z <: Part_struct} = check_epv_epm(ep1,ep2) && get_component_list(ep1) == get_component_list(ep2)

# define the function, initialize_component_list! is instantiated in the modules: ModElemental_pv, ModElemental_pm...
initialize_component_list!(ps::T) where T <: Part_struct = @error("should not be called")

# define the function, initialize_component_list! is instantiated in the modules: ModElemental_pv, ModElemental_pm...
get_ee_struct(ps::T) where T <: Part_struct = @error("should not be called")
end