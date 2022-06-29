# unsupported for now
module M_internal_pv

using LinearAlgebra
using ..M_abstract_element_struct, ..M_abstract_part_struct
using ..M_elt_vec, ..M_internal_elt_vec # element modules
using ..M_part_v, ..ModElemental_pv # partitoned modules

export Internal_pv
export get_iev, get_iev_set
export create_ipv, ipv_from_epv, rand_ipv

"""
    Internal_elt_vec{T}<:Elt_vec{T}

Type that represents an internal partitioned-vector.
"""
mutable struct Internal_pv{T}<:Part_v{T}
  N::Int
  n::Int
  iev_set::Vector{Internal_elt_vec{T}}
  v::Vector{T}
end

"""
    iev_set = get_iev_set(ipv::Internal_pv{T}) where T

Warning: unsupported and not tested.
Return the set of internal element-vectors `iev_set`, which are contribuating to the internal partitioned-vector `ipv`.
"""
@inline get_iev_set(ipv::Internal_pv{T}) where T = ipv.iev_set

"""
    iev = get_iev(ipv::Internal_pv{T}, i::Int) where T

Warning: unsupported and not tested.
Return the `i`-th internal element-vector of the internal partitioned-vector `ipv`.
"""
@inline get_iev(ipv::Internal_pv{T}, i::Int) where T = ipv.iev_set[i] # i <= N

"""
    ipv = ipv_from_epv(epv::Elemental_pv{T}) where T

Warning: unsupported and not tested.
Return an internal partitioned-vector `ipv` from the elemental partitioned vector `epv`.
The internal variables of every internal element-vectors are the same as the elemental variables of the elemental element-vectors.
"""
function ipv_from_epv(epv::Elemental_pv{T}) where T
  N = get_N(epv)
  n = get_n(epv)
  iev_set = iev_from_eev.(get_eev_set(epv))
  v = get_v(epv)
  ipv = Internal_pv{T}(N, n, iev_set, v)
  return ipv
end

"""
    ipv = create_ipv(iev_set::Vector{Internal_elt_vec{T}}; n=max_indices(iev_set)) where T

Warning: unsupported and not tested.
Return an internal partitioned-vector `ipv` from the set of internal element-vectors `iev_set`.
"""
function create_ipv(iev_set::Vector{Internal_elt_vec{T}}; n=max_indices(iev_set)) where T
  N = length(iev_set)
  v = zeros(T,n)
  ipv = Internal_pv{T}(N, n, iev_set, v)
  return ipv
end

"""
    ipv = new_internal_pv(N::Int,n::Int; nᵢ=3, T=Float64)

Warning: unsupported and not tested.
Define an internal partitioned-vector of `N` elemental element-vectors of size `nᵢ` and type `T`.
"""
function rand_ipv(N::Int,n::Int; nᵢ=3, T=Float64)
  iev_set = Vector{Internal_elt_vec{T}}(undef,N)
  for i in 1:N
    nᵢᴱ = rand(max(nᵢ-1,0):nᵢ+1)
    nᵢᴵ = rand(max(nᵢᴱ-3,1):nᵢᴱ+1)
    iev_set[i] = new_iev(nᵢᴱ, nᵢᴵ; T=T, n=n)
  end
  v = rand(T,n)
  return Internal_pv{T}(N, n, iev_set, v)
end

"""
    build_v!(ipv::Internal_pv{T}) where T

Warning1: unsupported and not tested.
Build in place the vector `ipv.v` by accumating the contributions of every internal element-vector.
Warning2: the order of `ipv.indices` is crucial to get the expected result.
The order of `ipv.lin_comb`, `ipv.vec`, `ipv.indices` must be synchronized.
"""
function M_part_v.build_v!(ipv::Internal_pv{T}) where T
  reset_v!(ipv)
  N = get_N(ipv)
  for i in 1:N
    ievᵢ = get_iev(ipv,i)
    nᵢᴱ = get_nie(ievᵢ)
    for j in 1:nᵢᴱ
      _id_j = get_indices(ievᵢ,j)
      build_tmp!(ievᵢ)
      val = get_tmp(ievᵢ,j)
      add_v!(ipv, _id_j, val)
    end
  end
  return ipv
end

end