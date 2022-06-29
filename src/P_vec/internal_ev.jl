# unsupported for now
module M_internal_elt_vec

using LinearAlgebra, SparseArrays, StatsBase
using ..M_elt_vec, ..ModElemental_ev, ..M_abstract_element_struct

import Base.==

export Internal_elt_vec
export get_lin_comb, get_nii, get_tmp
export set_lin_comb!, set_nii
export iev_from_eev, iev_from_sparse_vec, new_iev, ones_iev
export build_tmp!

# distinguish from internal_elt_vec
# Be careful with the order of the indices
"""
    Internal_elt_vec{T}<:Elt_vec{T}

Type that represents an internal element-vector.
"""
mutable struct Internal_elt_vec{T}<:Elt_vec{T}
  vec::Vector{T} # size nᵢᴵ
  indices::Vector{Int} # size nᵢᴱ
  lin_comb::SparseMatrixCSC{T,Int} # size nᵢᴵ× nᵢᴱ
  nie::Int
  nii::Int
  _tmp::Vector{T} # size nᵢᴱ
end

"""
    linear_combination = get_lin_comb(iev::Internal_elt_vec{T}) where T

Warning: unsupported and not tested.
Return `linear_combination`, as a `SparseMatrixCSC` informing the internal variables of the interal element-vector `iev`.
"""
@inline get_lin_comb(iev::Internal_elt_vec{T}) where T = iev.lin_comb

"""
    nii = get_nii(iev::Internal_elt_vec{T}) where T

Warning: unsupported and not tested.
Return `nii`, the internal dimension of the internal element-vector `iev`.
It may differ from the elemental dimension `iev.nie`.
"""
@inline get_nii(iev::Internal_elt_vec{T}) where T = iev.nii

"""
    tmp = get_tmp(iev::Internal_elt_vec{T}) where T
    tmp_i = get_tmp(iev::Internal_elt_vec{T}, i::Int) where T

Warning: unsupported and not tested.
Return the vector associated to the internal element-vector of `tmp`, or its `i`-th component.
The size of `tmp` is `iev.nie`.
`tmp` is the contribution of the internal element-vector `iev` as a part of a partitioned-vector.
"""
@inline get_tmp(iev::Internal_elt_vec{T}) where T = iev._tmp
@inline get_tmp(iev::Internal_elt_vec{T}, i::Int) where T = iev._tmp[i]

"""
    set_lin_comb!(iev::Internal_elt_vec{T}, lin_comb::SparseMatrixCSC{T,Int}) where T

Warning: unsupported and not tested.
Set the internal variables `iev.lin_comb` of the internal element-vector `iev` to the `lin_comb::SparseMatrixCSC`. 
"""
@inline set_lin_comb!(iev::Internal_elt_vec{T}, lin_comb::SparseMatrixCSC{T,Int}) where T = iev.lin_comb = lin_comb

"""
    set_nii!(iev::Internal_elt_vec{T}, nii::Int) where T

Warning: unsupported and not tested.
Set the internal dimension `iev.nii` of the internal element-vector to `nii`.
"""
@inline set_nii!(iev::Internal_elt_vec{T}, nii::Int) where T = iev.nii = nii

(==)(iev1::Internal_elt_vec{T}, iev2::Internal_elt_vec{T}) where T = (get_vec(iev1)==get_vec(iev2)) && (get_indices(iev1)==get_indices(iev2)) && (get_lin_comb(iev1)==get_lin_comb(iev2)) && (get_nie(iev1)==get_nie(iev2)) && (get_nii(iev1)==get_nii(iev2))

"""
    iev = new_iev(nᵢᴱ:: Int, nᵢᴵ:: Int; T=Float64, n=nᵢᴱ^2, prop=0.5)

Warning: unsupported and not tested.
Return a internal element-vector `iev` with random vectors of suitable size and a random `SparseMatrixCSC` informing the internal variables.
"""
@inline new_iev(nᵢᴱ:: Int, nᵢᴵ:: Int; T=Float64, n=nᵢᴱ^2, prop=0.5) = Internal_elt_vec(rand(T,nᵢᴵ), sample(1:n,nᵢᴱ,replace = false), sprand(T,nᵢᴵ, nᵢᴱ, prop), nᵢᴱ, nᵢᴵ, rand(T,nᵢᴱ))

"""
    iev = ones_iev(nᵢᴱ:: Int, nᵢᴵ:: Int; T=Float64, n=nᵢᴱ^2, prop=0.5)

Warning: unsupported and not tested.
Return a internal element-vector `iev`.
`iev.vec` is set to `ones(T, nᵢᴵ)`, the other vectors are randomly choose of suitable size.
In addition a random `SparseMatrixCSC` informing the internal variables.
"""
@inline ones_iev(nᵢᴱ:: Int, nᵢᴵ:: Int; T=Float64, n=nᵢᴱ^2, prop=0.5) = Internal_elt_vec(ones(T,nᵢᴵ), sample(1:n,nᵢᴱ,replace = false), sprand(T,nᵢᴵ, nᵢᴱ, prop), nᵢᴱ, nᵢᴵ, rand(T,nᵢᴱ))

"""
    iev = iev_from_sparse_vec(sv::SparseVector{T,Y}) where {T,Y}

Warning: unsupported and not tested.
Return an internal element-vector from `sv::SparseVector`.
`iev` is created from the elemental element-vector deduces of `sv`.
"""
@inline iev_from_sparse_vec(sv ::SparseVector{T,Y}) where {T,Y} = iev_from_eev(eev_from_sparse_vec(sv))

"""
    iev = iev_from_eev(eev::Elemental_elt_vec{T}) where T

Warning: unsupported and not tested.
Return an internal element-vector `iev` from an elemental element-vector `eev`.
The internal variables are the same than the element variables.
"""
function iev_from_eev(eev::Elemental_elt_vec{T}) where T
  nie = get_nie(eev)
  indices = get_indices(eev)
  vec = get_vec(eev)
  lin_com = spzeros(T,nie,nie) # identity matrix Matrix(I,n,n) didn't work
  [lin_com[i,i] = 1 for i in 1:nie]
  _tmp = rand(T,nie)
  iev = Internal_elt_vec{T}(vec, indices, lin_com, nie, nie, _tmp)
  return iev
end

"""
    build_tmp!(iev::Internal_elt_vec{T}) where T

Warning: unsupported and not tested.
Build in place `iev.tmp`, the contribution of the internal element-vector `iev` as a part of a partitioned-vector.
"""
build_tmp!(iev::Internal_elt_vec{T}) where T = mul!(iev._tmp, transpose(iev.lin_comb), iev.vec)

end