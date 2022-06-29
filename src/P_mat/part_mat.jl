module M_part_mat
using SparseArrays
using ..M_abstract_element_struct
using ..M_abstract_part_struct, ..M_elt_mat

import ..M_abstract_part_struct.initialize_component_list!

export Part_mat, Part_LO_mat
export get_N, get_n, get_permutation, get_spm
export hard_reset_spm!, reset_spm!, set_spm!
export hard_reset_L!, reset_L!
export set_N!, set_n!, set_permutation!
export get_eelom_set, get_ee_struct_Bie
export set_eelom_set!

export get_eelom_set, get_eelom_set, set_spm!, get_eelom_sub_set, get_eelom_set_Bie
export get_spm, get_L
export set_L!, set_L_to_spm!

"Abstract type representing partitioned-matrix"
abstract type Part_mat{T}<:Part_struct{T} end
"Abstract type representing partitioned-matrix using linear operators"
abstract type Part_LO_mat{T}<:Part_mat{T} end

@inline set_spm!(pm::T) where T<:Part_mat = @error("should not be called")

"""
    spm = get_spm(pm::T) where T<:Part_mat
    spm_ij = get_spm(pm::T, i::Int, j::Int) where T<:Part_mat

Get either the sparse matrix associated to the partitioned-matrix `pm` or `pm[i,j]`.
"""
@inline get_spm(pm::T) where T<:Part_mat = pm.spm
@inline get_spm(pm::T, i::Int, j::Int) where T<:Part_mat = @inbounds get_spm(pm)[i,j]

"""
    perm = get_permutation(pm::T) where T<:Part_mat

Gets the current permutation of the partitioned-matrix `pm`.
"""
@inline get_permutation(pm::T) where T<:Part_mat = pm.permutation

"""
    set_permutation!(pm::T, perm::Vector{Int}) where T<:Part_mat

Set the permutation of the partitioned-matrix `pm` to `perm`.
"""
@inline set_permutation!(pm::T, perm::Vector{Int}) where T<:Part_mat = pm.permutation .= perm

"""
    reset_spm!(pm::T) where {Y<:Number, T<:Part_mat{Y}}

Set the elements of sparse matrix `pm.spm` to `0`.
"""
@inline reset_spm!(pm::T) where {Y<:Number, T<:Part_mat{Y}} = pm.spm.nzval .= (Y)(0)

"""
    hard_reset_spm!(pm::T) where T<:Part_mat

Reset the sparse matrix `pm.spm`.
"""
@inline hard_reset_spm!(pm::T) where T<:Part_mat = pm.spm = spzeros(T, get_n(pm), get_n(pm))

"""
    reset_L!(pm)

Set the elements of sparse matrix `pm.L` to `0`.
"""
@inline reset_L!(pm::T) where T<:Part_mat{Y} where Y<:Number = pm.L.nzval .= (Y)(0)

"""
    hard_reset_L!(pm::T) where T<:Part_mat

Reset the sparse matrix `pm.L`.
"""
@inline hard_reset_L!(pm::T) where T<:Part_mat = pm.L = spzeros(T, get_n(pm), get_n(pm))

"""
    L = get_L(pm::T) where T<:Part_mat

Returns the sparse matrix `pm.L`, who aims to store a Cholesky factor.
By default `pm.L` is not instantiate.
"""
@inline get_L(pm::T) where T<:Part_mat = pm.L

"""
    L_ij = get_L(pm::T, i::Int, j::Int) where T<:Part_mat

Returns the value `pm.L[i,j]`, from the sparse matrix `pm.L`.
"""
@inline get_L(pm::T, i::Int, j::Int) where T<:Part_mat = @inbounds pm.L[i, j]

"""
    set_L!(pm::P, i::Int, j::Int, value::T) where {T<:Number, P<:Part_mat{T}}

Sets the value of `pm.L[i,j] = value`.
"""
@inline set_L!(pm::P, i::Int, j::Int, value::T) where {T<:Number, P<:Part_mat{T}} = @inbounds pm.L[i, j] = value

"""
    set_L_to_spm!(pm::T) where T<:Part_mat

Sets the sparse matrix `plm.L` to the sparse matrix `plm.spm`.
"""
@inline set_L_to_spm!(pm::T) where T<:Part_mat = pm.L .= pm.spm

"""
    eelmon_set = get_eelom_set(plm::T) where T<:Part_LO_mat

Returns the vector of every elemental element linear operator `plm.eelom_set`.
"""
@inline get_eelom_set(plm::T) where T<:Part_LO_mat = plm.eelom_set

"""
    eelom = get_eelom_set(plm::T, i::Int) where T<:Part_LO_mat

Returns the `i`-th elemental element linear operator `plm.eelom_set[i]`.
"""
@inline get_eelom_set(plm::T, i::Int) where T<:Part_LO_mat = @inbounds plm.eelom_set[i]

"""
    eelom_subset = get_eelom_sub_set(plm::T, indices::Vector{Int}) where T<:Part_LO_mat

Returns a subset of the elemental element linear operators composing `plm`.
`indices` selects the differents elemental element linear operators needed.
"""
@inline get_eelom_sub_set(plm::T, indices::Vector{Int}) where T<:Part_LO_mat = plm.eelom_set[indices]

"""
    Bie = get_eelom_set_Bie(plm::T, i::Int) where T<:Part_LO_mat

Returns the linear operator of the `i`-th elemental element linear operator of `plm`.
"""
@inline get_eelom_set_Bie(plm::T, i::Int) where T<:Part_LO_mat = get_Bie(get_eelom_set(plm, i))


@inline set_eelom_set!(plm::T) where T<:Part_LO_mat = @error("should not be called")

"""
    get_ee_struct_Bie(pm::T, i::Int) where T<:Part_mat

Returns the `i`-th elemental element-matrix of the partitioned-matrix `pm`.
"""
@inline get_ee_struct_Bie(pm::T, i::Int) where T<:Part_mat = get_Bie(get_ee_struct(pm, i))

"""
    initialize_component_list!(plm)

Builds for each index i (∈ {1, ..., n}) a list of the elements using the i-th variable.
"""
function initialize_component_list!(plm::P) where {T<:Number, P<:Part_LO_mat{T}}
  N = get_N(plm)
  for i in 1:N
    plmᵢ = get_eelom_set(plm, i)
    _indices = get_indices(plmᵢ)
    for j in _indices
      push!(get_component_list(plm, j), i)
    end
  end
end

"""
    set_spm!set_spm!(plm::P) where {T<:Number, P<:Part_LO_mat{T}}

Builds the sparse matrix of `plm` in `plm.spm` from all the elemental element linear operator.
The sparse matrix is built with respect to the indices of each elemental element linear operator.
"""
function set_spm!(plm::P) where {T<:Number, P<:Part_LO_mat{T}}
  reset_spm!(plm) # plm.spm .= 0
  N = get_N(plm)
  n = get_n(plm)
  spm = get_spm(plm)
  for i in 1:N
    plmᵢ = get_eelom_set(plm,i)
    nie = get_nie(plmᵢ)
    Bie = get_Bie(plmᵢ)
    indicesᵢ = get_indices(plmᵢ)
    value_Bie = zeros(T,nie,nie)
    map( (i -> value_Bie[:,i] .= Bie*SparseVector(nie,[i],[1])), 1:nie)
    spm[indicesᵢ,indicesᵢ] .+= value_Bie
  end
end

function Base.Matrix(pm::P) where {T<:Number, P<:Part_mat{T}}
  set_spm!(pm)
  sp_pm = get_spm(pm)
  m = Matrix(sp_pm)
  return m
end

function SparseArrays.SparseMatrixCSC(pm::P) where {T<:Number, P<:Part_mat{T}}
  set_spm!(pm)
  get_spm(pm)
end

end