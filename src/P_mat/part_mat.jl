module M_part_mat

using ..Acronyms
using SparseArrays
using ..M_abstract_element_struct
using ..M_abstract_part_struct, ..M_elt_mat

import ..M_abstract_part_struct.initialize_component_list!

export Part_mat, Part_LO_mat
export get_N, get_n, get_permutation, get_spm
export hard_reset_spm!, reset_spm!, set_spm!
export hard_reset_L!, reset_L!
export set_N!, set_n!, set_permutation!
export get_eelo_set, get_ee_struct_Bie
export set_eelo_set!

export get_eelo_sub_set, get_eelo_set_Bie
export get_spm, get_L
export set_L!, set_L_to_spm!

"Supertype of every partitioned-matrix, ex: `Elemental_pm`, `Elemental_plo_bfgs`, `Elemental_plo_sr1`, `Elemental_plo`."
abstract type Part_mat{T} <: AbstractPartitionedStructure{T} end
"Supertype of every partitioned limited-memory operator, ex: `Elemental_plo_bfgs`, `Elemental_plo_sr1`, `Elemental_plo`."
abstract type Part_LO_mat{T} <: Part_mat{T} end

"""
    set_spm!(pm::P) where {T <: Number, P <: Part_mat{T}}

Build the sparse matrix of the partitioned-matrix `pm` in `pm.spm` by gathering the contribution of every element-matrix.
The sparse matrix is built with respect to the indices of each elemental element linear-operator.
"""
@inline set_spm!(pm::T) where {T <: Part_mat} = @error("should not be called")

function set_spm!(plm::P) where {T <: Number, P <: Part_LO_mat{T}}
  reset_spm!(plm) # plm.spm .= 0
  N = get_N(plm)
  n = get_n(plm)
  spm = get_spm(plm)
  for i = 1:N
    elmᵢ = get_eelo_set(plm, i)    
    linear = get_linear(elmᵢ)
    if !linear
      nie = get_nie(elmᵢ)
      Bie = get_Bie(elmᵢ)
      indicesᵢ = get_indices(elmᵢ)
      value_Bie = zeros(T, nie, nie)
      map((i -> value_Bie[:, i] .= Bie * SparseVector(nie, [i], [1])), 1:nie)
      spm[indicesᵢ, indicesᵢ] .+= value_Bie
    end
  end
  return plm
end

"""
    spm = get_spm(pm::T) where T <: Part_mat
    spm_ij = get_spm(pm::T, i::Int, j::Int) where T <: Part_mat

Get either the sparse matrix associated to the partitioned-matrix `pm` or `pm[i,j]`.
"""
@inline get_spm(pm::T) where {T <: Part_mat} = pm.spm
@inline get_spm(pm::T, i::Int, j::Int) where {T <: Part_mat} = @inbounds get_spm(pm)[i, j]

"""
    perm = get_permutation(pm::T) where T <: Part_mat

Get the current permutation of the partitioned-matrix `pm`.
"""
@inline get_permutation(pm::T) where {T <: Part_mat} = pm.permutation

"""
    set_permutation!(pm::T, perm::Vector{Int}) where T <: Part_mat

Set the permutation of the partitioned-matrix `pm` to `perm`.
"""
@inline set_permutation!(pm::T, perm::Vector{Int}) where {T <: Part_mat} = pm.permutation .= perm

"""
    reset_spm!(pm::T) where {Y <: Number, T <: Part_mat{Y}}

Set the elements of sparse matrix `pm.spm` to `0`.
"""
@inline reset_spm!(pm::T) where {Y <: Number, T <: Part_mat{Y}} = pm.spm.nzval .= (Y)(0)

"""
    hard_reset_spm!(pm::T) where T <: Part_mat

Reset the sparse matrix `pm.spm`.
"""
@inline hard_reset_spm!(pm::T) where {T <: Part_mat} = pm.spm = spzeros(T, get_n(pm), get_n(pm))

"""
    reset_L!(pm)

Set the elements of sparse matrix `pm.L` to `0`.
"""
@inline reset_L!(pm::T) where {T <: Part_mat{Y}} where {Y <: Number} = pm.L.nzval .= (Y)(0)

"""
    hard_reset_L!(pm::T) where T <: Part_mat

Reset the sparse matrix `pm.L`.
"""
@inline hard_reset_L!(pm::T) where {T <: Part_mat} = pm.L = spzeros(T, get_n(pm), get_n(pm))

"""
    L = get_L(pm::T) where T <: Part_mat

Return the sparse matrix `pm.L`, who aims to store a Cholesky factor.
By default `pm.L` is not instantiate.
"""
@inline get_L(pm::T) where {T <: Part_mat} = pm.L

"""
    L_ij = get_L(pm::T, i::Int, j::Int) where T <: Part_mat

Return the value `pm.L[i,j]`, from the sparse matrix `pm.L`.
"""
@inline get_L(pm::T, i::Int, j::Int) where {T <: Part_mat} = @inbounds pm.L[i, j]

"""
    set_L!(pm::P, i::Int, j::Int, value::T) where {T <: Number, P <: Part_mat{T}}

Set the value of `pm.L[i,j] = value`.
"""
@inline set_L!(pm::P, i::Int, j::Int, value::T) where {T <: Number, P <: Part_mat{T}} =
  @inbounds pm.L[i, j] = value

"""
    set_L_to_spm!(pm::T) where T <: Part_mat

Set the sparse matrix `plm.L` to the sparse matrix `plm.spm`.
"""
@inline set_L_to_spm!(pm::T) where {T <: Part_mat} = pm.L .= pm.spm

"""
    eelo_set = get_eelo_set(plm::T) where T <: Part_LO_mat
    eelo = get_eelo_set(plm::T, i::Int) where T <: Part_LO_mat

Return either the vector of every elemental element linear-operator `plm.eelo_set` or the `i`-th elemental element linear-operator `plm.eelo_set[i]`.
"""
@inline get_eelo_set(plm::T) where {T <: Part_LO_mat} = plm.eelo_set
@inline get_eelo_set(plm::T, i::Int) where {T <: Part_LO_mat} = @inbounds plm.eelo_set[i]

"""
    eelo_subset = get_eelo_sub_set(plm::T, indices::Vector{Int}) where T <: Part_LO_mat

Return a subset of the elemental element linear-operators composing the elemental partitioned limited-memory operator `plm`.
`indices` selects the differents elemental element linear-operators needed.
"""
@inline get_eelo_sub_set(plm::T, indices::Vector{Int}) where {T <: Part_LO_mat} =
  plm.eelo_set[indices]

"""
    Bie = get_eelo_set_Bie(plm::T, i::Int) where T <: Part_LO_mat

Return the linear-operator of the `i`-th elemental element linear-operator of `plm`.
"""
@inline get_eelo_set_Bie(plm::T, i::Int) where {T <: Part_LO_mat} = get_Bie(get_eelo_set(plm, i))

"""
    set_eelo_set!(eplo::P, i::Int, eelo::Y) where {T, P <: Part_LO_mat{T}, Y <: LOEltMat{T}}

Set the `i`-th elemental element linear-operator `eplo.eelo` to `eelo`.
"""
@inline set_eelo_set!(eplo::T) where {T <: Part_LO_mat} = @error("should not be called")

"""
    get_ee_struct_Bie(pm::T, i::Int) where T <: Part_mat

Return the `i`-th elemental element-matrix of the partitioned-matrix `pm`.
"""
@inline get_ee_struct_Bie(pm::T, i::Int) where {T <: Part_mat} = get_Bie(get_ee_struct(pm, i))

# docstring in M_abstract_part_struct.initialize_component_list!
function initialize_component_list!(plm::P) where {T <: Number, P <: Part_LO_mat{T}}
  N = get_N(plm)
  for i = 1:N
    plmᵢ = get_eelo_set(plm, i)
    _indices = get_indices(plmᵢ)
    for j in _indices
      push!(get_component_list(plm, j), i)
    end
  end
  return plm
end

function Base.Matrix(pm::P) where {T <: Number, P <: Part_mat{T}}
  set_spm!(pm)
  sp_pm = get_spm(pm)
  m = Matrix(sp_pm)
  return m
end

function SparseArrays.SparseMatrixCSC(pm::P) where {T <: Number, P <: Part_mat{T}}
  set_spm!(pm)
  get_spm(pm)
end

end
