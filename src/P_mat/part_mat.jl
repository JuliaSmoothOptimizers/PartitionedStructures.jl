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

export get_eelom_set, get_eelom_set, set_spm!

"Abstract type representing partitioned matrix"
abstract type Part_mat{T} <: Part_struct{T} end
"Abstract type representing partitioned matrix using linear operators"
abstract type Part_LO_mat{T} <: Part_mat{T} end

@inline set_spm!(pm :: T) where T <: Part_mat = @error("should not be called")

"""
    get_spm(pm)

    get_spm(pm, i, j)

Get either the sparse matrix associated to the partitioned matrix `pm` or `pm[i,j]`.
"""
@inline get_spm(pm :: T) where T <: Part_mat = pm.spm
@inline get_spm(pm :: T, i :: Int, j :: Int) where T <: Part_mat = @inbounds get_spm(pm)[i,j]

"""
    get_permutation(pm)

Gets the current permutation of the partitioned matrix `pm`.
"""
@inline get_permutation(pm :: T) where T <: Part_mat = pm.permutation

"""
    get_permutation(pm, perm)

Set the permutation of the partitioned matrix `pm` to `perm`.
"""
@inline set_permutation!(pm :: T, perm :: Vector{Int}) where T <: Part_mat = pm.permutation .= perm

"""
    reset_spm!(pm)

Set the elements of sparse matrix `pm.spm` to `0`.
"""
@inline reset_spm!(pm :: T) where T <: Part_mat{Y} where Y <: Number  = pm.spm.nzval .= (Y)(0)

"""
    hard_reset_spm!(pm)

Reset the sparse matrix `pm.spm`.
"""
@inline hard_reset_spm!(pm :: T) where T <: Part_mat = pm.spm = spzeros(T, get_n(pm), get_n(pm))

"""
    reset_L!(pm)

Set the elements of sparse matrix `pm.L` to `0`.
"""
@inline reset_L!(pm :: T) where T <: Part_mat{Y} where Y <: Number = pm.L.nzval .= (Y)(0)

"""
    hard_reset_L!(pm)

Reset the sparse matrix `pm.L`.
"""
@inline hard_reset_L!(pm :: T) where T <: Part_mat = pm.L = spzeros(T, get_n(pm), get_n(pm))

# @inline get_eelom_set(plm :: T) where T <: Part_LO_mat = @error("should not be called")
"""
    eelmon_set = get_eelom_set(eplom)

Returns the vector of every elemental element linear operator `eplom.eelom_set`.
"""
@inline get_eelom_set(plm :: T) where T <: Part_LO_mat  = plm.eelom_set

"""
    eelom = get_eelom_set(eplom :: Elemental_plom_bfgs{T}, i :: Int)

Returns the `i`-th elemental element linear operator `eplom.eelom_set[i]`.
"""
@inline get_eelom_set(plm :: T, i :: Int) where T <: Part_LO_mat = @inbounds plm.eelom_set[i]

@inline set_eelom_set!(plm :: T) where T <: Part_LO_mat = @error("should not be called")

""" 
    get_ee_struct_Bie(pm, i)

Returns the `i`-th elemental element matrix of the partitioned matrix `pm`.
"""
@inline get_ee_struct_Bie(pm :: T, i :: Int) where T <: Part_mat = get_Bie(get_ee_struct(pm, i))

"""
    initialize_component_list!(eplom)

Builds for each index i (∈ {1, ..., n}) a list of the elements using the i-th variable.
"""
function initialize_component_list!(plm :: P) where {T<:Number, P <: Part_LO_mat{T}}
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
    set_spm!(eplom)

Builds the sparse matrix of `eplom` in `eplom.spm` from all the elemental element linear operator.
The sparse matrix is built with respect to the indices of each elemental element linear operator.
"""
function set_spm!(plm :: P) where {T<:Number, P <: Part_LO_mat{T}}
  reset_spm!(plm) # eplom.spm .= 0
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

function Base.Matrix(pm :: P) where {T<:Number, P <: Part_mat{T}}
  set_spm!(pm)
  sp_pm = get_spm(pm)
  m = Matrix(sp_pm)
  return m
end 

function SparseArrays.SparseMatrixCSC(pm :: P) where {T<:Number, P <: Part_mat{T}}
  set_spm!(pm)
  get_spm(pm)
end

end