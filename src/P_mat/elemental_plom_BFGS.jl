module ModElemental_plom_bfgs

using SparseArrays, LinearOperators
using ..M_abstract_part_struct, ..M_part_mat
using ..M_abstract_element_struct, ..M_elt_mat, ..ModElemental_elom_bfgs

import Base.==, Base.copy, Base.similar
import Base.Matrix, SparseArrays.SparseMatrixCSC
import ..M_part_mat.set_spm!, ..M_part_mat.get_eelom_set	
import ..M_abstract_part_struct: initialize_component_list!, get_ee_struct

export Elemental_plom_bfgs
export get_eelom_set, get_eelom_set_Bie, get_eelom_sub_set, get_L, get_spm
export set_L!, set_L_to_spm!
export identity_eplom_LBFGS, PLBFGS_eplom, PLBFGS_eplom_rand 

"""
    Elemental_plom_bfgs{T} <: Part_LO_mat{T}

Type that represents an elemental limited-memory partitioned quasi-Newton operator PLBFGS.
"""
mutable struct Elemental_plom_bfgs{T} <: Part_LO_mat{T}
  N :: Int
  n :: Int
  eelom_set :: Vector{Elemental_elom_bfgs{T}}
  spm :: SparseMatrixCSC{T, Int}
  L :: SparseMatrixCSC{T, Int}
  component_list :: Vector{Vector{Int}}
  permutation :: Vector{Int} # n-size vector 
end

"""
    eelom_set = get_eelom_set(eplom)

Returns the vector of every elemental element linear operator `eplom.eelom_set`.
"""
@inline get_eelom_set(eplom :: Elemental_plom_bfgs{T}) where T = eplom.eelom_set

"""
    eelom = get_eelom_set(eplom :: Elemental_plom_bfgs{T}, i :: Int)

Returns the `i`-th elemental element linear operator `eplom.eelom_set[i]`.
"""
@inline get_eelom_set(eplom :: Elemental_plom_bfgs{T}, i :: Int) where T = @inbounds eplom.eelom_set[i]

"""
    eelom_set = get_ee_struct(eplom)

Returns the vector of every elemental element linear operator `eplom.eelom_set`.
"""
@inline get_ee_struct(eplom :: Elemental_plom_bfgs{T}) where T = get_eelom_set(eplom)

"""
    eelom = get_ee_struct(eplom, i)

Returns the `i`-th elemental element linear operator `eplom.eelom_set[i]`.
"""
@inline get_ee_struct(eplom :: Elemental_plom_bfgs{T}, i :: Int) where T = get_eelom_set(eplom, i)

"""
    eelom_subset = get_eelom_set_Bie(eplom, indices)

Returns a subset of the elemental element linear operators composing `eplom`.
`indices` selects the differents elemental element linear operators needed.
"""
@inline get_eelom_sub_set(eplom :: Elemental_plom_bfgs{T}, indices :: Vector{Int}) where T = eplom.eelom_set[indices]

"""
    Bie = get_eelom_set_Bie(eplom, i)

Returns the linear operator of the `i`-th elemental element linear operator of `eplom`.
"""
@inline get_eelom_set_Bie(eplom :: Elemental_plom_bfgs{T}, i :: Int) where T = get_Bie(get_eelom_set(eplom, i))	

"""
    L = get_L(eplom)

Returns the sparse matrix `eplom.L`, who aims to store a Cholesky factor.
By default `eplom.L` is not instantiate.
"""
@inline get_L(eplom :: Elemental_plom_bfgs{T}) where T = eplom.L

"""
    Lij = get_L(eplom, i, j)

Returns the value `eplom.L[i,j]`, from the sparse matrix `eplom.L`.
"""
@inline get_L(eplom :: Elemental_plom_bfgs{T}, i :: Int, j :: Int) where T = @inbounds eplom.L[i, j]

"""
    set_L!(eplom, i, j, value)

Sets the value of `eplom.L[i,j] = value`.
"""
@inline set_L!(eplom :: Elemental_plom_bfgs{T}, i :: Int, j :: Int, value :: T) where T = @inbounds eplom.L[i, j] = value

"""
    set_L_to_spm!(eplom, i, j, value)

Sets the sparse matrix `eplom.L` to the sparse matrix `eplom.spm`.
"""
@inline set_L_to_spm!(eplom :: Elemental_plom_bfgs{T}) where T = eplom.L .= eplom.spm

@inline (==)(eplom1 :: Elemental_plom_bfgs{T}, eplom2 :: Elemental_plom_bfgs{T}) where T = (get_N(eplom1) == get_N(eplom2)) && (get_n(eplom1) == get_n(eplom2)) && (get_eelom_set(eplom1) .== get_eelom_set(eplom2)) && (get_permutation(eplom1) == get_permutation(eplom2))
@inline copy(eplom :: Elemental_plom_bfgs{T}) where T = Elemental_plom_bfgs{T}(copy(get_N(eplom)), copy(get_n(eplom)), copy.(get_eelom_set(eplom)), copy(get_spm(eplom)), copy(get_L(eplom)), copy(get_component_list(eplom)), copy(get_permutation(eplom)))
@inline similar(eplom :: Elemental_plom_bfgs{T}) where T = Elemental_plom_bfgs{T}(copy(get_N(eplom)), copy(get_n(eplom)), similar.(get_eelom_set(eplom)), similar(get_spm(eplom)), similar(get_L(eplom)), copy(get_component_list(eplom)), copy(get_permutation(eplom)))
  
"""
    eplom = identity_eplom_LBFGS(element_variables, N, n; T=T)
    
Returns an elemental partitioned limited-memory operator PLBFGS of `N` elemental element linear operators.
The positions are given by the vector of the element variables `element_variables`.
"""
function identity_eplom_LBFGS(element_variables :: Vector{Vector{Int}}, N :: Int, n :: Int; T=Float64)		
  eelom_set = map( (elt_var -> init_eelom_LBFGS(elt_var; T=T)), element_variables)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom_bfgs{T}(N, n, eelom_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplom)
  return eplom
end 

"""
    eplom = PLBFGS_eplom(;n, type, nie, overlapping)

Returns an elemental partitioned limited-memory operator PLBFGS of `N` (deduced from `n` and `nie`) elemental element linear operators.
Each element overlaps the coordinates of the next element by `overlapping` components.
"""
function PLBFGS_eplom(; n :: Int=9, T=Float64, nie :: Int=5, overlapping :: Int=1)		
  overlapping < nie || error("the overlapping must be lower than nie")
  mod(n-(nie-overlapping), nie-overlapping) == mod(overlapping, nie-overlapping) || error("wrong structure: mod(n-(nie-over), nie-over) == mod(over, nie-over) must hold")

  indices = filter(x -> x <= n-nie+1, vcat(1, (x -> x + (nie-overlapping)).([1:nie-overlapping:n-(nie-overlapping);])))
  eelom_set = map(i -> LBFGS_eelom(nie; T=T, index=i), indices)	
  N = length(indices)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom_bfgs{T}(N, n, eelom_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplom)
  return eplom
end 

"""
    eplom = PLBFGS_eplom_rand(N, n; type, nie)

Returns an elemental partitioned limited-memory operator PLBFGS of `N` elemental element linear operators.
The size of each element is `nie`, whose positions are random in the range `1:n`.
"""
function PLBFGS_eplom_rand(N :: Int, n :: Int; T=Float64, nie :: Int=5)		
  eelom_set = map(i -> LBFGS_eelom_rand(nie; T=T, n=n), [1:N;])
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom_bfgs{T}(N, n, eelom_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplom)	
  return eplom
end 

"""
    initialize_component_list!(eplom)

Builds for each index i (∈ {1, ..., n}) a list of the elements using the i-th variable.
"""
function initialize_component_list!(eplom :: Elemental_plom_bfgs)
  N = get_N(eplom)
  for i in 1:N
    eelomᵢ = get_eelom_set(eplom, i)
    _indices = get_indices(eelomᵢ)
    for j in _indices 
      push!(get_component_list(eplom, j), i)
    end 
  end 
end 

"""
    set_spm!(eplom)

Builds the sparse matrix of `eplom` in `eplom.spm` from all the elemental element linear operator.
The sparse matrix is built with respect to the indices of each elemental element linear operator.
"""
function set_spm!(eplom :: Elemental_plom_bfgs{T}) where T
  reset_spm!(eplom)
  N = get_N(eplom)	
  n = get_n(eplom)
  spm = get_spm(eplom)
  for i in 1:N
    eplomᵢ = get_eelom_set(eplom, i)
    nie = get_nie(eplomᵢ)
    Bie = get_Bie(eplomᵢ)
    indicesᵢ = get_indices(eplomᵢ)
    value_Bie = zeros(T, nie, nie)
    map( (i -> value_Bie[:, i] .= Bie*SparseVector(nie, [i], [1])), 1:nie)
    spm[indicesᵢ, indicesᵢ] .+= value_Bie			
  end 
  return spm
end

function Base.Matrix(eplom :: Elemental_plom_bfgs{T}) where T
  set_spm!(eplom)
  sp_eplom = get_spm(eplom)
  m = Matrix(sp_eplom)
  return m
end 

function SparseArrays.SparseMatrixCSC(eplom :: Elemental_plom_bfgs{T}) where T
  set_spm!(eplom)
  get_spm(eplom)
end

end 