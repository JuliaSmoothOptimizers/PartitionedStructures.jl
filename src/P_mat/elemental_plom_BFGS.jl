module ModElemental_plom_bfgs

using SparseArrays, LinearOperators
using ..M_abstract_part_struct, ..M_part_mat
using ..M_abstract_element_struct, ..M_elt_mat, ..ModElemental_elom_bfgs

import Base.==, Base.copy, Base.similar
import ..M_abstract_part_struct: get_ee_struct

export Elemental_plom_bfgs
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
    eelom_set = get_ee_struct(eplom)

Returns the vector of every elemental element linear operator `eplom.eelom_set`.
"""
@inline get_ee_struct(eplom :: Elemental_plom_bfgs{T}) where T = get_eelom_set(eplom)

"""
    eelom = get_ee_struct(eplom, i)

Returns the `i`-th elemental element linear operator `eplom.eelom_set[i]`.
"""
@inline get_ee_struct(eplom :: Elemental_plom_bfgs{T}, i :: Int) where T = get_eelom_set(eplom, i)

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

end 