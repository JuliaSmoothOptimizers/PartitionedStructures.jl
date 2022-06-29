module ModElemental_plom_sr1

using SparseArrays, LinearOperators
using ..Utils
using ..M_abstract_part_struct, ..M_part_mat
using ..M_abstract_element_struct, ..M_elt_mat, ..ModElemental_elom_sr1

import Base.==, Base.copy, Base.similar
import ..M_abstract_part_struct: get_ee_struct

export Elemental_plom_sr1
export identity_eplom_LSR1, PLSR1_eplom, PLSR1_eplom_rand

"""
    Elemental_plom_sr1{T}<:Part_LO_mat{T}

Type that represents an elemental limited-memory partitioned quasi-Newton operator PLSR1.
"""
mutable struct Elemental_plom_sr1{T}<:Part_LO_mat{T}
  N::Int
  n::Int
  eelom_set::Vector{Elemental_elom_sr1{T}}
  spm::SparseMatrixCSC{T, Int}
  L::SparseMatrixCSC{T, Int}
  component_list::Vector{Vector{Int}}
  permutation::Vector{Int} # n-size vector
end

"""
    eelom_set = get_ee_struct(eplom)

Return the vector of every elemental element linear operator `eplom.eelom_set`.
"""
@inline get_ee_struct(eplom::Elemental_plom_sr1{T}) where T = get_eelom_set(eplom)

"""
    eelom = get_ee_struct(eplom, i)

Return the `i`-th elemental element linear operator `eplom.eelom_set[i]`.
"""
@inline get_ee_struct(eplom::Elemental_plom_sr1{T}, i::Int) where T = get_eelom_set(eplom, i)

@inline (==)(eplom1::Elemental_plom_sr1{T}, eplom2::Elemental_plom_sr1{T}) where T = (get_N(eplom1)==get_N(eplom2)) && (get_n(eplom1)==get_n(eplom2)) && (get_eelom_set(eplom1)==get_eelom_set(eplom2)) && (get_permutation(eplom1)==get_permutation(eplom2))
@inline copy(eplom::Elemental_plom_sr1{T}) where T = Elemental_plom_sr1{T}(copy(get_N(eplom)), copy(get_n(eplom)), copy.(get_eelom_set(eplom)), copy(get_spm(eplom)), copy(get_L(eplom)), copy(get_component_list(eplom)), copy(get_permutation(eplom)))
@inline similar(eplom::Elemental_plom_sr1{T}) where T = Elemental_plom_sr1{T}(copy(get_N(eplom)), copy(get_n(eplom)), similar.(get_eelom_set(eplom)), similar(get_spm(eplom)), similar(get_L(eplom)), copy(get_component_list(eplom)), copy(get_permutation(eplom)))

"""
    eplom = identity_eplom_LSR1(element_variables; N, n, T=T)
    eplom = identity_eplom_LSR1(element_variables, N, n; T=T)

Return an elemental partitionned limited-memory operator PLSR1 of `N` elemental element linear operators.
The positions are given by the vector of the element variables `element_variables`.
"""
identity_eplom_LSR1(element_variables::Vector{Vector{Int}}; N::Int=length(element_variables), n::Int=max_indices(element_variables), T=Float64) = identity_eplom_LSR1(element_variables, N, n; T=Float64)

function identity_eplom_LSR1(element_variables::Vector{Vector{Int}}, N::Int, n::Int; T=Float64)
  length(element_variables) != N && @error("unvalid list of element indices, PLSR1")
  eelom_set = map( (elt_var -> init_eelom_LSR1(elt_var; T=T)), element_variables)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom_sr1{T}(N, n, eelom_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplom)
  return eplom
end

"""
    eplom = PLSR1_eplom(;n, type, nie, overlapping)

Return an elemental partitionned limited-memory operator PLSR1 of `N` (deduced from `n` and `nie`) elemental element linear operators.
Each element overlaps the coordinates of the next element by `overlapping` components.
"""
function PLSR1_eplom(; n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1)
  overlapping < nie || error("the overlapping must be lower than nie")
  mod(n-(nie-overlapping), nie-overlapping)==mod(overlapping, nie-overlapping) || error("wrong structure: mod(n-(nie-over), nie-over)==mod(over, nie-over) must hold")

  indices = filter(x -> x <= n-nie+1, vcat(1, (x -> x + (nie-overlapping)).([1:nie-overlapping:n-(nie-overlapping);])))
  eelom_set = map(i -> LSR1_eelom(nie; T=T, index=i), indices)
  N = length(indices)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom_sr1{T}(N, n, eelom_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplom)
  return eplom
end

"""
    eplom = PLSR1_eplom_rand(N, n; type, nie)

Return an elemental partitionned limited-memory operator PLSR1 of `N` elemental element linear operators.
The size of each element is `nie`, whose positions are random in the range `1:n`.
"""
function PLSR1_eplom_rand(N::Int, n::Int; T=Float64, nie::Int=5)
  eelom_set = map(i -> LSR1_eelom_rand(nie; T=T, n=n), [1:N;])
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom_sr1{T}(N, n, eelom_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplom)
  return eplom
end

end 