module ModElemental_plom

using SparseArrays, LinearOperators
using ..Utils
using ..M_part_mat, ..M_abstract_part_struct
using ..M_elt_mat, ..M_abstract_element_struct, ..ModElemental_elom_bfgs, ..ModElemental_elom_sr1

import Base.==, Base.copy, Base.similar
import ..M_part_mat: set_eelom_set!
import ..M_abstract_part_struct: get_ee_struct

export Elemental_plom
export identity_eplom_LOSE, PLBFGSR1_eplom, PLBFGSR1_eplom_rand

elom_type{T} = Union{Elemental_elom_sr1{T}, Elemental_elom_bfgs{T}}

"""
    Elemental_plom{T}<:Part_LO_mat{T}

Type that represents an elemental limited-memory partitioned quasi-Newton linear operator, each Bᵢ may use a LBFGS or LSR1 linear operator.
"""
mutable struct Elemental_plom{T}<:Part_LO_mat{T}
  N::Int
  n::Int
  eelom_set::Vector{elom_type{T}}
  spm::SparseMatrixCSC{T,Int}
  L::SparseMatrixCSC{T,Int}
  component_list::Vector{Vector{Int}}
  permutation::Vector{Int} # n-size vector
end

# docstring in ModElemental_plom.set_eelom_set!
@inline set_eelom_set!(eplom::Elemental_plom{T}, i::Int, eelom::Y) where Y<:LOEltMat{T} where T = @inbounds eplom.eelom_set[i] = eelom

# docstring defined in M_abstract_part_struct.get_ee_struct
@inline get_ee_struct(eplom::Elemental_plom{T}) where T = get_eelom_set(eplom)
@inline get_ee_struct(eplom::Elemental_plom{T}, i::Int) where T = get_eelom_set(eplom, i)

@inline (==)(eplom1::Elemental_plom{T}, eplom2::Elemental_plom{T}) where T = (get_N(eplom1)==get_N(eplom2)) && (get_n(eplom1)==get_n(eplom2)) && (get_eelom_set(eplom1)==get_eelom_set(eplom2)) && (get_permutation(eplom1)==get_permutation(eplom2))
@inline copy(eplom::Elemental_plom{T}) where T = Elemental_plom{T}(copy(get_N(eplom)),copy(get_n(eplom)),copy.(get_eelom_set(eplom)),copy(get_spm(eplom)), copy(get_L(eplom)),copy(get_component_list(eplom)),copy(get_permutation(eplom)))
@inline similar(eplom::Elemental_plom{T}) where T = Elemental_plom{T}(copy(get_N(eplom)),copy(get_n(eplom)),similar.(get_eelom_set(eplom)),similar(get_spm(eplom)), similar(get_L(eplom)),copy(get_component_list(eplom)),copy(get_permutation(eplom)))

"""
    eplom = identity_eplom_LOSE(element_variables::Vector{Vector{Int}}; N::Int=length(element_variables), n::Int=max_indices(element_variables), T=Float64)
    eplom = identity_eplom_LOSE(element_variables::Vector{Vector{Int}}, N::Int, n::Int; T=Float64)

Create an elemental partitionned limited-memory operator of `N` elemental element linear operators initialized with LBFGS operators.
The positions are given by the vector of the element variables `element_variables`.
"""
identity_eplom_LOSE(element_variables::Vector{Vector{Int}}; N::Int=length(element_variables), n::Int=max_indices(element_variables), T=Float64) = identity_eplom_LOSE(element_variables, N, n; T)

function identity_eplom_LOSE(element_variables::Vector{Vector{Int}}, N::Int, n::Int; T=Float64)
  eelom_set = map( (elt_var -> init_eelom_LBFGS(elt_var; T=T)), element_variables)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom{T}(N, n, eelom_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplom)
  return eplom
end

"""
    eplom = PLBFGSR1_eplom(;n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1, prob=0.5)

Create an elemental partitionned limited-memory operator PLSE of `N` (deduced from `n` and `nie`) elemental element linear operators.
Each element overlaps the coordinates of the next element by `overlapping` components.
Each element is randomly (`rand() > p`) choose between an elemental element LBFGS operator or an elemental element LSR1 operator.
"""
function PLBFGSR1_eplom(;n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1, prob=0.5)
  overlapping < nie || error("the overlapiing must be smaller than nie")
  mod(n-(nie-overlapping), nie-overlapping)==mod(overlapping, nie-overlapping) || error("wrong structure: mod(n-(nie-over), nie-over)==mod(over, nie-over) must holds")

  indices = filter(x -> x <= n-nie+1, vcat(1,(x -> x + (nie-overlapping)).([1:nie-overlapping:n-(nie-overlapping);])))
  eelom_set = map(i -> rand() > prob ? LBFGS_eelom(nie;T=T,index=i) : LSR1_eelom(nie;T=T,index=i), indices)
  N = length(indices)
  spm = spzeros(T,n,n)
  L = spzeros(T,n,n)
  component_list = map(i -> Vector{Int}(undef,0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom{T}(N,n,eelom_set,spm,L,component_list,no_perm)
  initialize_component_list!(eplom)
  return eplom
end

"""
    eplom = PLBFGSR1_eplom_rand(N::Int, n ::Int; T=Float64, nie::Int=5, prob=0.5)

Create an elemental partitionned limited-memory operator PLSE of `N` elemental element linear operators.
The size of each element is `nie`, whose positions are random in the range `1:n`.
Each element is randomly (rand() > p) choose between an elemental element LBFGS operator or an elemental element LSR1 operator.
"""
function PLBFGSR1_eplom_rand(N::Int, n ::Int; T=Float64, nie::Int=5, prob=0.5)
  eelom_set = map(i -> rand() > prob ? LBFGS_eelom_rand(nie;T=T,n=n) : LSR1_eelom_rand(nie;T=T,n=n), [1:N;])
  spm = spzeros(T,n,n)
  L = spzeros(T,n,n)
  component_list = map(i -> Vector{Int}(undef,0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom{T}(N,n,eelom_set,spm,L,component_list,no_perm)
  initialize_component_list!(eplom)
  return eplom
end

end