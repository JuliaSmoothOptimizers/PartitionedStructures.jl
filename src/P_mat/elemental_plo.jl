module ModElemental_plo

using ..Acronyms
using SparseArrays, LinearOperators
using ..Utils
using ..M_part_mat, ..M_abstract_part_struct
using ..M_elt_mat, ..M_abstract_element_struct, ..ModElemental_elo_bfgs, ..ModElemental_elo_sr1

import Base.==, Base.copy, Base.similar
import ..M_part_mat: set_eelo_set!
import ..M_abstract_part_struct: get_ee_struct

export Elemental_plo
export identity_eplo_LOSE, PLBFGSR1_eplo, PLBFGSR1_eplo_rand

elom_type{T} = Union{Elemental_elo_sr1{T}, Elemental_elo_bfgs{T}}

"""
    Elemental_plo{T} <: Part_LO_mat{T}

Represent an elemental partitioned quasi-Newton limited-memory operator PLSE.
Each element may either be a `LBFGSOperator` or a `LSR1Operator`.
`N` is the number of elements.
`n` is the size of the $(_eplmo).
`eelo_set` is the set of elemental element linear-operators.
`spm` and `L` are sparse matrices either to form the sparse matrix gathering the elements or the Cholesky factor of `spm`.
`component_list` summarizes for each variable i (∈ {1,..., n}) the list of elements (⊆ {1,...,N}) being parametrised by `i`.
`permutation` is the current permutation of the $(_eplmo) (`[1:n;]` initially).
"""
mutable struct Elemental_plo{T} <: Part_LO_mat{T}
  N::Int
  n::Int
  eelo_set::Vector{elom_type{T}}
  spm::SparseMatrixCSC{T, Int}
  L::SparseMatrixCSC{T, Int}
  component_list::Vector{Vector{Int}}
  permutation::Vector{Int} # n-size vector
end

# docstring in ModElemental_plo.set_eelo_set!
@inline set_eelo_set!(eplo::Elemental_plo{T}, i::Int, eelo::Y) where {Y <: LOEltMat{T}} where {T} =
  @inbounds eplo.eelo_set[i] = eelo

# docstring defined in M_abstract_part_struct.get_ee_struct
@inline get_ee_struct(eplo::Elemental_plo{T}) where {T} = get_eelo_set(eplo)
@inline get_ee_struct(eplo::Elemental_plo{T}, i::Int) where {T} = get_eelo_set(eplo, i)

@inline (==)(eplo1::Elemental_plo{T}, eplo2::Elemental_plo{T}) where {T} =
  (get_N(eplo1) == get_N(eplo2)) &&
  (get_n(eplo1) == get_n(eplo2)) &&
  (get_eelo_set(eplo1) == get_eelo_set(eplo2)) &&
  (get_permutation(eplo1) == get_permutation(eplo2))
@inline copy(eplo::Elemental_plo{T}) where {T} = Elemental_plo{T}(
  copy(get_N(eplo)),
  copy(get_n(eplo)),
  copy.(get_eelo_set(eplo)),
  copy(get_spm(eplo)),
  copy(get_L(eplo)),
  copy(get_component_list(eplo)),
  copy(get_permutation(eplo)),
)
@inline similar(eplo::Elemental_plo{T}) where {T} = Elemental_plo{T}(
  copy(get_N(eplo)),
  copy(get_n(eplo)),
  similar.(get_eelo_set(eplo)),
  similar(get_spm(eplo)),
  similar(get_L(eplo)),
  copy(get_component_list(eplo)),
  copy(get_permutation(eplo)),
)

"""
    eplo = identity_eplo_LOSE(element_variables::Vector{Vector{Int}}; N::Int=length(element_variables), n::Int=max_indices(element_variables), T=Float64)
    eplo = identity_eplo_LOSE(element_variables::Vector{Vector{Int}}, N::Int, n::Int; T=Float64)

Create an elemental partitionned limited-memory operator of `N` elemental element linear-operators initialized with LBFGS operators.
The positions are given by the vector of the element variables `element_variables`.
"""
identity_eplo_LOSE(
  element_variables::Vector{Vector{Int}};
  N::Int = length(element_variables),
  n::Int = max_indices(element_variables),
  T = Float64,
) = identity_eplo_LOSE(element_variables, N, n; T)

function identity_eplo_LOSE(element_variables::Vector{Vector{Int}}, N::Int, n::Int; T = Float64)
  eelo_set = map((elt_var -> init_eelo_LBFGS(elt_var; T = T)), element_variables)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplo = Elemental_plo{T}(N, n, eelo_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplo)
  return eplo
end

"""
    eplo = PLBFGSR1_eplo(;n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1, prob=0.5)

Create an elemental partitionned limited-memory operator PLSE of `N` (deduced from `n` and `nie`) elemental element linear-operators.
Each element overlaps the coordinates of the next element by `overlapping` components.
Each element is randomly (`rand() > p`) choose between an elemental element LBFGS operator or an elemental element LSR1 operator.
"""
function PLBFGSR1_eplo(; n::Int = 9, T = Float64, nie::Int = 5, overlapping::Int = 1, prob = 0.5)
  overlapping < nie || error("the overlapiing must be smaller than nie")
  mod(n - (nie - overlapping), nie - overlapping) == mod(overlapping, nie - overlapping) ||
    error("wrong structure: mod(n-(nie-over), nie-over)==mod(over, nie-over) must holds")

  indices = filter(
    x -> x <= n - nie + 1,
    vcat(1, (x -> x + (nie - overlapping)).([1:(nie - overlapping):(n - (nie - overlapping));])),
  )
  eelo_set = map(
    i -> rand() > prob ? LBFGS_eelo(nie; T = T, index = i) : LSR1_eelo(nie; T = T, index = i),
    indices,
  )
  N = length(indices)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplo = Elemental_plo{T}(N, n, eelo_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplo)
  return eplo
end

"""
    eplo = PLBFGSR1_eplo_rand(N::Int, n ::Int; T=Float64, nie::Int=5, prob=0.5)

Create an elemental partitionned limited-memory operator PLSE of `N` elemental element linear-operators.
The size of each element is `nie`, whose positions are random in the range `1:n`.
Each element is randomly (rand() > p) choose between an elemental element LBFGS operator or an elemental element LSR1 operator.
"""
function PLBFGSR1_eplo_rand(N::Int, n::Int; T = Float64, nie::Int = 5, prob = 0.5)
  eelo_set = map(
    i -> rand() > prob ? LBFGS_eelo_rand(nie; T = T, n = n) : LSR1_eelo_rand(nie; T = T, n = n),
    [1:N;],
  )
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplo = Elemental_plo{T}(N, n, eelo_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplo)
  return eplo
end

end
