module ModElemental_plo_bfgs

using ..Acronyms
using SparseArrays, LinearOperators
using ..M_abstract_part_struct, ..M_part_mat
using ..M_abstract_element_struct, ..M_elt_mat, ..ModElemental_elo_bfgs

import Base.==, Base.copy, Base.similar
import ..M_abstract_part_struct: get_ee_struct

export Elemental_plo_bfgs
export identity_eplo_LBFGS, PLBFGS_eplo, PLBFGS_eplo_rand

"""
    Elemental_plo_bfgs{T} <: Part_LO_mat{T}

Represent an elemental partitioned quasi-Newton limited-memory operator PLBFGS.
Each element is an elemental element `LBFGSOperator`.
The fields of `Elemental_plo_bfgs{T}`:
- `N` is the number of elements;
- `n` is the size of the $(_eplmo);
- `eelo_set` is the set of elemental element linear-operators;
- `spm` and `L` are sparse matrices either to form the sparse matrix gathering the elements or the Cholesky factor of `spm`;
- `component_list` summarizes for each variable i (∈ {1,..., n}) the list of elements (⊆ {1,...,N}) being parametrized by `i`;
- `permutation` is the current permutation of the $(_eplmo) (`[1:n;]` initially).
"""
mutable struct Elemental_plo_bfgs{T} <: Part_LO_mat{T}
  N::Int
  n::Int
  eelo_set::Vector{Elemental_elo_bfgs{T}}
  spm::SparseMatrixCSC{T, Int}
  L::SparseMatrixCSC{T, Int}
  component_list::Vector{Vector{Int}}
  permutation::Vector{Int} # n-size vector
end

# docstring defined in M_abstract_part_struct.get_ee_struct
@inline get_ee_struct(eplo::Elemental_plo_bfgs{T}) where {T} = get_eelo_set(eplo)
@inline get_ee_struct(eplo::Elemental_plo_bfgs{T}, i::Int) where {T} = get_eelo_set(eplo, i)

@inline (==)(eplo1::Elemental_plo_bfgs{T}, eplo2::Elemental_plo_bfgs{T}) where {T} =
  (get_N(eplo1) == get_N(eplo2)) &&
  (get_n(eplo1) == get_n(eplo2)) &&
  (get_eelo_set(eplo1) == get_eelo_set(eplo2)) &&
  (get_permutation(eplo1) == get_permutation(eplo2))
@inline copy(eplo::Elemental_plo_bfgs{T}) where {T} = Elemental_plo_bfgs{T}(
  copy(get_N(eplo)),
  copy(get_n(eplo)),
  copy.(get_eelo_set(eplo)),
  copy(get_spm(eplo)),
  copy(get_L(eplo)),
  copy(get_component_list(eplo)),
  copy(get_permutation(eplo)),
)
@inline similar(eplo::Elemental_plo_bfgs{T}) where {T} = Elemental_plo_bfgs{T}(
  copy(get_N(eplo)),
  copy(get_n(eplo)),
  similar.(get_eelo_set(eplo)),
  similar(get_spm(eplo)),
  similar(get_L(eplo)),
  copy(get_component_list(eplo)),
  copy(get_permutation(eplo)),
)

"""
    eplo = identity_eplo_LBFGS(element_variables::Vector{Vector{Int}}; N::Int=length(element_variables), n::Int=max_indices(element_variables), T=Float64, linear_vector::Vector{Bool} = zeros(Bool, N), mem=5)
    eplo = identity_eplo_LBFGS(element_variables::Vector{Vector{Int}}, N::Int, n::Int; T=Float64, linear_vector::Vector{Bool} = zeros(Bool, N), mem=5)

Return an $(_eplmo) PLBFGS of `N` elemental element linear-operators.
The positions are given by the vector of the element variables `element_variables`.
`linear_vector` indicates (with `true`) which element linear-operator should not contribute to the partitioned linear-operator.
"""
identity_eplo_LBFGS(
  element_variables::Vector{Vector{Int}};
  N::Int = length(element_variables),
  n::Int = max_indices(element_variables),
  T = Float64,
  linear_vector::Vector{Bool} = zeros(Bool, N),
  mem = 5,
) = identity_eplo_LBFGS(element_variables, N, n; T, linear_vector, mem)

function identity_eplo_LBFGS(
  element_variables::Vector{Vector{Int}},
  N::Int,
  n::Int;
  T = Float64,
  linear_vector::Vector{Bool} = zeros(Bool, N),
  mem = 5,
)
  eelo_set = map(
    (i -> init_eelo_LBFGS(element_variables[i]; T, linear = linear_vector[i], mem)),
    1:length(element_variables),
  )
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplo = Elemental_plo_bfgs{T}(N, n, eelo_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplo)
  return eplo
end

"""
    eplo = PLBFGS_eplo(; n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1)

Return an $(_eplmo) PLBFGS of `N` (deduced from `n` and `nie`) elemental element linear-operators.
Each element overlaps the coordinates of the next element by `overlapping` components.
"""
function PLBFGS_eplo(; n::Int = 9, T = Float64, nie::Int = 5, overlapping::Int = 1, mem = 5)
  overlapping < nie || error("the overlapping must be lower than nie")
  mod(n - (nie - overlapping), nie - overlapping) == mod(overlapping, nie - overlapping) ||
    error("wrong structure: mod(n-(nie-over), nie-over)==mod(over, nie-over) must hold")

  indices = filter(
    x -> x <= n - nie + 1,
    vcat(1, (x -> x + (nie - overlapping)).([1:(nie - overlapping):(n - (nie - overlapping));])),
  )
  eelo_set = map(i -> LBFGS_eelo(nie; T = T, index = i, mem), indices)
  N = length(indices)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplo = Elemental_plo_bfgs{T}(N, n, eelo_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplo)
  return eplo
end

"""
    eplo = PLBFGS_eplo_rand(N::Int, n::Int; T=Float64, nie::Int=5)

Return an $(_eplmo) PLBFGS of `N` elemental element linear-operators.
The size of each element is `nie` with random indices within the range `1:n`.
"""
function PLBFGS_eplo_rand(N::Int, n::Int; T = Float64, nie::Int = 5, mem = 5)
  eelo_set = map(i -> LBFGS_eelo_rand(nie; T = T, n = n, mem), [1:N;])
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplo = Elemental_plo_bfgs{T}(N, n, eelo_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplo)
  return eplo
end

end
