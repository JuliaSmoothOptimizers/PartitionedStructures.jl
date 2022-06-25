module ModElemental_pm

using SparseArrays
using ..Utils
using ..M_abstract_part_struct, ..M_part_mat
using ..M_abstract_element_struct, ..M_elt_mat, ..ModElemental_em

import Base.==, Base.copy, Base.similar
import Base.permute!
import ..M_part_mat: set_spm!
import ..M_abstract_part_struct: initialize_component_list!, get_ee_struct

export Elemental_pm
export get_eem_set, get_eem_set_Bie, get_eem_sub_set
export correlated_var
export identity_epm, ones_epm, ones_epm_and_id, n_i_sep, n_i_SPS, part_mat

"""
    Elemental_pm{T} <: Part_mat{T}

Type that represents an elemental partitioned quasi-Newton linear operator, each Bᵢ may apply a BFGS or a SR1 update.
"""
mutable struct Elemental_pm{T} <: Part_mat{T}
  N :: Int
  n :: Int
  eem_set :: Vector{Elemental_em{T}}
  spm :: SparseMatrixCSC{T, Int}
  L :: SparseMatrixCSC{T, Int}
  component_list :: Vector{Vector{Int}}
  permutation :: Vector{Int} # n-size vector
end

#getter/setter
"""
    eem_set = get_eem_set(epm)

Return the vector of every elemental element-matrices `epm.eem_set`.
"""
@inline get_eem_set(epm :: Elemental_pm{T}) where T = epm.eem_set

"""
    eem = get_eem_set(epm :: Elemental_pm_bfgs{T}, i :: Int)

Return the `i`-th elemental element-matrix `epm.eem_set[i]`.
"""
@inline get_eem_set(epm :: Elemental_pm{T}, i :: Int) where T = @inbounds epm.eem_set[i]

"""
    eem_set = get_ee_struct(epm)

Return the vector of every elemental element-matrices `epm.eem_set`.
"""
@inline get_ee_struct(epm :: Elemental_pm{T}) where T = get_eem_set(epm)

"""
    eem = get_ee_struct(epm, i)

Return the `i`-th elemental element-matrix `epm.eem_set[i]`.
"""
@inline get_ee_struct(epm :: Elemental_pm{T}, i :: Int) where T = get_eem_set(epm, i)

"""
    eem_subset = get_eem_set_Bie(epm, indices)

Return a subset of elemental element-matrices composing `epm`.
`indices` selects the different elemental element-matrices needed.
"""
@inline get_eem_sub_set(epm :: Elemental_pm{T}, indices :: Vector{Int}) where T = epm.eem_set[indices]

"""
    Bieget_eem_set_Bie(epm, i)

Get the matrix of the `i`-th elemental element-matrix of `epm`.
"""
@inline get_eem_set_Bie(epm :: Elemental_pm{T}, i :: Int) where T = get_Bie(get_eem_set(epm, i))

@inline (==)(epm1 :: Elemental_pm{T}, epm2 :: Elemental_pm{T}) where T = (get_N(epm1) == get_N(epm2)) && (get_n(epm1) == get_n(epm2)) && (get_eem_set(epm1).== get_eem_set(epm2)) && (get_permutation(epm1) == get_permutation(epm2))
@inline copy(epm :: Elemental_pm{T}) where T = Elemental_pm{T}(copy(get_N(epm)), copy(get_n(epm)), copy.(get_eem_set(epm)), copy(get_spm(epm)), copy(get_L(epm)), copy(get_component_list(epm)), copy(get_permutation(epm)))
@inline similar(epm :: Elemental_pm{T}) where T = Elemental_pm{T}(copy(get_N(epm)), copy(get_n(epm)), similar.(get_eem_set(epm)), similar(get_spm(epm)), similar(get_L(epm)), copy(get_component_list(epm)), copy(get_permutation(epm)))

"""
    epm = identity_epm(element_variables; N, n, T=T)
    epm = identity_epm(element_variables, N, n; T=T)

Return a partitionned matrix of type `T` of `N` identity elemental element-matrices.
`N` and `n` may be extrapolate from `element_variables`.
The elemental variables are based from the indices informed in `element_variables`.
"""
identity_epm(element_variables :: Vector{Vector{Int}}; N::Int=length(element_variables), n::Int=max_indices(element_variables), T=Float64) = identity_epm(element_variables, N, n; T=T)

function identity_epm(element_variables :: Vector{Vector{Int}}, N :: Int, n :: Int; T=Float64)
  eem_set = map( (elt_var -> create_id_eem(elt_var; T=T)), element_variables)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  epm = Elemental_pm{T}(N, n, eem_set, spm, L, component_list, no_perm)
  initialize_component_list!(epm)
  return epm
end

"""
    epm = identity_epm(N, n; T=T, nie)

Return a partitionned matrix of type `T` of `N` identity elemental element-matrices.
Each elemental element-matrix is of size `nie` with randoms positions.
"""
function identity_epm(N :: Int, n :: Int; T=Float64, nie :: Int=5)
  eem_set = map(i -> identity_eem(nie; T=T, n=n), [1:N;])
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  epm = Elemental_pm{T}(N, n, eem_set, spm, L, component_list, no_perm)
  initialize_component_list!(epm)
  return epm
end

"""
    epm = ones_epm(N, n; T=T, nie=nie)

Create a partitionned matrix of type `T` of `N` elemental element-matrices `ones(nie, nie)` whose positions are random.
The partitionned matrix created may be singular.
"""
function ones_epm(N :: Int, n :: Int; T=Float64, nie :: Int=5)
  eem_set = map(i -> ones_eem(nie; T=T, n=n), [1:N;])
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm= [1:n;]
  epm = Elemental_pm{T}(N, n, eem_set, spm, L, component_list, no_perm)
  initialize_component_list!(epm)
  return epm
end

"""
    ones_epm_and_id(N, n; T=T, nie=nie)

Create a partitionned matrix of type `T` with `N+n` elemental element-matrices.
The first `N` elemental element-matrices are `ones(nie, nie)` with randoms positions in the range `1:n`.
The remaining `n` elemental element-matrices are of size 1, with value [1], they are placed in the diagonal terms
This way, the partitionned matrix is generally not singular.
"""
function ones_epm_and_id(N :: Int, n :: Int; T=Float64, nie :: Int=5)
  eem_set1 = map(i -> ones_eem(nie; T=T, n=n), [1:N;])
  eem_set2 = map(i -> one_size_bloc(i; T=T), [1:n;])
  eem_set = vcat(eem_set1, eem_set2)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm= [1:n;]
  epm = Elemental_pm{T}(N+n, n, eem_set, spm, L, component_list, no_perm)
  initialize_component_list!(epm)
  return epm
end

"""
    n_i_sep(n; T=T, nie=nie, mul=mul)

Define a partitioned `nie` bloc separable matrix.
Each elemental element-matrix is composed of `1` except the diagonal terms which are of value `mul`.
"""
function n_i_sep(n :: Int; T=Float64, nie :: Int=5, mul=5.)
  mod(n, nie) == 0 || error("n must be a multiple of nie")
  eem_set = map(i -> fixed_ones_eem(i, nie; T=T, mul=mul), [1:nie:n;])
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  N = Int(floor(n/nie))
  epm = Elemental_pm{T}(N, n, eem_set, spm, L, component_list, no_perm)
  initialize_component_list!(epm)
  return epm
end

"""
    n_i_SPS(n; T, nie, overlapping, mul)

Define an elemental partitioned-matrix of size `n`.
The partitioned-matrix is composed by `N ≈ (n/nie)*2` elemental element-matrices, of size `nie`, they overlap onto the next element by `overlapping`.
The diagonal terms of each elemental element-matrix are of value `mul`, whereas the other terms are set to 1.
"""
function n_i_SPS(n :: Int; T=Float64, nie :: Int=5, overlapping :: Int=1, mul=5.)
  mod(n, nie) == 0 || error("n must be a multiple of nie")
  overlapping < nie || error("the overlapping must be lower than nie")
  eem_set1 = map(i -> fixed_ones_eem(i, nie; T=T, mul=mul), [1:nie:n;])
  eem_set2 = map(i -> fixed_ones_eem(i, 2*overlapping; T=T, mul=mul), [nie-overlapping:nie:n-nie-overlapping;])
  eem_set = vcat(eem_set1, eem_set2)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  N = length(eem_set)
  epm = Elemental_pm{T}(N, n, eem_set, spm, L, component_list, no_perm)
  initialize_component_list!(epm)
  return epm
end

"""
    part_mat(;n=n, T=T, nie=nie, overlapping=overlapping, mul=mul)

Define a elemental partitioned-matrix formed by `N` (deduced from `n` and `nie`) elemental element-matrices of size `nie`.
Each elemental element-matrix overlaps the previousand the next element by `overlapping`.
"""
function part_mat(;n :: Int=9, T=Float64, nie :: Int=5, overlapping :: Int=1, mul=5.)
  overlapping < nie || error("the overlapping must be lower than nie")
  mod(n-(nie-overlapping), nie-overlapping) == mod(overlapping, nie-overlapping) || error("wrong structure: mod(n-(nie-over), nie-over) == mod(over, nie-over) must holds")
  indices = filter(x -> x <= n-nie+1, vcat(1, (x -> x + (nie-overlapping)).([1:nie-overlapping:n-(nie-overlapping);])))
  eem_set = map(i -> fixed_ones_eem(i, nie;T=T, mul=mul), indices)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  N = length(eem_set)
  epm = Elemental_pm{T}(N, n, eem_set, spm, L, component_list, no_perm)
  initialize_component_list!(epm)
  # set_spm!(epm)
  return epm
end

"""
    initialize_component_list!(epm)

Build for each index i (∈ {1, ..., n}) a list of the elements using the i-th variable.
"""
function initialize_component_list!(epm :: Elemental_pm)
  N = get_N(epm)
  n = get_n(epm)
  for i in 1:N
    eemᵢ = get_eem_set(epm, i)
    _indices = get_indices(eemᵢ)
    for j in _indices
      push!(get_component_list(epm, j), i)
    end
  end
end

"""
    set_spm!(eplom)

Build the sparse matrix of `eplom` in `eplom.spm` from the blocs `eplom.eelom_set`.
The sparse matrix is built with respect to the indices of each elemental element linear operator.
"""
function set_spm!(epm :: Elemental_pm{T}) where T
  reset_spm!(epm) # epm.spm .= 0
  N = get_N(epm)
  n = get_n(epm)
  spm = get_spm(epm)
  for i in 1:N
    epmᵢ = get_eem_set(epm, i)
    nie = get_nie(epmᵢ)
    Bie = get_Bie(epmᵢ)
    for i in 1:nie, j in 1:nie
      val = Bie[i, j]
      real_i = get_indices(epmᵢ, i) # epmᵢ.indices[i]
      real_j = get_indices(epmᵢ, j) # epmᵢ.indices[j]
      spm[real_i, real_j] += val
    end
  end
end

"""
    permute!(epm, p)

Apply the permutation `p` to the elemental partitionned matrix `epm`.
The permutation is applied to every elemental element-matrix `eem` via `indices`.
The current `epm` permutation is stored in `epm.permutation`.
"""
function permute!(epm :: Elemental_pm{T}, p :: Vector{Int}) where T
  N = get_N(epm)
  n = get_n(epm)
  # permute on element-matrix
  for i in 1:N
    epmᵢ = get_eem_set(epm, i)
    indicesᵢ = get_indices(epmᵢ)
    e_perm = Vector(view(p, indicesᵢ))
    permute!(epmᵢ, e_perm)
  end
  # permute on the permutation vector
  perm = get_permutation(epm)
  permute!(perm, p)
  # permute component list
  new_component_list = Vector{Vector{Int}}(undef, n)
  for i in 1:n
    new_component_list[i] = get_component_list(epm, p[i])
  end
  hard_reset_spm!(epm) # hard reset of the sparse matrix
end

"""
    correlated_var(epm, i)

Get the variables that appears in the same elements than the `i`-th variable.
"""
function correlated_var(epm :: Elemental_pm{T}, i :: Int) where T
  component_list = get_component_list(epm)
  bloc_list = component_list[i]
  indices_list = Vector{Int}(undef, 0)
  for (id_j, j) in enumerate(bloc_list)
    eemᵢ = get_eem_set(epm, j)
    _indices = get_indices(eemᵢ)
    append!(indices_list, _indices)
  end
  var_list = vcat(indices_list...)
  unique!(var_list)
  return var_list
end

end