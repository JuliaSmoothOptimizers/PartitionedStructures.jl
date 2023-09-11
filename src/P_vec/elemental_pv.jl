module ModElemental_pv
using LinearAlgebra, SparseArrays

using ..Acronyms
using ..Utils
using ..M_abstract_element_struct, ..M_elt_vec, ..ModElemental_ev # element modules
using ..M_abstract_part_struct, ..M_part_v  # partitoned modules

import Base.Vector
import Base.==, Base.similar, Base.copy
import Base: +, -, *
import Base.broadcast!, Base.setindex!
import ..M_abstract_part_struct: initialize_component_list!, get_ee_struct

export Elemental_pv
export get_eev_set, get_eev, get_eev_value, get_eev_subset
export set_eev!, minus_epv!, add_epv!, set_epv!
export create_epv, ones_kchained_epv, part_vec, rand_epv
export scale_epv, scale_epv!
export epv_from_epv!, epv_from_v, epv_from_v!
export prod_part_vectors

"""
    Elemental_pv{T} <: Part_v{T}

Represent an elemental partitioned-vector.
"""
mutable struct Elemental_pv{T} <: Part_v{T}
  N::Int
  n::Int
  eev_set::Vector{Elemental_elt_vec{T}}
  v::Vector{T}
  component_list::Vector{Vector{Int}}
  permutation::Vector{Int} # n-size vector
end

function Elemental_pv{T}(
  N::Int,
  n::Int,
  eev_set::Vector{Elemental_elt_vec{T}},
  v::Vector{T};
  perm::Vector{Int} = [1:n;],
) where {T}
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  epv = Elemental_pv{T}(N, n, eev_set, v, component_list, perm)
  initialize_component_list!(epv)
  return epv
end

"""
    eev_set = get_eev_set(epv::Elemental_pv{T}) where T

Return either the vector of every elemental element-vector of the $(_epv) or the `i`-th elemental element-vector.
"""
@inline get_eev_set(epv::Elemental_pv{T}) where {T} = epv.eev_set
@inline get_eev_set(epv::Elemental_pv{T}, i::Int) where {T} = epv.eev_set[i]

# docstring defined in M_abstract_part_struct.get_ee_struct
@inline get_ee_struct(epv::Elemental_pv{T}) where {T} = get_eev_set(epv)
@inline get_ee_struct(epv::Elemental_pv{T}, i::Int) where {T} = get_eev_set(epv, i)

"""
    eev_subset = get_eev_subset(epv::Elemental_pv{T}, indices::Vector{Int}) where T

Return a subset of the elemental element vector composing the $(_epv).
`indices` selects the differents elemental element-vector needed.
"""
@inline get_eev_subset(epv::Elemental_pv{T}, indices::Vector{Int}) where {T} = epv.eev_set[indices]

"""
    eev_i_value = get_eev_value(epv::Elemental_pv{T}, i::Int) where T
    eev_ij_value = get_eev_value(epv::Elemental_pv{T}, i::Int, j::Int) where T

Return either the value of the `i`-th elemental element-vector of the $(_epv) or only the `j`-th component of the `i`-th elemental element-vector.
"""
@inline get_eev_value(epv::Elemental_pv{T}, i::Int) where {T} = get_vec(get_eev_set(epv, i))
@inline get_eev_value(epv::Elemental_pv{T}, i::Int, j::Int) where {T} =
  get_vec(get_eev_set(epv, i))[j]

"""
    set_eev!(epv::Elemental_pv{T}, i::Int, vec::Vector{T}) where T
    set_eev!(epv::Elemental_pv{T}, i::Int, j::Int, val:: T) where T

Set either the `i`-th elemental element-vector `epv` to `vec` or its `j`-th component to `val`.
"""
@inline set_eev!(epv::Elemental_pv{T}, i::Int, j::Int, val::T) where {T} =
  set_vec!(get_eev_set(epv, i), j, val)
@inline set_eev!(epv::Elemental_pv{T}, i::Int, vec::Vector{T}) where {T} =
  set_vec!(get_eev_set(epv, i), vec)

@inline (==)(ep1::Elemental_pv{T}, ep2::Elemental_pv{T}) where {T} =
  (get_N(ep1) == get_N(ep2)) && (get_n(ep1) == get_n(ep2)) && (get_eev_set(ep1) == get_eev_set(ep2))
@inline similar(epv::Elemental_pv{T}) where {T} =
  Elemental_pv{T}(get_N(epv), get_n(epv), similar.(get_eev_set(epv)), Vector{T}(undef, get_n(epv)))
@inline copy(epv::Elemental_pv{T}) where {T} =
  Elemental_pv{T}(get_N(epv), get_n(epv), copy.(get_eev_set(epv)), Vector{T}(get_v(epv)))

"""
    build_v!(epv::Elemental_pv{T}) where T

Build the vector `epv.v` by accumulating the contribution of each elemental element-vector.
"""
function M_part_v.build_v!(epv::Elemental_pv{T}) where {T}
  reset_v!(epv)
  N = get_N(epv)
  for i = 1:N
    eevᵢ = get_eev_set(epv, i)
    nᵢᴱ = get_nie(eevᵢ)
    for j = 1:nᵢᴱ
      add_v!(epv, get_indices(eevᵢ, j), get_vec(eevᵢ, j))
    end
  end
  return epv
end

"""
    minus_epv!(epv::Elemental_pv{T}) where T <: Number

Build in place the `-epv`, by inversing the value of each elemental element-vector.
"""
minus_epv!(epv::Elemental_pv{T}) where {T <: Number} =
  map((eev -> set_minus_vec!(eev)), get_eev_set(epv))

function (-)(epv::Elemental_pv)
  _epv = copy(epv)
  minus_epv!(_epv)
  return _epv
end

function (-)(epv1::Elemental_pv, epv2::Elemental_pv)
  _epv = -epv2
  add_epv!(epv1, _epv)
  return _epv
end

"""
    add_epv!(epv1::Elemental_pv{T}, epv2::Elemental_pv{T})

Build in place of `epv2` the elementwise addition of `epv1` and `epv2`.
"""
function add_epv!(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where {T <: Number}
  full_check_epv_epm(epv1, epv2) || @error("epv1 mismatch epv2 in add_epv!")
  N = get_N(epv1)
  for i = 1:N
    vec1 = get_vec(get_eev_set(epv1, i))
    set_add_vec!(get_eev_set(epv2, i), vec1)
  end
  return epv2
end

function (+)(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where {T}
  _epv = copy(epv1)::Elemental_pv{T}
  add_epv!(epv2, _epv)
  return _epv
end

(*)(val::Y, epv::Elemental_pv{T}) where {T, Y} = (*)(epv, val)

function (*)(epv::Elemental_pv{T}, val::Y) where {T, Y}
  _epv = copy(epv)::Elemental_pv{T}
  N = get_N(_epv)
  for i = 1:N
    get_eev_value(_epv, i) .= get_eev_value(_epv, i) .* val
  end
  return _epv
end

"""
    epv = create_epv(sp_set::Vector{SparseVector{T,Y}}; kwargs...) where {T,Y}
    epv = create_epv(eev_set::Vector{Elemental_elt_vec{T}}; n=max_indices(eev_set)) where T

Create an elemental partitioned-vector from a vector `eev_set` of: `SparseVector`, elemental element-vector or a vector of indices.
"""
@inline create_epv(sp_set::Vector{SparseVector{T, Y}}; kwargs...) where {T, Y} =
  create_epv(eev_from_sparse_vec.(sp_set); kwargs...)
@inline create_epv(
  vec_elt_var::Vector{Vector{Int}};
  n::Int = max_indices(vec_elt_var),
  type = Float64,
) = create_epv(vec_elt_var, n; type)

function create_epv(
  eev_set::Vector{Elemental_elt_vec{T}};
  n = max_indices(eev_set),
) where {T <: Number}
  N = (n != 0) ? length(eev_set) : 0
  v = zeros(T, n)
  return Elemental_pv{T}(N, n, eev_set, v)
end

function create_epv(vec_elt_var::Vector{Vector{Int}}, n::Int; type = Float64)
  eev_set =
    map((elt_var -> create_eev(elt_var; type)), vec_elt_var)::Vector{Elemental_elt_vec{type}}
  epv = create_epv(eev_set; n = n)
  return epv
end

"""
    set_epv!(epv::Elemental_pv{T}, vec_value_eev::Vector{Vector{T}}) where T <: Number

Set the values of the elemental element-vectors of `epv` with the components of `vec_value_eev`.
"""
function set_epv!(epv::Elemental_pv{T}, vec_value_eev::Vector{Vector{T}}) where {T <: Number}
  N = get_N(epv)
  length(vec_value_eev) == N ||
    @error("The size of vec_value_eev does not match the size of get_N(epv)")
  map(i -> set_eev!(epv, i, vec_value_eev[i]), 1:N)
  return epv
end

"""
    v = scale_epv(epv::Elemental_pv{T}, scalars::Vector{T}) where T <: Number

Return a vector `v` from `epv` where the contribution of each element-vector is multiply by the corresponding value from `scalars`.
"""
function scale_epv(epv::Elemental_pv{T}, scalars::Vector{T}) where {T <: Number}
  _tmp_v = get_v(epv)
  res_v = scale_epv!(epv, scalars)
  set_v!(epv, _tmp_v)
  return res_v
end

"""
    v = scale_epv!(epv::Elemental_pv{T}, scalars::Vector{T}) where T

Return a vector `v` from `epv` where the contribution of each element-vector is multiply by the corresponding value from `scalars`.
`v` is extract from `epv.v`
"""
function scale_epv!(epv::Elemental_pv{T}, scalars::Vector{T}) where {T}
  get_N(epv) == length(scalars) || error("epv, N != length(scalars")
  reset_v!(epv)
  N = get_N(epv)
  for i = 1:N
    eevᵢ = get_eev_set(epv, i)
    nᵢᴱ = get_nie(eevᵢ)
    scalar = scalars[i]
    for j = 1:nᵢᴱ
      add_v!(epv, get_indices(eevᵢ, j), scalar * get_vec(eevᵢ, j))
    end
  end
  return get_v(epv)
end

"""
    epv = rand_epv(N::Int,n::Int; nie=3, T=Float64)

Define an elemental partitioned-vector of `N` elemental element-vector of size `nᵢ` whose values are randoms and the indices are in the range `1:n`.
"""
function rand_epv(N::Int, n::Int; nie = 3, T = Float64)
  eev_set = [new_eev(nie; T = T, n = n) for i = 1:N]
  v = zeros(T, n)
  return Elemental_pv{T}(N, n, eev_set, v)
end

"""
    epv = ones_kchained_epv(N::Int, k::Int; T=Float64)

Construct an elemental partitioned-vector of `N` elemental element-vector of size `k` which overlaps the next element-vector on `k-1` variables.
"""
function ones_kchained_epv(N::Int, k::Int; T = Float64)
  n = N + k
  nᵢ = k
  eev_set = [ones_eev(nᵢ; T = T, n = n) for i = 1:N]
  v = zeros(T, n)
  return Elemental_pv{T}(N, n, eev_set, v)
end

"""
    epv = part_vec(;n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1, mul::Float64=1.)

Define an elemental partitioned-vector formed by `N` (deduced from `n` and `nie`) elemental element-vectors of size `nie`.
Each elemental element-vector overlaps the previous and the next element by `overlapping`.
"""
function part_vec(; n::Int = 9, T = Float64, nie::Int = 5, overlapping::Int = 1, mul::Float64 = 1.0)
  overlapping < nie || error("the overlapping must be lower than nie")
  mod(n - (nie - overlapping), nie - overlapping) == mod(overlapping, nie - overlapping) ||
    error("wrong structure: mod(n-(nie-over), nie-over)==mod(over, nie-over) must holds")
  indices = filter(
    x -> x <= n - nie + 1,
    vcat(1, (x -> x + (nie - overlapping)).([1:(nie - overlapping):(n - (nie - overlapping));])),
  )
  eev_set = map(i -> specific_ones_eev(nie, i; T = T, mul = mul), indices)
  N = length(eev_set)
  v = Vector{T}(undef, n)
  epv = Elemental_pv{T}(N, n, eev_set, v)
  return epv
end

function Base.Vector(epv::Elemental_pv{T}) where {T}
  build_v!(epv)
  return get_v(epv)
end

"""
    epv = epv_from_v(x::Vector{T}, shape_epv::Elemental_pv{T}) where T

Define a new elemental partitioned-vector from `x` that have the same structure than `shape_epv`.
The value of each elemental element-vector comes from the corresponding indices of `x`.
Usefull to define Uᵢ x, ∀ x.
"""
function epv_from_v(x::Vector{T}, shape_epv::Elemental_pv{T}) where {T}
  epv_x = similar(shape_epv)
  epv_from_v!(epv_x, x)
  return epv_x
end

"""
    epv_from_v!(epv_x::Elemental_pv{T}, x::Vector{T}) where T

Set the values of the element partitioned-vector `epv` to `x`.
Usefull to define Uᵢ x, ∀ i ∈ {1,...,N}.
"""
function epv_from_v!(epv_x::Elemental_pv{T}, x::Vector{T}) where {T}
  @inbounds @simd for idx = 1:get_N(epv_x)
    indices = get_indices(get_eev_set(epv_x, idx))
    vec = get_eev_value(epv_x, idx)
    _view_x = view(x, indices)
    # vec .= _view_x
    mul!(vec, I, _view_x, 1, 0)
  end
  return epv_x
end

"""
    epv_from_epv!(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where T

Set the elemental partitioned-vector `epv1` to `epv2`.
"""
function epv_from_epv!(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where {T}
  full_check_epv_epm(epv1, epv2) ||
    @error("different partitioned structures between eplo_B and epv_y")
  for idx = 1:get_N(epv1)
    set_eev!(epv1, idx, get_eev_value(epv2, idx))
  end
  return epv1
end

# docstring in M_abstract_part_struct.initialize_component_list!
function initialize_component_list!(epv::Elemental_pv)
  N = get_N(epv)
  for i = 1:N
    epvᵢ = get_eev_set(epv, i)
    _indices = get_indices(epvᵢ)
    for j in _indices # changer peut-être
      push!(get_component_list(epv, j), i)
    end
  end
  return epv
end

"""
    (acc, res) = prod_part_vectors(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where T

Perform an elementwise scalar product between the two elemental partitioned-vector `epv1` and `epv2`.
`acc` accumulates the sum of the element-vectors scalar product.
`res` contrains the details of every element-vector scalar product.
"""
function prod_part_vectors(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where {T}
  full_check_epv_epm(epv1, epv2) ||
    @error("different partitioned structures between eplo_B and epv_y")
  N = get_N(epv1)
  acc = (T)(0)
  res = Vector{T}(undef, N)
  for idx = 1:N
    eev1 = get_eev_set(epv1, idx)
    eev2 = get_eev_set(epv2, idx)

    vec1 = get_vec(eev1)
    vec2 = get_vec(eev2)

    yts = dot(vec1, vec2)
    res[idx] = yts
    acc += yts
  end
  return acc, res
end

end
