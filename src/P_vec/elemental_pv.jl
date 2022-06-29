module ModElemental_pv
using SparseArrays

using ..Utils
using ..M_abstract_element_struct, ..M_elt_vec, ..ModElemental_ev # element modules
using ..M_abstract_part_struct, ..M_part_v  # partitoned modules

import Base.Vector
import Base.==, Base.similar, Base.copy
import ..M_abstract_part_struct: initialize_component_list!, get_ee_struct

export Elemental_pv
export get_eev_set, get_eev, get_eev_value, get_eev_subset
export set_eev!, minus_epv!, add_epv!
export set_epv!
export create_epv, ones_kchained_epv, part_vec, rand_epv
export scale_epv, scale_epv!
export epv_from_epv!, epv_from_v, epv_from_v!
export prod_part_vectors

"""
    Elemental_pv{T}<:Part_v{T}

Type that represents an elemental partitioned-vector.
"""
mutable struct Elemental_pv{T}<:Part_v{T}
  N::Int
  n::Int
  eev_set::Vector{Elemental_elt_vec{T}}
  v::Vector{T}
  component_list::Vector{Vector{Int}}
  permutation::Vector{Int} # n-size vector
end
function Elemental_pv{T}(N::Int, n::Int, eev_set::Vector{Elemental_elt_vec{T}}, v::Vector{T}; perm::Vector{Int}=[1:n;]) where T
  component_list = map(i -> Vector{Int}(undef,0), [1:n;])
  epv = Elemental_pv{T}(N,n,eev_set,v,component_list,perm)
  initialize_component_list!(epv)
  return epv
end

@inline get_eev_set(pv::Elemental_pv{T}) where T = pv.eev_set
@inline get_eev(pv::Elemental_pv{T}, i::Int) where T = pv.eev_set[i]
@inline get_ee_struct(pv::Elemental_pv{T}) where T = get_eev_set(pv)
@inline get_ee_struct(pv::Elemental_pv{T}, i::Int) where T = get_eev(pv,i)
@inline get_eev_subset(pv::Elemental_pv{T}, indices::Vector{Int}) where T = pv.eev_set[indices]
@inline get_eev_value(pv::Elemental_pv{T}, i::Int) where T = get_vec(get_eev(pv,i))
@inline get_eev_value(pv::Elemental_pv{T}, i::Int, j::Int) where T = get_vec(get_eev(pv,i))[j]
@inline set_eev!(pv::Elemental_pv{T}, i::Int, j::Int, val:: T) where T = set_vec!(get_eev(pv,i),j,val)
@inline set_eev!(pv::Elemental_pv{T}, i::Int, vec::Vector{T}) where T = set_vec!(get_eev(pv,i),vec)

@inline (==)(ep1::Elemental_pv{T},ep2::Elemental_pv{T}) where T = (get_N(ep1)==get_N(ep2)) && (get_n(ep1)==get_n(ep2)) && (get_eev_set(ep1)==get_eev_set(ep2))
@inline similar(ep::Elemental_pv{T}) where T = Elemental_pv{T}(get_N(ep), get_n(ep), similar.(get_eev_set(ep)), Vector{T}(undef,get_n(ep)))
@inline copy(ep::Elemental_pv{T}) where T = Elemental_pv{T}(get_N(ep), get_n(ep), copy.(get_eev_set(ep)), Vector{T}(get_v(ep)))

"""
    build_v!(epv::Elemental_pv{T}) where T

Build the vector `epv.v` by accumulating the contribution of each elemental element-vector.
"""
function M_part_v.build_v!(epv::Elemental_pv{T}) where T
  reset_v!(epv)
  N = get_N(epv)
  for i in 1:N
    eevᵢ = get_eev(epv,i)
    nᵢᴱ = get_nie(eevᵢ)
    for j in 1:nᵢᴱ
      add_v!(epv, get_indices(eevᵢ,j), get_vec(eevᵢ,j))
    end
  end
end

"""
    minus_epv!(epv::Elemental_pv{T}) where T<:Number

Build in place the `-epv`, by inversing the value of each elemental element-vector.
"""
minus_epv!(epv::Elemental_pv{T}) where T<:Number = map( (eev -> set_minus_vec!(eev)), get_eev_set(epv))

"""
    add_epv!(epv1::Elemental_pv{T}, epv2::Elemental_pv{T})

Build in place of `epv2` the elementwise addition of `epv1` and `epv2`.
"""
function add_epv!(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where T<:Number
  full_check_epv_epm(epv1,epv2) || @error("epv1 mismatch epv2 in add_epv!")
  N = get_N(epv1)
  for i in 1:N
    vec1 = get_vec(get_eev(epv1,i))
    set_add_vec!(get_eev(epv2,i), vec1)
  end
  return epv2
end

"""
    epv = create_epv(sp_set::Vector{SparseVector{T,Y}}; kwargs...) where {T,Y}
    epv = create_epv(eev_set::Vector{Elemental_elt_vec{T}}; n=max_indices(eev_set)) where T

Create an elemental partitioned-vector from a vector `eev_set` of: `SparseVector`, elemental element-vector or a vector of indices.
"""
@inline create_epv(sp_set::Vector{SparseVector{T,Y}}; kwargs...) where {T,Y} = create_epv(eev_from_sparse_vec.(sp_set); kwargs...)

function create_epv(eev_set::Vector{Elemental_elt_vec{T}}; n=max_indices(eev_set)) where T
  N = length(eev_set)
  v = zeros(T,n)
  Elemental_pv{T}(N, n, eev_set, v)
end

@inline create_epv(vec_elt_var::Vector{Vector{Int}}; n::Int=max_indices(vec_elt_var), type=Float64) = create_epv(vec_elt_var, n; type)

function create_epv(vec_elt_var::Vector{Vector{Int}}, n::Int; type=Float64)
  eev_set = map((elt_var -> create_eev(elt_var,type=type)), vec_elt_var)
  epv = create_epv(eev_set; n=n)
  return epv
end

"""
    set_epv!(epv::Elemental_pv{T}, vec_value_eev::Vector{Vector{T}}) where T<:Number

Set the values of the elemental element-vectors of `epv` with the components of `vec_value_eev`.
"""
function set_epv!(epv::Elemental_pv{T}, vec_value_eev::Vector{Vector{T}}) where T<:Number
  N = get_N(epv)
  length(vec_value_eev)==N || @error("The size of vec_value_eev does not match the size of get_N(epv)")
  map(i -> set_eev!(epv, i, vec_value_eev[i]), 1:N)
  return epv
end

"""
    v = scale_epv(epv::Elemental_pv{T}, scalars::Vector{T}) where T<:Number

Return a vector `v` from `epv` where the contribution of each element-vector is multiply by the corresponding value from `scalars`.
"""
function scale_epv(epv::Elemental_pv{T}, scalars::Vector{T}) where T<:Number
  _tmp_v = get_v(epv)
  res_v = scale_epv!(epv, scalars)
  set_v!(epv,_tmp_v)
  return res_v
end

"""
    v = scale_epv!(epv::Elemental_pv{T}, scalars::Vector{T}) where T

Return a vector `v` from `epv` where the contribution of each element-vector is multiply by the corresponding value from `scalars`.
`v` is extract from `epv.v`
"""
function scale_epv!(epv::Elemental_pv{T}, scalars::Vector{T}) where T
  get_N(epv)==length(scalars) || error("epv, N != length(scalars")
  reset_v!(epv)
  N = get_N(epv)
  for i in 1:N
    eevᵢ = get_eev(epv,i)
    nᵢᴱ = get_nie(eevᵢ)
    scalar = scalars[i]
    for j in 1:nᵢᴱ
      add_v!(epv, get_indices(eevᵢ,j), scalar*get_vec(eevᵢ,j))
    end
  end
  return get_v(epv)
end

"""
    epv = rand_epv(N::Int,n::Int; nie=3, T=Float64)

Define an elemental partitioned-vector of `N` elemental element-vector of size `nᵢ` whose values are randoms and the indices are in the range `1:n`.
"""
function rand_epv(N::Int,n::Int; nie=3, T=Float64)
  eev_set = [new_eev(nie;T=T,n=n) for i in 1:N]
  v = zeros(T,n)
  return Elemental_pv{T}(N, n, eev_set, v)
end

"""
    epv = ones_kchained_epv(N::Int, k::Int; T=Float64)

Construct an elemental partitioned-vector of `N` elemental element-vector of size `k` which overlaps the next element-vector on `k-1` variables.
"""
function ones_kchained_epv(N::Int, k::Int; T=Float64)
  n = N+k
  nᵢ = k
  eev_set = [ones_eev(nᵢ;T=T,n=n) for i in 1:N]
  v = zeros(T,n)
  return Elemental_pv{T}(N, n, eev_set, v)
end

"""
    epv = part_vec(;n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1, mul::Float64=1.)

Define a elemental partitioned-vector formed by `N` (deduced from `n` and `nie`) elemental element-vectors of size `nie`.
Each elemental element-vector overlaps the previous and the next element by `overlapping`.
"""
function part_vec(;n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1, mul::Float64=1.)
  overlapping < nie || error("the overlapping must be lower than nie")
  mod(n-(nie-overlapping), nie-overlapping)==mod(overlapping, nie-overlapping) || error("wrong structure: mod(n-(nie-over), nie-over)==mod(over, nie-over) must holds")
  indices = filter(x -> x <= n-nie+1, vcat(1,(x -> x + (nie-overlapping)).([1:nie-overlapping:n-(nie-overlapping);])))
  eev_set = map(i -> specific_ones_eev(nie,i;T=T, mul=mul), indices)
  N = length(eev_set)
  v = Vector{T}(undef,n)
  epv = Elemental_pv{T}(N,n,eev_set,v)
  return epv
end

function Base.Vector(pv::Elemental_pv{T}) where T
	build_v!(pv)
	get_v(pv)
end

"""
    epv = epv_from_v(x::Vector{T}, shape_epv::Elemental_pv{T}) where T

Define a new elemental partitioned-vector from `x` that have the same structure than epv.
The value of each elemental element-vector comes from the corresponding indices of `x`.
Usefull to define Uᵢ x, ∀ x.
"""
function epv_from_v(x::Vector{T}, shape_epv::Elemental_pv{T}) where T
  epv_x = similar(shape_epv)
  epv_from_v!(epv_x, x)
  return epv_x
end

"""
    epv_from_v!(epv_x::Elemental_pv{T}, x::Vector{T}) where T

Set the values of the element partitioned-vector `epv` to `x`.
Usefull to define Uᵢ x, ∀ x.
"""
function epv_from_v!(epv_x::Elemental_pv{T}, x::Vector{T}) where T
  for idx in 1:get_N(epv_x)
    set_eev!(epv_x, idx, x[get_indices(get_eev(epv_x,idx))]) # met le vecteur élément comme une copie de x
  end
end

"""
    epv_from_epv!(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where T

Set the elemental partitioned-vector `epv1` to `epv2`.
"""
function epv_from_epv!(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where T
  full_check_epv_epm(epv1,epv2) || @error("different partitioned structures between eplom_B and epv_y")
  for idx in 1:get_N(epv1)
    set_eev!(epv1, idx, get_eev_value(epv2, idx))
  end
end

"""
    initialize_component_list!(epm)

Build for each index i (∈ {1,...,n}) the list of the elements using the variables `i`.
"""
function initialize_component_list!(epv::Elemental_pv)
  N = get_N(epv)
  n = get_n(epv)
  for i in 1:N
    epvᵢ = get_eev(epv,i)
    _indices = get_indices(epvᵢ)
    for j in _indices # changer peut-être
      push!(get_component_list(epv,j),i)
    end
  end
end

"""
    (acc, res) = prod_part_vectors(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where T

Perform an elementwise scalar product between the two elemental partitioned-vector `epv1` and `epv2`.
`acc` accumulates the sum of the element-vectors scalar product.
`res` contrains the details of every element-vector scalar product.
"""
function prod_part_vectors(epv1::Elemental_pv{T}, epv2::Elemental_pv{T}) where T
  full_check_epv_epm(epv1,epv2) || @error("different partitioned structures between eplom_B and epv_y")
  N = get_N(epv1)
  acc = (T)(0)
  res = Vector{T}(undef, N)
  for idx in 1:N
    eev1 = get_eev(epv1,idx)
    eev2 = get_eev(epv2,idx)

    vec1 = get_vec(eev1)
    vec2 = get_vec(eev2)
    yts = dot(vec1, vec2)
    res[idx] = yts
    acc += yts
  end
  return acc, res
end

end