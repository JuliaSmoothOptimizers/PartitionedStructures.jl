module ModElemental_em

using ..Acronyms
using CUDA, LinearAlgebra
using ..M_abstract_element_struct, ..M_elt_mat

import Base.==, Base.copy, Base.permute!, Base.similar

export Elemental_em
export identity_eem, create_id_eem, fixed_ones_eem, ones_eem, one_size_bloc

"""
    Elemental_em{T} <: DenseEltMat{T}

Represent an elemental element-matrix.
It has fields:

* `indices`: indices of elemental variables;
* `nie`: elemental size (`=length(indices)`);
* `Bie::Symmetric{T, Matrix{T}}`: the elemental matrix;
* `counter`: counts how many update the elemental matrix goes through from its allocation;
* `convex`: if `Elemental_em` is by default update with BFGS or SR1;
* `_Bsr`: a vector used during the quasi-Newton update of en elemental matrix.
"""
mutable struct Elemental_em{T, S<:AbstractMatrix{T}, V<:AbstractVector{T}, Z<:AbstractVector{Int}} <: DenseEltMat{T}
  nie::Int # nᵢᴱ
  indices::Z # size nᵢᴱ
  Bie::Symmetric{T, S} # size nᵢᴱ × nᵢᴱ
  counter::Counter_elt_mat
  convex::Bool
  _Bsr::V # size nᵢᴱ
end

@inline (==)(eem1::Elemental_em, eem2::Elemental_em) =
  (get_nie(eem1) == get_nie(eem2)) &&
  (get_Bie(eem1) == get_Bie(eem2)) &&
  (get_indices(eem1) == get_indices(eem2)) &&
  (get_convex(eem1) == get_convex(eem2))
@inline copy(eem::Elemental_em{T,S,V,Z}) where {T,S,V,Z} = Elemental_em{T,S,V,Z}(
  copy(get_nie(eem)),
  copy(get_indices(eem)),
  copy(get_Bie(eem)),
  Counter_elt_mat(),
  copy(get_convex(eem)),
  copy(get_Bsr(eem)),
)
@inline similar(eem::Elemental_em{T,S,V,Z}) where {T,S,V,Z} = Elemental_em{T,S,V,Z}(
  copy(get_nie(eem)),
  copy(get_indices(eem)),
  similar(get_Bie(eem)),
  Counter_elt_mat(),
  copy(get_convex(eem)),
  similar(get_Bsr(eem)),
)

"""
    eem = create_id_eem(elt_var::Vector{Int}; T=Float64)

Create a `nie` identity elemental element-matrix of type `T` based on the vector of the elemental variables `elt_var`.
"""
function create_id_eem(elt_var::Vector{Int}; T = Float64, bool = false, gpu=false)
  nie = length(elt_var)
  Bie = gpu ? CUDA.zeros(T, nie, nie) : zeros(T, nie, nie)  
  [Bie[i, i] = 1 for i = 1:nie]
  counter = Counter_elt_mat()
  _Bsr = (gpu ? CUDA.CuVector : Vector)(Vector{T}(undef, nie))
  elt_var = (gpu ? CUDA.CuVector : Vector)(elt_var)
  S = typeof(Bie)
  V = typeof(_Bsr)
  Z = typeof(elt_var)
  eem = Elemental_em{T,S,V,Z}(nie, elt_var, Symmetric(Bie), counter, bool, _Bsr)
  return eem
end

"""
    eem = identity_eem(nie::Int; T=Float64, n=nie^2)

Return a `nie` identity elemental element-matrix of type `T` from `nie` random indices in the range `1:n`.
"""
function identity_eem(nie::Int; T = Float64, n = nie^2, bool = false, gpu=false)
  indices = rand(1:n, nie)
  Bie = gpu ? CUDA.zeros(T, nie, nie) : zeros(T, nie, nie)  
  [Bie[i, i] = 1 for i = 1:nie]
  counter = Counter_elt_mat()
  _Bsr = (gpu ? CUDA.CuVector : Vector)(Vector{T}(undef, nie))
  elt_var = (gpu ? CUDA.CuVector : Vector)(indices)
  S = typeof(Bie)
  V = typeof(_Bsr)
  Z = typeof(elt_var)  
  eem = Elemental_em{T,S,V,Z}(nie, elt_var, Symmetric(Bie), counter, bool, _Bsr)
  return eem
end

"""
    eem = ones_eem(nie::Int; T=Float64, n=nie^2)

Return a `nie` ones elemental element-matrix of type `T` from `nie` random indices in the range `1:n`.
"""
function ones_eem(nie::Int; T = Float64, n = nie^2, bool = false, gpu=false)
  indices = rand(1:n, nie)
  Bie = gpu ? CUDA.ones(T, nie, nie) : ones(T, nie, nie)  
  counter = Counter_elt_mat()
  _Bsr = (gpu ? CUDA.CuVector : Vector)(Vector{T}(undef, nie))
  elt_var = (gpu ? CUDA.CuVector : Vector)(indices)
  S = typeof(Bie)
  V = typeof(_Bsr)
  Z = typeof(elt_var)
  eem = Elemental_em{T,S,V,Z}(nie, elt_var, Symmetric(Bie), counter, bool, _Bsr)
  return eem
end

"""
    eem = fixed_ones_eem(i::Int, nie::Int; T=Float64, mul=5.)

Create a `nie` elemental element-matrix of type `T` at indices `index:index+nie-1`.
All the components of the element-matrix are set to `1` except the diagonal terms that are set to `mul`.
This method is used to define diagonal dominant element-matrix.
"""
function fixed_ones_eem(i::Int, nie::Int; T = Float64, mul = 5.0, bool = false, gpu=false)
  indices = [i:(i + nie - 1);]  
  Bie = gpu ? CUDA.ones(T, nie, nie) : ones(T, nie, nie)  
  [Bie[i, i] = mul for i = 1:nie]
  counter = Counter_elt_mat()
  _Bsr = (gpu ? CUDA.CuVector : Vector)(Vector{T}(undef, nie))
  elt_var = (gpu ? CUDA.CuVector : Vector)(indices)
  S = typeof(Bie)
  V = typeof(_Bsr)
  Z = typeof(elt_var)
  eem = Elemental_em{T,S,V,Z}(nie, elt_var, Symmetric(Bie), counter, bool, _Bsr)
  return eem
end

"""
    eem = one_size_bloc(index::Int; T=Float64)

Return an elemental element-matrix of type `T` of size one at `index`.
"""
function one_size_bloc(index::Int; T = Float64, bool=false, gpu=false)
  nie = 1
  indices = [index]  
  Bie = gpu ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
  counter = Counter_elt_mat()
  _Bsr = (gpu ? CUDA.CuVector : Vector)(Vector{T}(undef, nie))
  elt_var = (gpu ? CUDA.CuVector : Vector)(indices)
  S = typeof(Bie)
  V = typeof(_Bsr)
  Z = typeof(elt_var)
  eem = Elemental_em{T,S,V,Z}(nie, elt_var, Symmetric(Bie), counter, bool, _Bsr)
  return eem
end  

"""
    permute!(eem::Elemental_em{T}, p::Vector{Int}) where T

Set the indices of the element variables of `eem` to `p`.
Must be use with caution.
"""
function permute!(eem::Elemental_em{T,S,V,Z}, p::Z) where {T,S,V,Z<:AbstractVector{Int}}
  eem.indices .= p
  return eem
end

end
