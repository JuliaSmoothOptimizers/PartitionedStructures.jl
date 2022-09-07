module PartitionedVectors
using LinearAlgebra

using ..M_elt_vec, ..M_abstract_element_struct
using ..M_part_v, ..M_abstract_part_struct
using ..ModElemental_pv, ..ModElemental_ev

import Base: +, -, ==
import Base: copy, similar
import Base: show, size, length, getindex, setindex!, firstindex, lastindex

import LinearAlgebra: norm, dot, axpy!

export PartitionedVector
export build!

# mutable struct PartitionedVector{T} <: AbstractVector{T}
mutable struct PartitionedVector{T} <: DenseVector{T} # for Krylov
  epv::Elemental_pv{T}
  vec::Vector{T}
end

function PartitionedVector(eevar::Vector{Vector{Int}}; T::DataType=Float64)
  epv = create_epv(eevar; type=T)
  vec = Vector{T}(undef, get_n(epv))
  pv = PartitionedVector{T}(epv, vec)
  return pv
end 

function PartitionedVector(epv::Elemental_pv{T}) where T
  vec = Vector{T}(undef, get_n(epv))
  pv = PartitionedVector{T}(epv, vec)
  return pv
end 

length(pv::PartitionedVector) = get_n(pv.epv)
size(pv::PartitionedVector) = (get_N(pv.epv),)

show(pv::PartitionedVector) = show(stdout, pv)
show(io::IO, ::MIME"text/plain", pv::PartitionedVector) = show(io, pv)

function show(io::IO, pv::PartitionedVector)
  println(io, typeof(pv))
  show(io, get_v(pv.epv))
  return nothing
end

# function broadcast!(f::Function, pv::PartitionedVector, As...)
#   broadcast!(f, pv.epv, As...)
#   return pv
# end

# function broadcast!(f::Function, pv1::PartitionedVector, pv2::PartitionedVector, As...)
#   epv1 = pv1.epv
#   epv2 = pv2.epv
#   broadcast!(f, epv1, epv2, As...)
#   return pv
# end

# function setindex!(vec::AbstractVector, pv::PartitionedVector, inds...)
#   setindex!(vec, pv.epv, inds...)  
#   return vec
# end


getindex(pv::PartitionedVector, inds...) = get_eev_set(pv.epv)[inds...]

function setindex!(pv::PartitionedVector, eev::Elemental_elt_vec, index::Int)
  get_eev_set(pv.epv)[index] = copy(eev)
  return pv
end 

function setindex!(pv::PartitionedVector{T}, val::T, index::Int) where T
  get_eev_value(pv.epv, index) .= val
  get_v(pv.epv) .= val
  return pv
end 

firstindex(pv::PartitionedVector) = get_N(pv.epv) > 0 ? 1 : 0
lastindex(pv::PartitionedVector) = get_N(pv.epv)

function Broadcast.broadcasted(::typeof(+), pv1::PartitionedVector, pv2::PartitionedVector)
  epv1 = pv1.epv
  epv2 = pv2.epv
  _epv = (+)(epv1, epv2)
  return PartitionedVector(_epv)
end

function Broadcast.broadcasted(::typeof(-), pv::PartitionedVector)
  epv = pv.epv  
  _epv = (-)(epv)
  return PartitionedVector(_epv)
end

function Broadcast.broadcasted(::typeof(-), pv1::PartitionedVector, pv2::PartitionedVector)
  epv1 = pv1.epv
  epv2 = pv2.epv
  _epv = (-)(epv1, epv2)
  return PartitionedVector(_epv)
end

function Broadcast.broadcasted(::typeof(*), pv::PartitionedVector, val::Y) where Y
  epv = pv.epv
  _epv = (*)(epv, val)
  return PartitionedVector(_epv)
end

function (+)(pv1::PartitionedVector, pv2::PartitionedVector)
  epv1 = pv1.epv
  epv2 = pv2.epv
  _epv = (+)(epv1, epv2)
  return PartitionedVector(_epv)
end

function (-)(pv1::PartitionedVector, pv2::PartitionedVector)
  epv1 = pv1.epv
  epv2 = pv2.epv
  _epv = (-)(epv1, epv2)
  return PartitionedVector(_epv)
end

function (-)(pv::PartitionedVector)
  epv = pv.epv
  _epv = (-)(epv)
  return PartitionedVector(_epv)
end

function (==)(pv1::PartitionedVector, pv2::PartitionedVector)
  epv1 = pv1.epv
  epv2 = pv2.epv
  return (==)(epv1, epv2)
end

copy(pv::PartitionedVector{T}) where {T <: Number} = PartitionedVector{T}(copy(pv.epv), copy(pv.vec))
similar(pv::PartitionedVector{T}) where {T <: Number} = PartitionedVector{T}(similar(pv.epv), similar(pv.vec))
similar(pv::PartitionedVector{T}, ::Type{T}, n::Int) where {T <: Number} = PartitionedVector{T}(similar(pv.epv), similar(pv.vec))

build!(pv::PartitionedVector) = build_v!(pv.epv)

function norm(pv::PartitionedVector, p::Real=2)
  pv.vec .= get_v(pv.epv)
  build!(pv)
  _norm = norm(get_v(pv.epv), p)
  get_v(pv.epv).= pv.vec
  return _norm
end

function dot(pv1::PartitionedVector{T}, pv2::PartitionedVector{T}) where T
  dot(get_v(pv1.epv), get_v(pv2.epv))
end

function axpy!(s::Y, x::PartitionedVector{T}, y::PartitionedVector{T}) where {T<:Number,Y<:Number}
  y .+= s .* x
  return y
end

end