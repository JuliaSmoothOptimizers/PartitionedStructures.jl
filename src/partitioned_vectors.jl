module PartitionedVectors
using LinearAlgebra

using ..M_elt_vec, ..M_abstract_element_struct
using ..M_part_v, ..M_abstract_part_struct
using ..ModElemental_pv, ..ModElemental_ev

import Base: +, -, ==, *
import Base: copy, similar
import Base: show, size, length, getindex, setindex!, firstindex, lastindex

import LinearAlgebra: norm, dot, axpy!, axpby!

export PartitionedVector
export build!

abstract type AbstractPartitionedVector{T} <: DenseVector{T} end # for Krylov 

# mutable struct PartitionedVector{T} <: AbstractVector{T}
mutable struct PartitionedVector{T} <: AbstractPartitionedVector{T}
  epv::Elemental_pv{T}
  vec::Vector{T}
  simulate_vector::Bool
end

function PartitionedVector(eevar::Vector{Vector{Int}}; T::DataType=Float64, simulate_vector::Bool=false)
  epv = create_epv(eevar; type=T)
  vec = Vector{T}(undef, get_n(epv))
  pv = PartitionedVector{T}(epv, vec,simulate_vector)
  return pv
end 

function PartitionedVector(epv::Elemental_pv{T}; simulate_vector::Bool=false) where T
  vec = Vector{T}(undef, get_n(epv))
  pv = PartitionedVector{T}(epv, vec, simulate_vector)
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

getindex(pv::PartitionedVector, inds...) = get_eev_set(pv.epv)[inds...]

function setindex!(pv::PartitionedVector, eev::Elemental_elt_vec, index::Int)
  # println("setindex")
  # println(eev)
  get_eev_value(pv.epv, index) .= get_vec(eev)
  return pv
end 

function setindex!(pv::PartitionedVector{T}, val::T, index::Int) where T<:Number
  get_eev_value(pv.epv, index) .= val
  get_v(pv.epv) .= val
  return pv
end 

firstindex(pv::PartitionedVector) = get_N(pv.epv) > 0 ? 1 : 0
lastindex(pv::PartitionedVector) = get_N(pv.epv)

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

function (*)(pv::PartitionedVector{Y}, val::T) where {Y<:Number, T<:Number}
  epv = pv.epv
  _epv = (*)(epv, val)
  return PartitionedVector(_epv)
end

(*)(val::T, pv::PartitionedVector{Y}) where {Y<:Number, T<:Number} = (*)(pv, val)

function (==)(pv1::PartitionedVector, pv2::PartitionedVector)
  epv1 = pv1.epv
  epv2 = pv2.epv
  return (==)(epv1, epv2)
end

copy(pv::PartitionedVector{T}; simulate_vector::Bool=pv.simulate_vector) where {T <: Number} = PartitionedVector{T}(copy(pv.epv), copy(pv.vec), simulate_vector)
similar(pv::PartitionedVector{T}; simulate_vector::Bool=pv.simulate_vector) where {T <: Number} = PartitionedVector{T}(similar(pv.epv), similar(pv.vec), simulate_vector)
similar(pv::PartitionedVector{T}, ::Type{T}, n::Int; simulate_vector::Bool=pv.simulate_vector) where {T <: Number} = PartitionedVector{T}(similar(pv.epv), similar(pv.vec), simulate_vector)

build!(pv::PartitionedVector) = build!(pv, Val(pv.simulate_vector))

function build!(pv::PartitionedVector, ::Val{true}) 
  epv = pv.epv
  vec = epv.v
  component_list = epv.component_list
  for i in 1:length(vec)
    index_element = component_list[i][1]
    eev = get_eev_set(epv, index_element)
    val = get_vec_from_indices(eev, i)
    vec[i] = val
  end
  return pv
end

build!(pv::PartitionedVector, ::Val{false}) = build_v!(pv.epv)

function norm(pv::PartitionedVector, p::Real=2)
  build!(pv)
  _norm = norm(get_v(pv.epv), p)
  return _norm
end

function dot(pv1::PartitionedVector{T}, pv2::PartitionedVector{T}) where T
  build!(pv1)
  build!(pv2)
  dot(get_v(pv1.epv), get_v(pv2.epv))
end

function axpy!(s::Y, x::PartitionedVector{T}, y::PartitionedVector{T}) where {T<:Number,Y<:Number}
  axpy!(s,x,y,Val(x.simulate_vector), Val(y.simulate_vector))
end

function axpy!(s::Y, x::PartitionedVector{T}, y::PartitionedVector{T}, ::Val{true}, ::Val{false}) where {T<:Number,Y<:Number}
  build!(x)
  build!(y)
  xvector = x.epv.v
  yvector = y.epv.v
  epv_from_v!(y.epv, s .* xvector .+ yvector)
  return y
end

function axpy!(s::Y, x::PartitionedVector{T}, y::PartitionedVector{T}, ::Val{true}, ::Val{true}) where {T<:Number,Y<:Number}
  y .+= s .* x
  return y
end

function axpy!(s::Y, x::PartitionedVector{T}, y::PartitionedVector{T}, ::Val{false}, ::Val{true}) where {T<:Number,Y<:Number}
  build!(x)
  build!(y)
  xvector = x.epv.v
  yvector = y.epv.v
  epv_from_v!(y.epv, s .* xvector .+ yvector)
  return y
  y .+= s .* x
  return y
end

function axpby!(s::Y1, x::PartitionedVector{T}, t::Y2, y::PartitionedVector{T}) where {T<:Number,Y1<:Number,Y2<:Number}
  axpby!(s, x, t, y, Val(x.simulate_vector), Val(y.simulate_vector))
end

function axpby!(s::Y1, x::PartitionedVector{T}, t::Y2, y::PartitionedVector{T}, ::Val{false}, ::Val{true}) where {T<:Number,Y1<:Number,Y2<:Number}
  build!(x)
  build!(y)
  xvector = x.epv.v
  yvector = y.epv.v
  epv_from_v!(y.epv, s .* xvector .+ yvector .* t)
end

function axpby!(s::Y1, x::PartitionedVector{T}, t::Y2, y::PartitionedVector{T}, ::Val{true}, ::Val{true}) where {T<:Number,Y1<:Number,Y2<:Number}
  y .= x .* s .+ y .* t
end


Base.broadcastable(pv::PartitionedVector) = pv

struct PartitionedVectorStyle <: Base.Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{PartitionedVector{T}}) where T = PartitionedVectorStyle()

Base.BroadcastStyle(::PartitionedVectorStyle, ::PartitionedVectorStyle) = PartitionedVectorStyle()
Base.BroadcastStyle(::PartitionedVectorStyle, ::Base.Broadcast.BroadcastStyle) = PartitionedVectorStyle()
Base.BroadcastStyle(::Base.Broadcast.BroadcastStyle, ::PartitionedVectorStyle) = PartitionedVectorStyle()

function Base.similar(bc::Base.Broadcast.Broadcasted{PartitionedVectorStyle}, ::Type{ElType}) where ElType
  # println("similar")
  pv = find_pv(bc)
  pvres = similar(pv)
  return pvres
end

function Base.copy(bc::Base.Broadcast.Broadcasted{PartitionedVectorStyle})
  # println("copy")
  pv = find_pv(bc)
  pvres = copy(pv)
  return pvres
end

find_pv(bc::Base.Broadcast.Broadcasted) = find_pv(bc.args)
find_pv(args::Tuple) = find_pv(find_pv(args[1]), Base.tail(args))
find_pv(x) = x
find_pv(::Tuple{}) = nothing
find_pv(a::PartitionedVector, rest) = a
find_pv(::Any, rest) = find_pv(rest)

function Base.Vector(pv::PartitionedVector{T}) where T
  build!(pv)
  vector = pv.epv.v  
  return Vector{T}(copy(vector))
end

end