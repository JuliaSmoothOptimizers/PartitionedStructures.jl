module PartitionedVectors
using LinearAlgebra

using ..M_elt_vec, ..M_abstract_element_struct
using ..M_part_v, ..M_abstract_part_struct
using ..ModElemental_pv

import Base.+, Base.-
import Base.show, Base.size
import Base.copy, Base.similar
import Base.getindex, Base.setindex!
import Base.broadcast!

export PartitionedVector

mutable struct PartitionedVector{T} <: AbstractVector{T}
  epv::Elemental_pv{T}
end

function PartitionedVector(eevar::Vector{Vector{Int}}; T::DataType=Float64)
  epv = create_epv(eevar; type=T)
  pv = PartitionedVector{T}(epv)
  return pv
end 

size(pv::PartitionedVector) = (get_n(pv.epv),)

show(pv::PartitionedVector) = show(stdout, pv)
function Base.show(io::IO, pv::PartitionedVector)
  print(io, "myshow")
  show(io, string(get_v(pv.epv)))
  return nothing
end

getindex(pv::PartitionedVector, inds...) = getindex(get_v(pv.epv), inds...)

function broadcast!(f::Function, pv::PartitionedVector, As...)
  broadcast!(f, pv.epv, As...)
  return pv
end

function broadcast!(f::Function, pv1::PartitionedVector, pv2::PartitionedVector, As...)
  epv1 = pv1.epv
  epv2 = pv2.epv
  broadcast!(f, epv1, epv2, As...)
  return pv
end

function setindex!(pv::PartitionedVector, vec, inds...)
  setindex!(pv.epv, vec, inds...)    
  return pv
end 

function setindex!(vec::AbstractVector, pv::PartitionedVector, inds...)
  setindex!(vec, pv.epv, inds...)  
  return vec
end

function setindex!(pv1::PartitionedVector, pv2::PartitionedVector, inds...)
  setindex!(pv1.epv, pv2.epv, inds...)  
  return vec
end

function (+)(pv1::PartitionedVector, pv2::PartitionedVector)
  epv1 = pv1.epv
  epv2 = pv2.epv
  _epv = (+)(epv1, epv2)
  return _epv
end

function (-)(pv1::PartitionedVector, pv2::PartitionedVector)
  epv1 = pv1.epv
  epv2 = pv2.epv
  _epv = (-)(epv1, epv2)
  return _epv
end

function (-)(pv::PartitionedVector)
  epv = pv.epv
  _epv = (-)(epv)
  return _epv
end

copy(pv::PartitionedVector{T}) where {T <: Number} = PartitionedVector{T}(copy(pv.epv))
similar(pv::PartitionedVector{T}) where {T <: Number} = PartitionedVector{T}(similar(pv.epv))


end