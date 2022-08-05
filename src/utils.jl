module Utils

using LinearAlgebra

export BFGS, BFGS!, SR1, SR1!, SE, SE!
export my_and, max_indices, min_indices

"""
    my_and(a::Bool, b::Bool)

Return `a && b`.
"""
my_and = (a::Bool, b::Bool) -> (a && b)

"""
    indice_max = max_indices(list_of_element_variables::Vector{Vector{T}}) where T
    indice_max = max_indices(elt_set::Vector{T}) where T <: Element_struct

Return the maximum index of the element variables in `list_of_element_variables` or in `elt_set`.
"""
max_indices(elt_vars::Vector{Vector{T}}) where {T <: Number} =
  isempty(elt_vars) ? 0 : maximum(maximum.(elt_vars))

"""
    indice_min = min_indices(list_of_element_variables::Vector{Vector{T}}) where T
    indice_min = min_indices(elt_set::Vector{T}) where T <: Element_struct

Return the minimum index of the element variables in `list_of_element_variables` or in `elt_set`.
"""
min_indices(elt_vars::Vector{Vector{T}}) where {T <: Number} =
  isempty(elt_vars) ? 0 : minimum(minimum.(elt_vars))

"""
    BFGS(s::Vector{Y}, y::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number
    BFGS(x0::Vector{Y}, x1::Vector{Y}, g0::Vector{Y}, g1::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number

Perform the BFGS update over the matrix `B` by using the vectors `s = x1 - x0` and `y = g1 - g0`.
"""
function BFGS(s::Vector{Y}, y::Vector{Y}, B::Array{Y, 2}; kwargs...) where {Y <: Number}
  _B = copy(B)
  BFGS!(s, y, _B; kwargs...)
  return _B
end

function BFGS(
  x::Vector{Y},
  x_1::Vector{Y},
  g::Vector{Y},
  g_1::Vector{Y},
  B::Array{Y, 2};
  kwargs...,
) where {Y <: Number}
  _B = copy(B)
  BFGS!(x_1 - x, g_1 - g, _B; kwargs...)
  return _B
end

"""
    BFGS!(x0::Vector{Y}, x1::Vector{Y}, g0::Vector{Y}, g1::Vector{Y}, B0::Array{Y,2}; kwargs...) where Y <: Number
    BFGS!(s::Vector{Y}, y::Vector{Y}, B::Symmetric{Y,Matrix{Y}}; kwargs...) where Y <: Number
    BFGS!(s::Vector{Y}, y::Vector{Y}, B::Array{Y,2}; index=0, reset=4, kwargs...)

Perform the BFGS update in place of the matrix `B1` by using the vectors `s = x1 - x0` and `y = g1 - g0` and the current matrix `B0`.
"""
BFGS!(
  x::Vector{Y},
  x_1::Vector{Y},
  g::Vector{Y},
  g_1::Vector{Y},
  B::Array{Y, 2};
  kwargs...,
) where {Y <: Number} = BFGS!(x_1 - x, g_1 - g, B; kwargs...)
BFGS!(s::Vector{Y}, y::Vector{Y}, B::Symmetric{Y, Matrix{Y}}; kwargs...) where {Y <: Number} =
  BFGS!(s, y, B.data; kwargs...)
function BFGS!(
  s::Vector{Y},
  y::Vector{Y},
  B::Array{Y, 2};
  index = 0,
  reset = 4,
  Bs::Vector{Y} = similar(s),
  kwargs...,
) where {Y <: Number} #Array that will store the next approximation of the Hessia
  ys = dot(y, s)
  if ys > eps(Y) # curvature condition    
    mul!(Bs, B, s)
    mul!(B, y, y', 1 / ys, 1) # first term    
    mul!(B, Bs, Bs', -1 / dot(Bs, s), 1) # second term    
    return 1
  elseif index < reset #
    B .= B
    return 0
  else
    n = length(s)
    B .= [(i == j ? (Y)(1) : (Y)(0)) for i = 1:n, j = 1:n]
    return -1
  end
end

"""
    SR1(s::Vector{Y}, y::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number
    SR1(x::Vector{Y}, x1::Vector{Y}, g::Vector{Y}, g1::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number

Perform the SR1 update over the matrix `B` by using the vectors `s = x1 - x0` and `y = g1 - g0`.
"""
function SR1(s::Vector{Y}, y::Vector{Y}, B::Array{Y, 2}; kwargs...) where {Y <: Number}
  _B = copy(B)
  SR1!(s, y, _B; kwargs...)
  _B
end

function SR1(
  x::Vector{Y},
  x_1::Vector{Y},
  g::Vector{Y},
  g_1::Vector{Y},
  B::Array{Y, 2};
  kwargs...,
) where {Y <: Number}
  _B = copy(B)
  SR1!(x_1 - x, g_1 - g, _B; kwargs...)
  _B
end

"""
    SR1!(s::Vector{Y}, y::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number
    SR1!(x::Vector{Y}, x1::Vector{Y}, g::Vector{Y}, g1::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number
    SR1!(s::Vector{Y}, y::Vector{Y}, B::Array{Y,2}, B1::Array{Y,2}; index=0, reset=4, ω = 1e-6, kwargs...)

Perform the SR1 update in place of the matrix `B1` by using the vectors `s = x1 - x0` and `y = g1 - g0` and the current matrix `B`.
"""
SR1!(
  x::Vector{Y},
  x_1::Vector{Y},
  g::Vector{Y},
  g_1::Vector{Y},
  B::Array{Y, 2};
) where {Y <: Number} = SR1!(x_1 - x, g_1 - g, B)
SR1!(s::Vector{Y}, y::Vector{Y}, B::Symmetric{Y, Matrix{Y}}; kwargs...) where {Y <: Number} =
  SR1!(s, y, B.data; kwargs...)
function SR1!(
  s::Vector{Y},
  y::Vector{Y},
  B::Array{Y, 2};
  r::Vector{Y} = similar(s),
  index = 0,
  reset = 4,
  ω = 1e-6,
  kwargs...,
) where {Y <: Number}
  r .= y
  mul!(r, B, s, -1, 1)
  sr = dot(s, r)
  if abs(sr) > ω * norm(s, 2) * norm(r, 2)
    mul!(B, r, r', 1 / sr, 1) # first term    
    return 1
  elseif index < reset #
    B .= B
    return 0
  else
    n = length(s)
    B .= [(i == j ? (Y)(1) : (Y)(0)) for i = 1:n, j = 1:n]
    return -1
  end
end

"""
    SE(s::Vector{Y}, y::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number
    SE(x::Vector{Y}, x1::Vector{Y}, g::Vector{Y}, g1::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number

Perform a BFGS update over the matrix `B` by using the vectors `s = x1 - x0` and `y = g1 - g0` if the curvature condition `dot(s,y) > eps(eltype(s))` holds.
Otherwise, it performs a SR1 update with `B, s, y`.
"""
function SE(s::Vector{Y}, y::Vector{Y}, B::Array{Y, 2}; kwargs...) where {Y <: Number}
  _B = copy(B)
  SE!(s, y, _B; kwargs...)
  return _B
end

function SE(
  x::Vector{Y},
  x_1::Vector{Y},
  g::Vector{Y},
  g_1::Vector{Y},
  B::Array{Y, 2};
  kwargs...,
) where {Y <: Number}
  _B = copy(B)
  SE!(x_1 - x, g_1 - g, _B; kwargs...)
  _B
end

"""
    SE!(s::Vector{Y}, y::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number
    SE!(x::Vector{Y}, x1::Vector{Y}, g::Vector{Y}, g1::Vector{Y}, B::Array{Y,2}; kwargs...) where Y <: Number
    SE!(s::Vector{Y}, y::Vector{Y}, B::Array{Y,2}, B1::Array{Y,2}; index=0, reset=4, ω = 1e-6, kwargs...)

Perform a BFGS update in place of `B1` by using the matrix `B`, the vectors `s = x1 - x0` and `y = g1 - g0` if the curvature condition `dot(s,y) > eps(eltype(s))` holds.
Otherwise, it performs a SR1 update onto `B1` with `B, s, y`.
"""
SE!(
  x::Vector{Y},
  x_1::Vector{Y},
  g::Vector{Y},
  g_1::Vector{Y},
  B::Array{Y, 2},
) where {Y <: Number} = SE!(x_1 - x, g_1 - g, B)
SE!(s::Vector{Y}, y::Vector{Y}, B::Symmetric{Y, Matrix{Y}}; kwargs...) where {Y <: Number} =
  SE!(s, y, B.data; kwargs...)
function SE!(
  s::Vector{Y},
  y::Vector{Y},
  B::Array{Y, 2};
  Bs_r::Vector{Y} = similar(s),
  index = 0,
  reset = 4,
  ω = 1e-6,
  kwargs...,
) where {Y <: Number}
  ys = dot(s, y)
  if ys > eps(Y) # curvature condition
    mul!(Bs_r, B, s)
    mul!(B, y, y', 1 / ys, 1) # first term    
    mul!(B, Bs_r, Bs_r', -1 / dot(Bs_r, s), 1) # second term   
    return 1
  else
    Bs_r .= y
    mul!(Bs_r, B, s, -1, 1)
    sr = dot(s, Bs_r)
    if abs(dot(s, Bs_r)) > ω * norm(s, 2) * norm(Bs_r, 2)
      mul!(B, Bs_r, Bs_r', 1 / sr, 1) # first term    
      return 1
    elseif index < reset #
      B .= B
      return 0
    else
      n = length(s)
      B .= [(i == j ? (Y)(1) : (Y)(0)) for i = 1:n for j = 1:n]
      return -1
    end
  end
end

end
