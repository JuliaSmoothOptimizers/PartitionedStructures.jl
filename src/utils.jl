module Utils

using LinearAlgebra

export BFGS, BFGS!, SR1, SR1!, SE, SE!
export my_and, max_indices, min_indices

"""
    my_and(a :: Bool,b :: Bool)

Return `a && b`
"""
my_and = (a :: Bool,b :: Bool) -> (a && b)

"""
    indice_max = max_indices(list_of_element_variables :: Vector{Vector{T}}) where T

Return the maximum of the element variables in `list_of_element_variables`.
"""
max_indices(elt_vars :: Vector{Vector{T}}) where T <: Number = isempty(elt_vars) ? 0 : maximum(maximum.(elt_vars))

"""
    indice_min = min_indices(list_of_element_variables :: Vector{Vector{T}}) where T

Return the minimum of the element variables in `list_of_element_variables`.
"""
min_indices(elt_vars :: Vector{Vector{T}}) where T <: Number = isempty(elt_vars) ? 0 : minimum(minimum.(elt_vars))

"""
    BFGS(s, y, B)
    BFGS(x0, x1, g0, g1, B)

Perform the BFGS update over the matrix `B` by using the vectors `s = x1 - x0` and `y = g1 - g0`.
"""
BFGS(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); BFGS!(s,y,B,B_1;kwargs...); B_1 end
BFGS(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); BFGS!(x_1 - x, g_1 - g, B, B_1; kwargs...); B_1 end

"""
    BFGS!(s, y, B, B1)
    BFGS!(x0, x1, g0, g1, B, B1)

Perform the BFGS update in place of the matrix `B1` by using the vectors `s = x1 - x0` and `y = g1 - g0` and the current matrix `B`.
"""
BFGS!(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}; kwargs...) where Y <: Number = BFGS!(x_1 - x, g_1 - g, B, B_1; kwargs...)
BFGS!(s :: Vector{Y}, y :: Vector{Y}, B :: Symmetric{Y,Matrix{Y}}, B_1 :: Symmetric{Y,Matrix{Y}}; kwargs...) where Y <: Number = BFGS!(s,y,B.data, B_1.data; kwargs...)
function BFGS!(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}; index=0, reset=4, kwargs...) where Y <: Number #Array that will store the next approximation of the Hessian
  if dot(s,y) > eps(Y)  # curvature condition
    Bs = B * s
    terme1 =  (y * y') ./ dot(y,s)
    terme2 = (Bs * Bs') ./ dot(Bs,s)
    B_1 .= B .+ terme1 .- terme2
    return 1
  elseif index < reset #
    B_1 .= B
    return 0
  else
    n = length(s)
    B_1 .= reshape([ (i==j ? (Y)(1) : (Y)(0)) for i = 1:n for j =1:n], n, n)
    return -1
  end
end

"""
    SR1(s, y, B)
    SR1(x0, x1, g0, g1, B)

Perform the BFGS update over the matrix `B` by using the vectors `s = x1 - x0` and `y = g1 - g0`.
"""
SR1(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); SR1!(s,y,B,B_1;kwargs...); B_1 end
SR1(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); SR1!(x_1 - x, g_1 - g, B, B_1; kwargs...); B_1 end

"""
    SR1!(s, y, B, B1)
    SR1!(x0, x1, g0, g1, B, B1)

Perform the BFGS update in place of the matrix `B1` by using the vectors `s = x1 - x0` and `y = g1 - g0` and the current matrix `B`.
"""
SR1!(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}) where Y <: Number = SR1!(x_1 - x, g_1 - g, B, B_1)
SR1!(s :: Vector{Y}, y :: Vector{Y}, B :: Symmetric{Y,Matrix{Y}}, B_1 :: Symmetric{Y,Matrix{Y}}; kwargs...) where Y <: Number = SR1!(s,y,B.data, B_1.data; kwargs...)
function SR1!(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}; index=0, reset=4, ω = 1e-6, kwargs...) where Y <: Number
  r = y .- B*s
  if abs(dot(s,r)) > ω * norm(s,2) * norm(r,2)
    B_1 .= B .+ ((r * r')./dot(s,r))
    return 1
  elseif index < reset #
    B_1 .= B
    return 0
  else
    n = length(s)
    B_1 .= reshape([ (i==j ? (Y)(1) : (Y)(0)) for i = 1:n for j =1:n], n, n)
    return -1
  end
end

"""
    SE(s, y, B)
    SE(x0, x1, g0, g1, B)

Perform a BFGS update over the matrix `B` by using the vectors `s = x1 - x0` and `y = g1 - g0` if the curvature condition `dot(s,y) > eps(eltype(s))` holds.
Otherwise, it performs a SR1 update with `B, s, y`.
"""
SE(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); SE!(s,y,B,B_1;kwargs...); B_1 end
SE(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); SE!(x_1 - x, g_1 - g, B, B_1; kwargs...); B_1 end

"""
    SE!(s, y, B, B1)
    SE!(x0, x1, g0, g1, B, B1)

Perform a BFGS update in place of `B1` by using the matrix `B`, the vectors `s = x1 - x0` and `y = g1 - g0` if the curvature condition `dot(s,y) > eps(eltype(s))` holds.
Otherwise, it performs a SR1 update onto `B1` with `B, s, y`.
"""
SE!(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}) where Y <: Number = SE!(x_1 - x, g_1 - g, B, B_1)
SE!(s :: Vector{Y}, y :: Vector{Y}, B :: Symmetric{Y,Matrix{Y}}, B_1 :: Symmetric{Y,Matrix{Y}}; kwargs...) where Y <: Number = SE!(s,y,B.data, B_1.data; kwargs...)
function SE!(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}; index=0, reset=4, ω = 1e-6, kwargs...) where Y <: Number
  if dot(s,y) > eps(Y)  # curvature condition
    Bs = B * s
    terme1 =  (y * y') ./ dot(y,s)
    terme2 = (Bs * Bs') ./ dot(Bs,s)
    B_1 .= B .+ terme1 .- terme2
    return 1
  else
    r = y .- B*s
    if abs(dot(s,r)) > ω * norm(s,2) * norm(r,2)
      B_1 .= B .+ ((r * r')./dot(s,r))
      return 1
    elseif index < reset #
      B_1 .= B
      return 0
    else
      n = length(s)
      B_1 .= reshape([ (i==j ? (Y)(1) : (Y)(0)) for i = 1:n for j =1:n], n, n)
      return -1
    end
  end
end

end
