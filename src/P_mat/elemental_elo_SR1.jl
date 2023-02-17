module ModElemental_elo_sr1

using ..Acronyms
using LinearOperators
using ..M_elt_mat, ..M_abstract_element_struct

import Base.==, Base.copy, Base.similar

export Elemental_elo_sr1
export init_eelo_LSR1, LSR1_eelo_rand, LSR1_eelo
export reset_eelo_sr1!

"""
    Elemental_elo_sr1{T} <: LOEltMat{T}

Represent an elemental element `LSR1Operator`;
* `indices` retains the indices of the elemental variables;
* `nie` is the elemental size (`=length(indices)`);
* `Bie` a `LSR1Operator`;
* `linear`: if `linear==true`, then the element matrix contribution is null;
* `counter` counts how many update the elemental limited-memory operator goes through from its allocation.
"""
mutable struct Elemental_elo_sr1{T} <: LOEltMat{T}
  nie::Int # nᵢᴱ
  indices::Vector{Int} # size nᵢᴱ
  Bie::LinearOperators.LSR1Operator{T}
  counter::Counter_elt_mat
  linear::Bool
end

@inline (==)(eelo1::Elemental_elo_sr1{T}, eelo2::Elemental_elo_sr1{T}) where {T} =
  (get_nie(eelo1) == get_nie(eelo2)) &&
  (get_indices(eelo1) == get_indices(eelo2)) &&
  (get_linear(eelo1) == get_linear(eelo2)) &&
  begin
    v = rand(get_nie(eelo1))
    (get_Bie(eelo1) * v == get_Bie(eelo2) * v)
  end
@inline copy(eelo::Elemental_elo_sr1{T}) where {T} = Elemental_elo_sr1{T}(
  copy(get_nie(eelo)),
  copy(get_indices(eelo)),
  deepcopy(get_Bie(eelo)),
  copy(get_cem(eelo)),
  copy(get_linear(eelo)),
)
@inline similar(eelo::Elemental_elo_sr1{T}) where {T} = Elemental_elo_sr1{T}(
  copy(get_nie(eelo)),
  copy(get_indices(eelo)),
  LSR1Operator(T, get_nie(eelo)),
  copy(get_cem(eelo)),
  copy(get_linear(eelo)),
)

"""
    eelo = init_eelo_LSR1(elt_var::Vector{Int}; T=Float64)

Return an `Elemental_elo_sr1` of type `T` based on the vector of the elemental variables `elt_var`.
"""
function init_eelo_LSR1(elt_var::Vector{Int}; T = Float64, linear = false)
  nie = length(elt_var)
  _nie = (!linear) * nie
  Bie = LinearOperators.LSR1Operator(T, _nie)
  counter = Counter_elt_mat()
  eelo = Elemental_elo_sr1{T}(nie, elt_var, Bie, counter, linear)
  return eelo
end

"""
    eelo = LSR1_eelo_rand(nie::Int; T=Float64, n=nie^2)

Return an `Elemental_elo_sr1` of type `T` with `nie` random indices within the range `1:n`.
"""
function LSR1_eelo_rand(nie::Int; T = Float64, n = nie^2, linear = false)
  indices = rand(1:n, nie)
  _nie = (!linear) * nie
  Bie = LinearOperators.LSR1Operator(T, _nie)
  counter = Counter_elt_mat()
  eelo = Elemental_elo_sr1{T}(nie, indices, Bie, counter, linear)
  return eelo
end

"""
    eelo = LSR1_eelo(nie::Int; T=Float64, index=1)

Return an `Elemental_elo_sr1` of type `T` of size `nie`, the indices are all the values in the range `index:index+nie-1`.
"""
function LSR1_eelo(nie::Int; T = Float64, index = 1, linear = false)
  indices = [index:1:(index + nie - 1);]
  _nie = (!linear) * nie
  Bie = LinearOperators.LSR1Operator(T, _nie)
  counter = Counter_elt_mat()
  eelo = Elemental_elo_sr1{T}(nie, indices, Bie, counter, linear)
  return eelo
end

"""
    reset_eelo_sr1!(eelo::Elemental_elo_sr1{T}) where T <: Number

Reset the LSR1 linear-operator of the elemental element linear-operator matrix `eelo`.
"""
function reset_eelo_sr1!(eelo::Elemental_elo_sr1{T}) where {T <: Number}
  eelo.Bie = LinearOperators.LSR1Operator(T, eelo.nie)
  return eelo
end

end
