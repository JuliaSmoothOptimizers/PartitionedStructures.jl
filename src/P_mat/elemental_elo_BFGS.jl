module ModElemental_elo_bfgs

using LinearOperators
using ..M_abstract_element_struct, ..M_elt_mat

import Base.==, Base.copy, Base.similar

export Elemental_elo_bfgs
export init_eelo_LBFGS, LBFGS_eelo, LBFGS_eelo_rand
export reset_eelo_bfgs!

"""
    Elemental_elo_bfgs{T}<:LOEltMat{T}

Type that represents an elemental element linear operator LBFGS.
"""
mutable struct Elemental_elo_bfgs{T}<:LOEltMat{T}
  nie::Int # nᵢᴱ
  indices::Vector{Int} # size nᵢᴱ
  Bie::LinearOperators.LBFGSOperator{T}
  counter::Counter_elt_mat
end

@inline (==)(eelo1::Elemental_elo_bfgs{T}, eelo2::Elemental_elo_bfgs{T}) where T = (get_nie(eelo1)==get_nie(eelo2)) && (get_indices(eelo1)==get_indices(eelo2)) && begin v=rand(get_nie(eelo1)); (get_Bie(eelo1)*v==get_Bie(eelo2)*v) end
@inline copy(eelo::Elemental_elo_bfgs{T}) where T = Elemental_elo_bfgs{T}(copy(get_nie(eelo)), copy(get_indices(eelo)), deepcopy(get_Bie(eelo)), copy(get_cem(eelo)))
@inline similar(eelo::Elemental_elo_bfgs{T}) where T = Elemental_elo_bfgs{T}(copy(get_nie(eelo)), copy(get_indices(eelo)), similar(get_Bie(eelo)), copy(get_cem(eelo)))

"""
    eelo = init_eelo_LBFGS(elt_var::Vector{Int}; T=Float64)

Return an `Elemental_elo_bfgs` of type `T` based on the vector of the elemental variables`elt_var`.
"""
function init_eelo_LBFGS(elt_var::Vector{Int}; T=Float64)
  nie = length(elt_var)
  Bie = LinearOperators.LBFGSOperator(T, nie)
  counter = Counter_elt_mat()
  eelo = Elemental_elo_bfgs{T}(nie, elt_var, Bie, counter)
  return eelo
end

"""
    eelo = LBFGS_eelo_rand(nie::Int; T=Float64, n=nie^2)

Return an `Elemental_elo_bfgs` of type `T` with `nie` random indices within the range `1:n`.
"""
function LBFGS_eelo_rand(nie::Int; T=Float64, n=nie^2)
  indices = rand(1:n, nie)
  Bie = LinearOperators.LBFGSOperator(T, nie)
  counter = Counter_elt_mat()
  eelo = Elemental_elo_bfgs{T}(nie, indices, Bie, counter)
  return eelo
end

"""
    eelo = LBFGS_eelo(nie::Int; T=Float64, index=1)

Return an `Elemental_elo_bfgs` of type `T` of size `nie`, the indices are all the values in the range `index:index+nie-1`.
"""
function LBFGS_eelo(nie::Int; T=Float64, index=1)
  indices = [index:1:index+nie-1;]
  Bie = LinearOperators.LBFGSOperator(T, nie)
  counter = Counter_elt_mat()
  eelo = Elemental_elo_bfgs{T}(nie, indices, Bie, counter)
  return eelo
end

"""
    reset_eelo_bfgs!(eelo::Elemental_elo_bfgs{T}) where T<:Number

Reset the LBFGS linear operator of the elemental element linear operator `eelo`.
"""
function reset_eelo_bfgs!(eelo::Elemental_elo_bfgs{T}) where T<:Number
  eelo.Bie = LinearOperators.LBFGSOperator(T, eelo.nie)
  return eelo
end

end