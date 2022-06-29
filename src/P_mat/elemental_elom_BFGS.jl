module ModElemental_elom_bfgs

using LinearOperators
using ..M_abstract_element_struct, ..M_elt_mat

import Base.==, Base.copy, Base.similar

export Elemental_elom_bfgs
export init_eelom_LBFGS, LBFGS_eelom, LBFGS_eelom_rand
export reset_eelom_bfgs!

"""
    Elemental_elom_bfgs{T}<:LOEltMat{T}

Type that represents an elemental element linear operator LBFGS.
"""
mutable struct Elemental_elom_bfgs{T}<:LOEltMat{T}
  nie::Int # nᵢᴱ
  indices::Vector{Int} # size nᵢᴱ
  Bie::LinearOperators.LBFGSOperator{T}
  counter::Counter_elt_mat
end

@inline (==)(eelom1::Elemental_elom_bfgs{T}, eelom2::Elemental_elom_bfgs{T}) where T = (get_nie(eelom1)==get_nie(eelom2)) && begin v=rand(get_nie(eelom1)); (get_Bie(eelom1)*v==get_Bie(eelom2)*v) end && (get_indices(eelom1)==get_indices(eelom2))
@inline copy(eelom::Elemental_elom_bfgs{T}) where T = Elemental_elom_bfgs{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), deepcopy(get_Bie(eelom)), copy(get_cem(eelom)))
@inline similar(eelom::Elemental_elom_bfgs{T}) where T = Elemental_elom_bfgs{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), similar(get_Bie(eelom)), copy(get_cem(eelom)))

"""
    eelom = init_eelom_LBFGS(elt_var; T=T)

Return an `Elemental_elom_bfgs` of type `T` based on the vector of the elemental variables`elt_var`.
"""
function init_eelom_LBFGS(elt_var::Vector{Int}; T=Float64)
  nie = length(elt_var)
  Bie = LinearOperators.LBFGSOperator(T, nie)
  counter = Counter_elt_mat()
  eelom = Elemental_elom_bfgs{T}(nie, elt_var, Bie, counter)
  return eelom
end

"""
    eelom = LBFGS_eelom_rand(nie; T=T, n=n)

Return an `Elemental_elom_bfgs` of type `T` with `nie` random indices within the range `1:n`.
"""
function LBFGS_eelom_rand(nie::Int; T=Float64, n=nie^2)
  indices = rand(1:n, nie)
  Bie = LinearOperators.LBFGSOperator(T, nie)
  counter = Counter_elt_mat()
  eelom = Elemental_elom_bfgs{T}(nie, indices, Bie, counter)
  return eelom
end

"""
    eelom = LBFGS_eelom(nie; T=T, index=index)

Return an `Elemental_elom_bfgs` of type `T` of size `nie`, the indices are all the values in the range `index:index+nie-1`.
"""
function LBFGS_eelom(nie::Int; T=Float64, index=1)
  indices = [index:1:index+nie-1;]
  Bie = LinearOperators.LBFGSOperator(T, nie)
  counter = Counter_elt_mat()
  eelom = Elemental_elom_bfgs{T}(nie, indices, Bie, counter)
  return eelom
end

"""
    reset_eelom_bfgs!(eelom)

Reset the LBFGS linear operator of the elemental element linear operator `eelom`.
"""
reset_eelom_bfgs!(eelom::Elemental_elom_bfgs{T}) where T<:Number = eelom.Bie = LinearOperators.LBFGSOperator(T, eelom.nie)

end 