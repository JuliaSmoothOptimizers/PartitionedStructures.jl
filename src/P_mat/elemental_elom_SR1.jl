module ModElemental_elom_sr1

using LinearOperators
using ..M_elt_mat, ..M_abstract_element_struct

import Base.==, Base.copy, Base.similar

export Elemental_elom_sr1
export init_eelom_LSR1, LSR1_eelom_rand, LSR1_eelom
export reset_eelom_sr1!

"""
    Elemental_elom_sr1{T}<:LOEltMat{T}

Type that represents an elemental element linear operator LSR1.
"""
mutable struct Elemental_elom_sr1{T}<:LOEltMat{T}
  nie::Int # nᵢᴱ
  indices::Vector{Int} # size nᵢᴱ
  Bie::LinearOperators.LSR1Operator{T}
  counter::Counter_elt_mat
end

@inline (==)(eelom1::Elemental_elom_sr1{T}, eelom2::Elemental_elom_sr1{T}) where T = (get_nie(eelom1)== get_nie(eelom2)) && begin v=rand(get_nie(eelom1)); (get_Bie(eelom1) *v == get_Bie(eelom2)*v) end && (get_indices(eelom1)== get_indices(eelom2))
@inline copy(eelom::Elemental_elom_sr1{T}) where T = Elemental_elom_sr1{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), deepcopy(get_Bie(eelom)), copy(get_cem(eelom)))
@inline similar(eelom::Elemental_elom_sr1{T}) where T = Elemental_elom_sr1{T}(copy(get_nie(eelom)), copy(get_indices(eelom)), similar(get_Bie(eelom)), copy(get_cem(eelom)))

"""
    init_eelom_LSR1(elt_var; T=T)

Return an `Elemental_elom_sr1` of type `T` based on the vector of the elemental variables `elt_var`.
"""
function init_eelom_LSR1(elt_var::Vector{Int}; T=Float64)
  nie = length(elt_var)
  Bie = LinearOperators.LSR1Operator(T, nie)
  counter = Counter_elt_mat()
  eelom = Elemental_elom_sr1{T}(nie, elt_var, Bie, counter)
  return eelom
end

"""
    eelom =LSR1_eelom_rand(nie, T=T, n=n)

Return an `Elemental_elom_sr1` of type `T` with `nie` random indices within the range `1:n`.
"""
function LSR1_eelom_rand(nie::Int; T=Float64, n=nie^2)
  indices = rand(1:n, nie)
  Bie = LinearOperators.LSR1Operator(T, nie)
  counter = Counter_elt_mat()
  eelom = Elemental_elom_sr1{T}(nie, indices, Bie, counter)
  return eelom
end

"""
    eelom = LSR1_eelom(nie, T=T, index=index)

Return an `Elemental_elom_sr1` of type `T` of size `nie`, the indices are all the values in the range `index:index+nie-1`.
"""
function LSR1_eelom(nie::Int; T=Float64, index=1)
  indices = [index:1:index+nie-1;]
  Bie = LinearOperators.LSR1Operator(T, nie)
  counter = Counter_elt_mat()
  eelom = Elemental_elom_sr1{T}(nie, indices, Bie, counter)
  return eelom
end

"""
    index_eelom_sr1!(eelom)

Reset the LSR1 linear operator of the elemental element linear operator matrix `eelom`.
"""
reset_eelom_sr1!(eelom::Elemental_elom_sr1{T}) where T<:Number = eelom.Bie = LinearOperators.LSR1Operator(T, eelom.nie)

end 