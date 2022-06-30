module M_elt_mat

using ..Acronyms
using ..M_abstract_element_struct

export Elt_mat, DenseEltMat, LOEltMat
export get_Bie, get_counter_elt_mat, get_cem, get_current_untouched, get_index

export Counter_elt_mat
export update_counter_elt_mat!, iter_info, total_info

import Base.copy, Base.similar

"Supertype of every element-matrix, ex: Elemental_em, Elemental_elo_sr1, Elemental_elo_bfgs"
abstract type Elt_mat{T} <: Element_struct{T} end
"Supertype of every dense element-matrix, ex: Elemental_em"
abstract type DenseEltMat{T} <: Elt_mat{T} end
"Supertype of every element linear-operator, ex: Elemental_elo_sr1, Elemental_elo_bfgs"
abstract type LOEltMat{T} <: Elt_mat{T} end

"""
    Counter_elt_mat

Count for an element-matrix the updates performed on it, from its allocation.
`total_update + total_reset + total_untouched == attempt`, .
"""
mutable struct Counter_elt_mat
  total_update::Int # count the total of update perform by the element linear-operator
  current_update::Int # count how many time by the element linear-operator
  total_untouched::Int
  current_untouched::Int # must be ≤ reset defined in the update
  total_reset::Int
  current_reset::Int # ≤ 1 as long as reset ≥ 2 in any update performed
end

"""
    get_Bie(elt_mat::T) where T <: Elt_mat

Return the element-matrix `elt_mat.Bie`.
"""
@inline get_Bie(elt_mat::T) where {T <: Elt_mat} = elt_mat.Bie

"""
    cem = get_counter_elt_mat(elt_mat::T) where T <: Elt_mat

Return the `Counter_elt_mat` of the elemental element-matrix `elt_mat`.
"""
@inline get_counter_elt_mat(elt_mat::T) where {T <: Elt_mat} = elt_mat.counter

"""
    cem = get_cem(elt_mat::T) where T <: Elt_mat

Return the `Counter_elt_mat` of the elemental element-matrix `elt_mat`.
"""
@inline get_cem(elt_mat::T) where {T <: Elt_mat} = elt_mat.counter

"""
    index = get_index(elt_mat::T) where T <: Elt_mat

Return index: the number of the last partitioned-updates that did not update the element-matrix `elt_mat`.
If the last partitioned-update updates `elt_mat` then `index` will be equal to `0`.
"""
@inline get_index(elt_mat::T) where {T <: Elt_mat} = get_current_untouched(elt_mat.counter)

Counter_elt_mat() = Counter_elt_mat(0, 0, 0, 0, 0, 0)
copy(cem::Counter_elt_mat) = Counter_elt_mat(
  cem.total_update,
  cem.current_update,
  cem.total_untouched,
  cem.current_untouched,
  cem.total_reset,
  cem.current_reset,
)
similar(cem::Counter_elt_mat) = Counter_elt_mat()

"""
    index = get_current_untouched(cem::Counter_elt_mat)

Return index: the number of the last partitioned-updates that did not update the element-matrix `elt_mat`.
If the last partitioned-update updates `elt_mat` then `index` will be equal to `0`.
"""
get_current_untouched(cem::Counter_elt_mat) = cem.current_untouched

"""
    (current_update, current_untouched, current_reset) = iter_info(cem::Counter_elt_mat)

Return the information about the last partitioned quasi-Newton update applied onto the element counter `cem` (associated to an element-matrix).
"""
iter_info(cem::Counter_elt_mat) = (cem.current_update, cem.current_untouched, cem.current_reset)

"""
    (total_update, total_untouched, current_reset) = iter_info(cem::Counter_elt_mat)

Return the informations about all the quasi-Newton updates applied onto the element associated to the counter `cem`.
"""
total_info(cem::Counter_elt_mat) = (cem.total_update, cem.total_untouched, cem.current_reset)

"""
    update_counter_elt_mat!(cem::Counter_elt_mat, qn::Int)

Update the `cem` counter given the index `qn` from the quasi-Newton update BFGS!, SR1!, SE!.
"""
function update_counter_elt_mat!(cem::Counter_elt_mat, qn::Int)
  if qn == 1
    cem.total_update += 1
    cem.current_update += 1
    cem.current_untouched = 0
    cem.current_reset = 0
  elseif qn == 0
    cem.total_untouched += 1
    cem.current_untouched += 1
    cem.current_update = 0
    cem.current_reset = 0
  else # qn==-1
    cem.total_reset += 1
    cem.current_reset += 1
    cem.current_untouched = 0
    cem.current_update = 0
  end
  return cem
end

end
