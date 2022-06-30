module Link

using ..Acronyms
using LinearAlgebra, SparseArrays

using ..M_elt_mat
using ..M_part_mat, ..M_part_v
using ..ModElemental_em, ..ModElemental_ev
using ..ModElemental_pv,
  ..ModElemental_plo_bfgs, ..ModElemental_plo_sr1, ..ModElemental_plo, ..ModElemental_pm
using ..M_abstract_element_struct, ..M_abstract_part_struct

export eplo_lbfgs_from_epv, eplo_lsr1_from_epv, eplo_lose_from_epv, epm_from_epv
export epv_from_eplo, epv_from_epm
export mul_epm_epv, mul_epm_epv!, mul_epm_vector, mul_epm_vector!
export string_counters_iter, string_counters_total

"""
    epv = epv_from_eplo(eplo::T) where T<:Part_mat{Y} 

Create an $(_epv) with the same partitioned structure than `eplo`.
Each element-vector of `epv` is set to a random vector of suitable size.
Make a name difference with `epv_from_epm`.
"""
@inline epv_from_eplo(eplo) = epv_from_epm(eplo)

"""
    epv = epv_from_epm(epm::T) where T<:Part_mat{Y}

Create an $(_epv) with the same partitioned structure than `epm`.
Each element-vector of `epv` is set to a random vector of suitable size.
"""
function epv_from_epm(epm::T) where {Y <: Number, T <: Part_mat{Y}}
  N = get_N(epm)
  n = get_n(epm)
  eev_set = Vector{Elemental_elt_vec{Y}}(undef, N)
  for i = 1:N
    eesi = get_ee_struct(epm, i)
    indices = get_indices(eesi)
    nie = get_nie(eesi)
    eev_set[i] = Elemental_elt_vec{Y}(rand(Y, nie), indices, nie)
  end
  component_list = M_abstract_part_struct.get_component_list(epm)
  v = rand(Y, n)
  perm = [1:n;]
  epv = Elemental_pv{Y}(N, n, eev_set, v, component_list, perm)
  return epv
end

"""
    epm = epm_from_epv(epv::T) where {Y<:Number, T<:Elemental_pv{Y}}

Create an elemental partitioned quasi-Newton operator `epm` with the same partitioned structure than `epv`.
Each element-matrix of `epm` is set with an identity matrix of suitable size.
"""
function epm_from_epv(epv::T) where {Y <: Number, T <: Elemental_pv{Y}}
  N = get_N(epv)
  n = get_n(epv)
  eelo_indices_set = Vector{Vector{Int}}(undef, N)
  for i = 1:N
    eesi = get_ee_struct(epv, i)
    indices = get_indices(eesi)
    eelo_indices_set[i] = indices
  end
  epm = identity_epm(eelo_indices_set, N, n; T = Y)
  return epm
end

"""
    eplo = eplo_lbfgs_from_epv(epv::T) where {Y <: Number, T <: Elemental_pv{Y}}

Create an $(_elmpqno) PLBFGS `eplo` with the same partitioned structure than `epv`.
Each element linear-operator of `eplo` is set to a `LBFGSOperator` of suitable size.
"""
function eplo_lbfgs_from_epv(epv::T) where {Y <: Number, T <: Elemental_pv{Y}}
  N = get_N(epv)
  n = get_n(epv)
  eelo_indices_set = Vector{Vector{Int}}(undef, N)
  for i = 1:N
    eesi = get_ee_struct(epv, i)
    indices = get_indices(eesi)
    eelo_indices_set[i] = indices
  end
  eplo = identity_eplo_LBFGS(eelo_indices_set, N, n; T = Y)
  return eplo
end

"""
    eplo = eplo_lsr1_from_epv(epv::T) where {Y <: Number, T <: Elemental_pv{Y}}

Create an $(_elmpqno) PLSR1 `eplo` with the same partitioned structure than `epv`.
Each element linear-operator of `eplo` is set to a `LSR1Operator` of suitable size.
"""
function eplo_lsr1_from_epv(epv::T) where {Y <: Number, T <: Elemental_pv{Y}}
  N = get_N(epv)
  n = get_n(epv)
  eelo_indices_set = Vector{Vector{Int}}(undef, N)
  for i = 1:N
    eesi = get_ee_struct(epv, i)
    indices = get_indices(eesi)
    eelo_indices_set[i] = indices
  end
  eplo = identity_eplo_LSR1(eelo_indices_set, N, n; T = Y)
  return eplo
end

"""
    eplo = eplo_lose_from_epv(epv::Elemental_pv{T}) where {T <: Number}

Create an $(_elmpqno) PLSE `eplo` with the same partitioned structure than `epv`.
Each element linear-operator of `eplo` is set to a `LBFGSOperator` of suitable size, but it may change to a `LSR1Operator` later on.
"""
function eplo_lose_from_epv(epv::Elemental_pv{T}) where {T <: Number}
  N = get_N(epv)
  n = get_n(epv)
  eelo_indices_set = Vector{Vector{Int}}(undef, N)
  for i = 1:N
    eesi = get_ee_struct(epv, i)
    indices = get_indices(eesi)
    eelo_indices_set[i] = indices
  end
  eplo = identity_eplo_LOSE(eelo_indices_set, N, n; T = T)
  return eplo
end

"""
    result = mul_epm_vector(epm::T, x::Vector{Y}) where {Y<:Number, T<:Part_mat{Y}}
    result = mul_epm_vector(epm::T, epv::Elemental_pv{Y}, x::Vector{Y}) where {Y<:Number, T<:Part_mat{Y}}

Compute the product between the elemental partitioned-matrix `epm<:Part_mat` and the vector `x`.
The method uses temporary the $(_epv).
The method returns `result`, a vector similar to `x`.
"""
function mul_epm_vector(epm::T, x::Vector{Y}) where {Y <: Number, T <: Part_mat{Y}}
  epv = epv_from_epm(epm)
  return mul_epm_vector(epm, epv, x)
end

function mul_epm_vector(
  epm::T,
  epv::Elemental_pv{Y},
  x::Vector{Y},
) where {Y <: Number, T <: Part_mat{Y}}
  res = similar(x)
  mul_epm_vector!(res, epm, epv, x)
  return res
end

"""
    mul_epm_vector!(res::Vector{Y}, epm::T, x::Vector{Y}) where {Y<:Number, T<:Part_mat{Y}}
    mul_epm_vector!(res::Vector{Y}, epm::T, epv::Elemental_pv{Y}, x::Vector{Y}) where {Y<:Number, T<:Part_mat{Y}}

Compute the product between the $(_epm) and the vector `x`.
The method uses temporary the $(_epv).
The result is stored in `res`, a vector similar to `x`.
"""
function mul_epm_vector!(res::Vector{Y}, epm::T, x::Vector{Y}) where {Y <: Number, T <: Part_mat{Y}}
  epv = epv_from_epm(epm)
  mul_epm_vector!(res, epm, epv, x)
  return res
end

function mul_epm_vector!(
  res::Vector{Y},
  epm::T,
  epv::Elemental_pv{Y},
  x::Vector{Y},
) where {Y <: Number, T <: Part_mat{Y}}
  epv_from_v!(epv, x)
  mul_epm_epv!(epv, epm, epv)
  build_v!(epv)
  res .= get_v(epv)
  return res
end

"""
    epv_res = mul_epm_epv(epm::T, epv::Elemental_pv{Y}) where {Y<:Number, T<:Part_mat{Y}}

Compute the elementwise product between the $(_epm) and the $(_epv).
The result is an elemental partitioned-vector `epv_res` storing the elementwise products between `epm` and `epv`.
"""
function mul_epm_epv(epm::T, epv::Elemental_pv{Y}) where {Y <: Number, T <: Part_mat{Y}}
  epv_res = similar(epv)
  mul_epm_epv!(epv_res, epm, epv)
  return epv_res
end

"""
    mul_epm_epv!mul_epm_epv!(epv_res::Elemental_pv{Y}, epm::T, epv::Elemental_pv{Y}) where {Y<:Number, T<:Part_mat{Y}}

Compute the elementwise product between the $(_epm) and the $(_epv).
The result of each element-matrix element-vector product is stored in the elemental partitioned-vector `epv_res`.
"""
function mul_epm_epv!(
  epv_res::Elemental_pv{Y},
  epm::T,
  epv::Elemental_pv{Y},
) where {Y <: Number, T <: Part_mat{Y}}
  full_check_epv_epm(epm, epv) || error("Different partitioned structure epm/epv")
  N = get_N(epm)
  for i = 1:N
    Bie = get_ee_struct_Bie(epm, i)
    vie = get_eev_value(epv, i)
    set_eev!(epv_res, i, Bie * vie)
  end
  return epv_res
end

"""
    s = string_counters_iter(pm::T; name = :PQN) where {T <: Part_mat}

Produce a `s::String` that summarizes the partitioned update applied onto `pm` at the last iterate.
The method accumulates the informations gathered by each element-counter during the last iterate.
"""
function string_counters_iter(pm::T; name = :PQN) where {T <: Part_mat}
  epm_vectors = get_ee_struct(pm)
  counters = (epm -> epm.counter).(epm_vectors)
  update = 0
  untouch = 0
  reset = 0
  for counter in counters
    (up, un, re) = iter_info(counter)
    update += (up > 0 ? 1 : 0)
    untouch += (un > 0 ? 1 : 0)
    reset += (re > 0 ? 1 : 0)
  end
  N = get_N(pm)
  str = "\t structure: $(T) based from $(N) elements; update: $(update), untouch: $(untouch), reset: $(reset) \n"
  return str
end

"""
    s = string_counters_total(pm::T; name = :PQN) where {T <: Part_mat}

Produce a `s::String` that summarizes the partitioned update applied onto `pm` since its allocations.
The method accumulates the informations gathered by each element-counter since their allocations.
"""
function string_counters_total(pm::T; name = :PQN) where {T <: Part_mat}
  epm_vectors = get_ee_struct(pm)
  counters = (epm -> epm.counter).(epm_vectors)
  update = 0
  untouch = 0
  reset = 0
  for counter in counters
    (up, un, re) = total_info(counter)
    update += up
    untouch += un
    reset += re
  end
  N = get_N(pm)
  str = "\t structure: $(T) based from $(N) elements; update: $(update), untouch: $(untouch), reset: $(reset) \n"
  return str
end

end
