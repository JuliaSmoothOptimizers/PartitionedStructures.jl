module PartitionedLOQuasiNewton

using ..Acronyms
using LinearOperators, LinearAlgebra

using ..M_abstract_part_struct, ..M_elt_vec, ..M_part_mat, ..M_elt_mat
using ..M_abstract_element_struct
using ..Utils, ..Link
using ..ModElemental_ev, ..ModElemental_pv
using ..ModElemental_elo_bfgs, ..ModElemental_elo_sr1, ..ModElemental_plo
using ..ModElemental_plo_bfgs, ..ModElemental_plo_sr1

export PLBFGS_update, PLBFGS_update!
export PLSR1_update, PLSR1_update!
export PLSE_update, PLSE_update!
export Part_update, Part_update!

"""
    copy_eplo_B = PLBFGS_update(eplo_B::Elemental_plo_bfgs{T}, epv_y::Elemental_pv{T}, s::Vector{T}; kwargs...) where T

Perform the PLBFGS update onto a copy of the partitioned limited-memory operator `eplo_B`, given the step `s` and the difference of elemental partitioned-gradients `epv_y`.
Return the updated copy of `eplo_B`.
"""
function PLBFGS_update(
  eplo_B::Elemental_plo_bfgs{T},
  epv_y::Elemental_pv{T},
  s::Vector{T};
  kwargs...,
) where {T}
  epm_copy = copy(eplo_B)
  PLBFGS_update!(epm_copy, epv_y, s; kwargs...)
  return epm_copy
end

"""
    PLBFGS_update!(eplo_B::Elemental_plo_bfgs{T}, epv_y::Elemental_pv{T}, s::Vector{T}; kwargs...) where T
    PLBFGS_update!(eplo_B::Elemental_plo_bfgs{T}, epv_y::Elemental_pv{T}, epv_s::Elemental_pv{T}; verbose=true, reset=true, kwargs...) where T

Perform the PLBFGS update onto the partitioned limited-memory operator `eplo_B`, given the step `s` (or the element-steps `epv_s`) and the difference of elemental partitioned-gradients `epv_y`.
"""
function PLBFGS_update!(
  eplo_B::Elemental_plo_bfgs{T},
  epv_y::Elemental_pv{T},
  s::Vector{T};
  kwargs...,
) where {T}
  epv_s = epv_from_v(s, epv_y)
  PLBFGS_update!(eplo_B, epv_y, epv_s; kwargs...)
  return eplo_B
end

function PLBFGS_update!(
  eplo_B::Elemental_plo_bfgs{T},
  epv_y::Elemental_pv{T},
  epv_s::Elemental_pv{T};
  verbose = true,
  reset = 4,
  kwargs...,
) where {T}
  full_check_epv_epm(eplo_B, epv_y) ||
    @error("different partitioned structures between eplo_B and epv_y")
  full_check_epv_epm(eplo_B, epv_s) ||
    @error("different partitioned structures between eplo_B and epv_s")
  N = get_N(eplo_B)
  for i = 1:N
    eelo = get_eelo_set(eplo_B, i)
    linear = get_linear(eelo)
    if !linear
      si = get_vec(get_eev_set(epv_s, i))
      yi = get_vec(get_eev_set(epv_y, i))
      index = get_index(eelo)
      if (dot(si, yi) > eps(T))
        Bi = get_Bie(eelo)
        push!(Bi, si, yi)
        update = 1
      elseif index < reset # Bi is not updated nor reset
        update = 0
      else
        reset_eelo_bfgs!(eelo)
        update = -1
      end
      cem = get_cem(eelo)
      update_counter_elt_mat!(cem, update)
    end
  end
  verbose && (str = string_counters_iter(eplo_B))
  verbose && (print("\n PLBFGS" * str))
  return eplo_B
end

"""
    copy_eplo_B = PLSR1_update(eplo_B::Elemental_plo_sr1{T}, epv_y::Elemental_pv{T}, s::Vector{T}; kwargs...) where T

Perform the PSR1 update onto a copy of the partitioned limited-memory operator `eplo_B`, given the step `s` and the difference of elemental partitioned-gradients `epv_y`.
Return the updated copy of `eplo_B`.
"""
function PLSR1_update(
  eplo_B::Elemental_plo_sr1{T},
  epv_y::Elemental_pv{T},
  s::Vector{T};
  kwargs...,
) where {T}
  epm_copy = copy(eplo_B)
  PLSR1_update!(epm_copy, epv_y, s; kwargs...)
  return epm_copy
end

"""
    PLSR1_update!(eplo_B::Elemental_plo_sr1{T}, epv_y::Elemental_pv{T}, s::Vector{T}; kwargs...) where T
    PLSR1_update!(eplo_B::Elemental_plo_sr1{T}, epv_y::Elemental_pv{T}, s::Vector{T}; kwargs...) where T

Perform the PLSR1 update onto the partitioned limited-memory operator `eplo_B`, given the step `s` (or the element-steps `epv_s`) and the difference of elemental partitioned-gradients `epv_y`.
"""
function PLSR1_update!(
  eplo_B::Elemental_plo_sr1{T},
  epv_y::Elemental_pv{T},
  s::Vector{T};
  kwargs...,
) where {T}
  epv_s = epv_from_v(s, epv_y)
  PLSR1_update!(eplo_B, epv_y, epv_s; kwargs...)
  return eplo_B
end

function PLSR1_update!(
  eplo_B::Elemental_plo_sr1{T},
  epv_y::Elemental_pv{T},
  epv_s::Elemental_pv{T};
  ω = 1e-6,
  verbose = true,
  reset = 4,
  kwargs...,
) where {T}
  full_check_epv_epm(eplo_B, epv_y) ||
    @error("different partitioned structures between eplo_B and epv_y")
  full_check_epv_epm(eplo_B, epv_s) ||
    @error("different partitioned structures between eplo_B and epv_s")
  N = get_N(eplo_B)
  for i = 1:N
    eelo = get_eelo_set(eplo_B, i)
    linear = get_linear(eelo)
    if !linear
      si = get_vec(get_eev_set(epv_s, i))
      yi = get_vec(get_eev_set(epv_y, i))
      Bi = get_Bie(eelo)
      ri = yi .- Bi * si
      index = get_index(eelo)
      if abs(dot(si, ri)) > ω * norm(si, 2) * norm(ri, 2)
        push!(Bi, si, yi)
        update = 1
      elseif index < reset # Bi is not updated nor reset
        update = 0
      else
        reset_eelo_sr1!(eelo)
        update = -1
      end
      cem = get_cem(eelo)
      update_counter_elt_mat!(cem, update)
    end
  end
  verbose && (str = string_counters_iter(eplo_B))
  verbose && (print("\n PLSR1" * str))
  return eplo_B
end

"""
    copy_eplo_B = Part_update(eplo_B::Y, epv_y::Elemental_pv{T}, s::Vector{T}) where {T, Y <: Part_LO_mat{T}}

Perform a quasi-Newton partitionned update onto a copy of the partitioned limited-memory operator `eplo_B`, given the step `s` and the difference of elemental partitioned-gradients `epv_y`.
Each elemental element linear-operator from `eplo_B` is either a `LBFGSOperator` or `LSR1Operator`.
The update performs on each element the quasi-Newton update associated to the linear-operator.
Return the updated copy of `eplo_B`.
"""
function Part_update(
  eplo_B::Y,
  epv_y::Elemental_pv{T},
  s::Vector{T};
  kwargs...,
) where {T, Y <: Part_LO_mat{T}}
  epm_copy = copy(eplo_B)
  Part_update!(epm_copy, epv_y, s; kwargs...)
  return epm_copy
end

"""
    Part_update!(eplo_B::Y, epv_y::Elemental_pv{T}, s::Vector{T}) where {T, Y <: Part_LO_mat{T}}
    Part_update!(eplo_B::Y, epv_y::Elemental_pv{T}, epv_s::Elemental_pv{T}; kwargs...) where {T, Y <: Part_LO_mat{T}}

Perform a partitioned quasi-Newton update onto the partitioned limited-memory operator `eplo_B`, given the step `s` (or the element-steps `epv_s`) and the difference of elemental partitioned-gradients `epv_y`.
Each elemental element linear-operator from `eplo_B` is either a `LBFGSOperator` or `LSR1Operator`.
The update performs on each element the quasi-Newton update associated to the linear-operator.
"""
function Part_update!(
  eplo_B::Y,
  epv_y::Elemental_pv{T},
  s::Vector{T};
  kwargs...,
) where {T, Y <: Part_LO_mat{T}}
  epv_s = epv_from_v(s, epv_y)
  Part_update!(eplo_B, epv_y, epv_s; kwargs...)
  return eplo_B
end

function Part_update!(
  eplo_B::Y,
  epv_y::Elemental_pv{T},
  epv_s::Elemental_pv{T};
  kwargs...,
) where {T, Y <: Part_LO_mat{T}}
  full_check_epv_epm(eplo_B, epv_y) ||
    @error("different partitioned structures between eplo_B and epv_y")
  full_check_epv_epm(eplo_B, epv_s) ||
    @error("different partitioned structures between eplo_B and epv_s")
  N = get_N(eplo_B)
  for i = 1:N
    eelo = get_eelo_set(eplo_B, i)
    linear = get_linear(eelo)
    if !linear
      Bi = get_Bie(eelo)
      si = get_vec(get_eev_set(epv_s, i))
      yi = get_vec(get_eev_set(epv_y, i))
      push!(Bi, si, yi)
    end
  end
  return eplo_B
end

"""
    copy_eplo_B = PLSE_update(eplo_B::Y, epv_y::Elemental_pv{T}, s::Vector{T}; kwargs...) where {T, Y <: Part_LO_mat{T}}

Perform the partitionned update PLSE onto the partitioned limited-memory operator `eplo_B`, given the step `s` (or the element-steps `epv_s`) and the difference of elemental partitioned-gradients `epv_y`.
Each element linear-operator from `eplo_B` is either a `LBFGSOperator` or a `LSR1Operator`.
The update tries to apply a LBFGS update to every Bᵢ, but if the curvature condition yᵢᵀUᵢs > 0 is not satisfied, then it replaces the `LBFGSOperator` by a `LSR1Operator` and applies a LSR1 update.
If Bᵢ is initally a LSR1Opeartor, we replace it by a `LBFGSOperator` if the curvature condition yᵢᵀUᵢs > 0 holds and we update it, otherwise the `LSR1Operator` Bᵢ remains and is updated.
Return the updated copy of `eplo_B`.
"""
function PLSE_update(
  eplo_B::Y,
  epv_y::Elemental_pv{T},
  s::Vector{T};
  kwargs...,
) where {T, Y <: Part_LO_mat{T}}
  epm_copy = copy(eplo_B)
  PLSE_update!(epm_copy, epv_y, s; kwargs...)
  return epm_copy
end

"""
    PLSE_update!(eplo_B::Y, epv_y::Elemental_pv{T}, s::Vector{T}; kwargs...) where {T, Y <: Part_LO_mat{T}}
    PLSE_update!(eplo_B::Y, epv_y::Elemental_pv{T}, epv_s::Elemental_pv{T}; ω = 1e-6, verbose=true, reset=4, kwargs...) where {T, Y <: Part_LO_mat{T}}

Perform the partitionned update PLSE onto the partitioned limited-memory operator `eplo_B`, given the step `s` (or the element-steps `epv_s`) and the difference of elemental partitioned-gradients `epv_y`.
Each element linear-operator from `eplo_B` is either a `LBFGSOperator` or a `LSR1Operator`.
The update tries to apply a LBFGS update to every Bᵢ, but if the curvature condition yᵢᵀUᵢs > 0 is not satisfied, then it replaces the `LBFGSOperator` by a `LSR1Operator` and applies a LSR1 update.
If Bᵢ is initally a LSR1Opeartor, we replace it by a `LBFGSOperator` if the curvature condition yᵢᵀUᵢs > 0 holds and we update it, otherwise the `LSR1Operator` Bᵢ remains and is updated.
"""
function PLSE_update!(
  eplo_B::Y,
  epv_y::Elemental_pv{T},
  s::Vector{T};
  kwargs...,
) where {T, Y <: Part_LO_mat{T}}
  epv_s = epv_from_v(s, epv_y)
  PLSE_update!(eplo_B, epv_y, epv_s; kwargs...)
  return eplo_B
end

function PLSE_update!(
  eplo_B::Y,
  epv_y::Elemental_pv{T},
  epv_s::Elemental_pv{T};
  ω = 1e-6,
  verbose = true,
  reset = 4,
  kwargs...,
) where {T, Y <: Part_LO_mat{T}}
  full_check_epv_epm(eplo_B, epv_y) ||
    @error("different partitioned structures between eplo_B and epv_y")
  full_check_epv_epm(eplo_B, epv_s) ||
    @error("different partitioned structures between eplo_B and epv_s")
  N = get_N(eplo_B)
  acc_lbfgs = 0
  acc_lsr1 = 0
  acc_untouched = 0
  acc_reset = 0
  for i = 1:N
    eelo = get_eelo_set(eplo_B, i)
    linear = get_linear(eelo)
    if !linear
      Bi = get_Bie(eelo)
      si = get_vec(get_eev_set(epv_s, i))
      yi = get_vec(get_eev_set(epv_y, i))
      ri = yi .- Bi * si
      index = get_index(eelo)
      if isa(Bi, LBFGSOperator{T})
        if dot(si, yi) > eps(T) # curvature condition
          push!(Bi, si, yi)
          acc_lbfgs += 1
        elseif abs(dot(si, ri)) > ω * norm(si, 2) * norm(ri, 2) # numerical condition of LSR1
          indices = get_indices(eelo)
          eelo_ = init_eelo_LSR1(indices; T = T, mem = Bi.data.mem)
          set_eelo_set!(eplo_B, i, eelo_)
          Bi = get_Bie(eelo)
          push!(Bi, si, yi)
          acc_lsr1 += 1
        elseif index < reset
          acc_untouched += 1
        else
          reset_eelo_bfgs!(eelo)
          acc_reset += 1
        end
      else # isa(Bi, LSR1Operator{T})
        if abs(dot(si, ri)) > ω * norm(si, 2) * norm(ri, 2)
          push!(Bi, si, yi)
          acc_lsr1 += 1
        elseif index < reset
          acc_untouched += 1
        else
          indices = get_indices(eelo)
          eelo_ = init_eelo_LBFGS(indices; T = T, mem = Bi.data.mem)
          set_eelo_set!(eplo_B, i, eelo_)
          acc_reset += 1
        end
      end
    end
  end
  verbose && println(
    "\n PLSE : LBFGS updates $(acc_lbfgs)/$(N), LSR1 $(acc_lsr1)/$(N), untouched $(acc_untouched)/$(N), reset $(acc_reset)/$(N)",
  )
  return eplo_B
end

end
