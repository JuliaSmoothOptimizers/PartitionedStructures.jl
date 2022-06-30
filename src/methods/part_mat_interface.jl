module PartMatInterface

using ..M_part_mat, ..M_part_v
using ..ModElemental_em, ..ModElemental_ev
using ..ModElemental_pv, ..ModElemental_plo_bfgs, ..ModElemental_plo_sr1, ..ModElemental_plo, ..ModElemental_pm
using ..M_abstract_element_struct, ..M_abstract_part_struct
using ..PartitionedLOQuasiNewton, ..PartitionedQuasiNewton

export update, update!

"""
    B = update(epm, epv_y, s)

Update the elemental partitioned operator `epm<:Part_mat` with a partitioned quasi-Newton update considering the difference of elemental partitioned-gradients `epv_y` and the step `s`.
If `epm` is an elemental partitioned-matrix, the PSE update is run by default.
You can apply a PBFGS or a PSR1 update with the optionnal argument `name`, respectively `name=:pbfgs` or `name=:psr1`.
It returns a matrix formed from the updated `epm`.
Warning: this method should be use to test your algorithm, if you don't intend to form the matrix use `update!(epm, epv_y, s)`.
"""
@inline function update(epm::T, epv_y::Elemental_pv{Y}, s::Vector{Y}; kwargs...) where T<:Part_mat{Y} where Y<:Number
  update!(epm, epv_y, s; kwargs...)
  return Matrix(epm)
end

"""
    update!(epm, epv_y, s)
    update!(epm, epv_y, epv_s)

Update the elemental partitioned operator `epm` with a partitioned quasi-Newton update considering the difference of elemental partitioned-gradients `epv_y` and the step `s` (or elemental steps `epv_s`).
The PSE update is run by default, you can apply a PBFGS or a PSR1 update with the optionnal argument `name`, respectively `name=:pbfgs` or `name=:psr1`.
"""
@inline update!(epm::T, epv_y::Elemental_pv{Y}, s::Vector{Y}; kwargs...) where T<:Part_mat{Y} where Y<:Number= update!(epm, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline function update!(epm::T, epv_y::Elemental_pv{Y}, epv_s::Elemental_pv{Y}; name=:pse, kwargs...) where T<:Part_mat{Y} where Y<:Number
  (name==:pse) && PSE_update!(epm, epv_y, epv_s; kwargs...)
  (name==:pbfgs) && PBFGS_update!(epm, epv_y, epv_s; kwargs...)
  (name==:psr1) && PSR1_update!(epm, epv_y, epv_s; kwargs...)
  return epm
end

"""
    update!(eplo, epv_y, s)
    update!(eplo, epv_y, epv_s)

Updates the limited-memory elemental partitioned operator `eplo` with the partitioned quasi-Newton update PLSE considering the difference of elemental partitioned-gradients `epv_y` and the step `s` (or elemental steps `epv_s`).
"""
@inline update!(eplo::Elemental_plo{Y}, epv_y::Elemental_pv{Y}, s::Vector{Y}; kwargs...) where Y<:Number = update!(eplo, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline update!(eplo::Elemental_plo{Y}, epv_y::Elemental_pv{Y}, epv_s::Elemental_pv{Y}; kwargs...) where  Y<:Number = PLSE_update!(eplo, epv_y, epv_s; kwargs...)

"""
    update(eplo, epv_y, s)
    update(eplo, epv_y, epv_s)

Updates the limited-memory elemental partitioned operator `eplo` with the partitioned quasi-Newton update PLSR1 considering the difference of elemental partitioned-gradients `epv_y` and the step `s` (or elemental steps `epv_s`).
"""
@inline update!(eplo::Elemental_plo_sr1{Y}, epv_y::Elemental_pv{Y}, s::Vector{Y}; kwargs...) where Y<:Number = update!(eplo, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline update!(eplo::Elemental_plo_sr1{Y}, epv_y::Elemental_pv{Y}, epv_s::Elemental_pv{Y}; kwargs...) where  Y<:Number = PLSR1_update!(eplo, epv_y, epv_s; kwargs...)

"""
    update(eplo, epv_y, s)
    update(eplo, epv_y, epv_s)

Updates the limited-memory elemental partitioned operator `eplo` with the partitioned quasi-Newton update PLBFGS considering the difference of elemental partitioned-gradients `epv_y` and the step `s` (or elemental steps `epv_s`).
"""
@inline update!(eplo::Elemental_plo_bfgs{Y}, epv_y::Elemental_pv{Y}, s::Vector{Y}; kwargs...) where Y<:Number = update!(eplo, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline update!(eplo::Elemental_plo_bfgs{Y}, epv_y::Elemental_pv{Y}, epv_s::Elemental_pv{Y}; kwargs...) where  Y<:Number = PLBFGS_update!(eplo, epv_y, epv_s; kwargs...)

end