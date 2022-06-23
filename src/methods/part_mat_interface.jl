module PartMatInterface

using ..M_part_mat, ..M_part_v
using ..ModElemental_em, ..ModElemental_ev
using ..ModElemental_pv, ..ModElemental_plom_bfgs, ..ModElemental_plom_sr1, ..ModElemental_plom, ..ModElemental_pm  
using ..M_abstract_element_struct, ..M_abstract_part_struct
using ..PartitionedLOQuasiNewton, ..PartitionedQuasiNewton

export update, update!

"""
    B = update(epm, epv_y, s)

Updates the elemental partitioned operator `epm <: Part_mat` with a partitioned quasi-Newton update considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
If `epm` is an elemental partitioned matrix, the PSE update is run by default.
You can apply a PBFGS or a PSR1 update with the optionnal argument `name`, respectively `name=:pbfgs` or `name=:psr1`.
It returns a matrix formed from the updated `epm`.
Warning: this method should be use to test your algorithm, if you don't intend to form the matrix use `update!`.
"""
@inline function update(epm :: T, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where T <: Part_mat{Y} where Y <: Number
  update!(epm, epv_y, s; kwargs...)
  return Matrix(epm)
end 

"""
    update!(epm, epv_y, s)

Updates the elemental partitioned operator `epm` with a partitioned quasi-Newton update considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
The PSE update is run by default, you can apply a PBFGS or a PSR1 update with the optionnal argument `name`, respectively `name=:pbfgs` or `name=:psr1`.
"""
@inline update!(epm :: T, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where T <: Part_mat{Y} where Y <: Number= update!(epm, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline function update!(epm :: T, epv_y :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}; name=:pse, kwargs...) where T <: Part_mat{Y} where Y <: Number
  (name == :pse) && PSE_update!(epm, epv_y, epv_s; kwargs...)
  (name == :pbfgs) && PBFGS_update!(epm, epv_y, epv_s; kwargs...)
  (name == :psr1) && PSR1_update!(epm, epv_y, epv_s; kwargs...)
  return epm
end

"""
    update!(eplom, epv_y, s)

Updates the limited-memory elemental partitioned operator `eplom` with the partitioned quasi-Newton update PLSE considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
"""
@inline update!(eplom :: Elemental_plom{Y}, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where Y <: Number = update!(eplom, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline update!(eplom :: Elemental_plom{Y}, epv_y :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}; kwargs...) where  Y <: Number = PLSE_update!(eplom, epv_y, epv_s; kwargs...)

"""
    update(eplom, epv_y, s)

Updates the limited-memory elemental partitioned operator `eplom` with the partitioned quasi-Newton update PLSR1 considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
"""
@inline update!(eplom :: Elemental_plom_sr1{Y}, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where Y <: Number = update!(eplom, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline update!(eplom :: Elemental_plom_sr1{Y}, epv_y :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}; kwargs...) where  Y <: Number = PLSR1_update!(eplom, epv_y, epv_s; kwargs...)
  
"""
    update(eplom, epv_y, s)

Updates the limited-memory elemental partitioned operator `eplom` with the partitioned quasi-Newton update PLBFGS considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
"""
@inline update!(eplom :: Elemental_plom_bfgs{Y}, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where Y <: Number = update!(eplom, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline update!(eplom :: Elemental_plom_bfgs{Y}, epv_y :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}; kwargs...) where  Y <: Number = PLBFGS_update!(eplom, epv_y, epv_s; kwargs...)
    
end