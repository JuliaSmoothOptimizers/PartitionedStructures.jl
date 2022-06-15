module PartMatInterface

using ..M_part_mat, ..M_part_v
using ..ModElemental_em, ..ModElemental_ev
using ..ModElemental_pv, ..ModElemental_plom_bfgs, ..ModElemental_plom_sr1, ..ModElemental_plom, ..ModElemental_pm  
using ..M_abstract_element_struct, ..M_abstract_part_struct
using ..PartitionedLOQuasiNewton, ..PartitionedQuasiNewton

export update, update!

"""
		update(epm, epv_y, s)

Update the elemental partitioned matrix `epm` with a partitioned quasi-Newton update considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
"""
@inline function update(epm :: T, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where T <: Part_mat{Y} where Y <: Number
	update!(epm, epv_y, s; kwargs...)
	return Matrix(epm)
end 

"""
		update!(epm, epv_y, s)

Update the elemental partitioned matrix `epm` with a partitioned quasi-Newton update considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
"""
@inline update!(epm :: T, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where T <: Part_mat{Y} where Y <: Number= update!(epm, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline function update!(epm :: T, epv_y :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}; name=:pbfgs, kwargs...) where T <: Part_mat{Y} where Y <: Number
	(name == :pbfgs) && PBFGS_update!(epm, epv_y, epv_s; kwargs...)
	(name == :psr1) && PSR1_update!(epm, epv_y, epv_s; kwargs...)
	(name == :pse) && PSE_update!(epm, epv_y, epv_s; kwargs...)
end

"""
		update(epm, epv_y, s)

Update the limited-memory elemental partitioned matrix `epm` with the partitioned quasi-Newton update PLSE considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
"""
@inline update!(epm :: Elemental_plom{Y}, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where Y <: Number = update!(epm, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline update!(epm :: Elemental_plom{Y}, epv_y :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}; kwargs...) where  Y <: Number = PLSE_update!(epm, epv_y, epv_s; kwargs...)

"""
		update(epm, epv_y, s)

Update the limited-memory elemental partitioned matrix `epm` with the partitioned quasi-Newton update PLSR1 considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
"""
@inline update!(epm :: Elemental_plom_sr1{Y}, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where Y <: Number = update!(epm, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline update!(epm :: Elemental_plom_sr1{Y}, epv_y :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}; kwargs...) where  Y <: Number = PLSR1_update!(epm, epv_y, epv_s; kwargs...)
	
"""
		update(epm, epv_y, s)

Update the limited-memory elemental partitioned matrix `epm` with the partitioned quasi-Newton update PLSR1 considering the difference of elemental partitioned gradients `epv_y` and the step `s`.
"""
@inline update!(epm :: Elemental_plom_bfgs{Y}, epv_y :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where Y <: Number = update!(epm, epv_y, epv_from_v(s, epv_y); kwargs...)
@inline update!(epm :: Elemental_plom_bfgs{Y}, epv_y :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}; kwargs...) where  Y <: Number = PLBFGS_update!(epm, epv_y, epv_s; kwargs...)
		
end