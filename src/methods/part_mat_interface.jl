module PartMatInterface

	using ..M_part_mat, ..M_part_v
  using ..ModElemental_em, ..ModElemental_ev
  using ..ModElemental_pv, ..ModElemental_plom_bfgs, ..ModElemental_plom_sr1, ..ModElemental_plom, ..ModElemental_pm  
  using ..M_abstract_element_struct, ..M_abstract_part_struct
	using ..PartitionedLOQuasiNewton, ..PartitionedQuasiNewton

	export update, update!

	@inline function update(epm :: T, epv_res :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where T <: Part_mat{Y} where Y <: Number
		update!(epm, epv_res, s; kwargs...)
		return Matrix(epm)
	end 
	
	@inline update!(epm :: T, epv_res :: Elemental_pv{Y}, s :: Vector{Y}; kwargs...) where T <: Part_mat{Y} where Y <: Number= update!(epm, epv_res, epv_from_v(s, epv_res); kwargs...)
	@inline function update!(epm :: T, epv_res :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}; name=:pbfgs) where T <: Part_mat{Y} where Y <: Number
		(name == :pbfgs) && PBFGS_update!(epm, epv_res, epv_s)
		(name == :psr1) && PSR1_update!(epm, epv_res, epv_s)
	end
	
	@inline update!(epm :: Elemental_plom{Y}, epv_res :: Elemental_pv{Y}, s :: Vector{Y}) where Y <: Number = update!(epm, epv_res, epv_from_v(s, epv_res))
	@inline update!(epm :: Elemental_plom{Y}, epv_res :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}) where  Y <: Number = PLSE_update!(epm, epv_res, epv_s)
	
	@inline update!(epm :: Elemental_plom_sr1{Y}, epv_res :: Elemental_pv{Y}, s :: Vector{Y}) where Y <: Number = update!(epm, epv_res, epv_from_v(s, epv_res))
	@inline update!(epm :: Elemental_plom_sr1{Y}, epv_res :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}) where  Y <: Number = PLSR1_update!(epm, epv_res, epv_s)
		
	@inline update!(epm :: Elemental_plom_bfgs{Y}, epv_res :: Elemental_pv{Y}, s :: Vector{Y}) where Y <: Number = update!(epm, epv_res, epv_from_v(s, epv_res))
	@inline update!(epm :: Elemental_plom_bfgs{Y}, epv_res :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}) where  Y <: Number = PLBFGS_update!(epm, epv_res, epv_s)
	
	@inline update!(epm :: Elemental_plom{Y}, epv_res :: Elemental_pv{Y}, s :: Vector{Y}) where Y <: Number = update!(epm, epv_res, epv_from_v(s, epv_res))
	@inline update!(epm :: Elemental_plom{Y}, epv_res :: Elemental_pv{Y}, epv_s :: Elemental_pv{Y}) where  Y <: Number = Part_update!(epm, epv_res, epv_s)
				
end