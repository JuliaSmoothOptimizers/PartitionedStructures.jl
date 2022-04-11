module M_elt_mat

  using ..M_abstract_element_struct

  export Elt_mat, DenseEltMat, LOEltMat
  export get_Bie, get_counter_elt_mat, get_cem, get_current_untouched, get_index

	export Counter_elt_mat
	export update_counter_elt_mat!, iter_info, total_info

	import Base.copy, Base.similar


  "Abstract type representing element matrix"
  abstract type Elt_mat{T} <: Element_struct{T} end
	abstract type DenseEltMat{T} <: Elt_mat{T} end
	abstract type LOEltMat{T} <: Elt_mat{T} end

  """
      get_Bie(elt_mat) 
  Return the element matrix elt_mat.Bie.
  """
  @inline get_Bie(elt_mat :: T) where T <: Elt_mat = elt_mat.Bie

	@inline get_counter_elt_mat(elt_mat :: T) where T <: Elt_mat = elt_mat.counter
	@inline get_cem(elt_mat :: T) where T <: Elt_mat = elt_mat.counter


	@inline get_index(elt_mat :: T) where T <: Elt_mat = get_current_untouched(elt_mat.counter)
	"""
			Counter_elt_mat
	Count for a element matrix the update performed on it, from its definition.
	`total_update + total_reset + total_untouched == iter `.
	"""
	mutable struct Counter_elt_mat
		total_update :: Int # count the total of update perform by the element linear operator
		current_update :: Int # count how many time by the element linear operator
		total_untouched :: Int
		current_untouched :: Int # must be ≤ reset defined in the update
		total_reset :: Int
		current_reset :: Int # ≤ 1 as long as reset ≥ 2 in any update performed		
	end 
	Counter_elt_mat() = Counter_elt_mat(0,0,0,0,0,0)
	copy(cem :: Counter_elt_mat) = Counter_elt_mat(cem.total_update, cem.current_update, cem.total_untouched, cem.current_untouched, cem.total_reset, cem.current_reset)
	similar(cem :: Counter_elt_mat) = Counter_elt_mat()

	get_current_untouched(cem :: Counter_elt_mat) = cem.current_untouched

	iter_info(cem :: Counter_elt_mat) = (cem.current_update, cem.current_untouched, cem.current_reset)
	total_info(cem :: Counter_elt_mat) = (cem.total_update, cem.total_untouched, cem.current_reset)

	"""
		update_counter_elt_mat!(cem, qn)
	Update the `cem` counter given the index `qn` from the quasi-Newton update.
	"""
	function update_counter_elt_mat!(cem :: Counter_elt_mat, qn :: Int)
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
		else # qn == -1
			cem.total_reset += 1
			cem.current_reset += 1
			cem.current_untouched = 0
			cem.current_update = 0
		end
	end

end