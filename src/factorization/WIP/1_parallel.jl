module M_1_parallel

using ..M_part_mat, ..M_elt_mat, ..M_elemental_pm, ..M_elemental_em
using ..M_part_v, ..M_elemental_pv, ..M_elt_vec, ..M_elemental_elt_vec
using ..M_link

using LinearAlgebra, Statistics
# using Ipopt, ADNLPModels, NLPModelsIpopt 
# using JuMP
# using BlackBoxOptim, NaturalES


	function first_parallel(epm_A :: Elemental_pm{T}, epv_b :: Elemental_pv{T}) where T	
		epv_x = similar(epv_b)
		res = Vector{T}(undef, get_n(epm_A))
		first_parallel!(epm_A, epv_b, epv_x, res)
		return res
	end 

	function first_parallel!(epm_A :: Elemental_pm{T}, epv_b :: Elemental_pv{T}, epv_x :: Elemental_pv{T}, res :: Vector{T}) where T
		check_epv_epm(epm_A, epv_b)
		N = get_N(epm_A)
		n = get_n(epm_A)
		length(res) == n || @error("wrong size res first_parallel!")

		#résolution de chaque system linéaire élément
		for i in [1:N;]
			set_eev!(epv_x, i, get_eem_set_hie(epm_A,i)\get_eev_value(epv_b,i))
		end 
		#procédure pour chaque coordonnée
		# grad = Vector(epv_b); order = reverse(sortperm(grad)); for i in order # ne change rien car complètement parallèle.
		for i in 1:n		
			_comp_list = M_elemental_pm.get_component_list(epm_A,i) # element list using tha i-th variable
			if length(_comp_list)==1 # in case only one element uses it
				eev = get_eev(epv_x, _comp_list[1]) # retrieve elemental element vector
				j = findfirst((index->index==i), eev.indices) # find the corresponding index 
				res[i] = get_vec(eev,j) # store the result
			else 
				xi_opt = subproblem(epm_A, epv_b, epv_x, _comp_list, i)		
				res[i] = xi_opt		
			end 
		end 
	end 

	"""
			subproblem(epm_A, epv_b, epv_x, comp_list, i) 
	define the subproblem which must be solve for the i-th variable
	"""
	function subproblem(epm_A :: Elemental_pm{T}, epv_b :: Elemental_pv{T}, epv_x :: Elemental_pv{T}, comp_list::Vector{Int}, i :: Int) where T
		_x = Vector{T}(undef,length(comp_list))
		_indices = Vector{Int}(undef,length(comp_list))
		_columns = Vector{Elemental_elt_vec{T}}(undef,length(comp_list))
		for (idx,val) in enumerate(comp_list)
			eev = get_eev(epv_x, val) # retrieve elemental element vector
			_indices[idx] = findfirst((index->index==i), eev.indices) # find the corresponding index 
			_x[idx] = get_vec(eev,_indices[idx]) # store the result
			eem = get_eem_set(epm_A,val)
			vec = get_hie(eem)[:,_indices[idx]] #la colonne de A associé à la variable xᵢ
			indices = M_elt_mat.get_indices(eem)
			nie = M_elt_mat.get_nie(eem)
			_columns[idx] = Elemental_elt_vec{T}(vec,indices,nie)
		end
		tmp_epv = create_epv(_columns)

		f(xi) = sum( (y->y^2).(scale_epv(tmp_epv, (i -> _x[i]-xi).([1:length(comp_list);]))) )
		multiplicator = (λᵢ -> 2*λᵢ).(scale_epv(tmp_epv, ones(T,length(comp_list))))
		val(xi) = scale_epv(tmp_epv, (i -> _x[i]-xi).([1:length(comp_list);]))
		f_prim(xi) = sum(multiplicator .* val(xi))
		f_seconde() = sum((v -> 2*(v^2)).(scale_epv(tmp_epv, ones(T,length(comp_list)))))
		
		# _bary = mean(_x)
		# xk = _bary
		xk = 1.

		# @show _x
		# @show _bary, f(_bary), f_prim(_bary), f_seconde()
		s1 = f_prim(xk)/f_seconde()
		s2 = -f_prim(xk)/f_seconde()
		# @show s1,s2
		# @show f(s1),f(s2)

		# @show f(_bary + s1), f(_bary + s2)
		if f(xk + s1) < f(xk + s2)
			xi_opt = xk + s1	
		else 			
			xi_opt = xk + s2
		end 
		
		# x = rand(T,length(comp_list))
		# diff_comp_list = (i -> _x[i]-x[i]).([1:length(comp_list);])
		# res = scale_epv(tmp_epv, diff_comp_list)
		# @show x, diff_comp_list, res
		#fonction à minimiser 
		# sub_prob(xi::T) = norm(scale_epv(tmp_epv, (i -> _x[i]-xi).([1:length(comp_list);])))
		# _bary = mean(_x)
		# @show f(_bary)
		# @show _bary, sub_prob(_bary)
		# xi_opt = bboptimize(sub_prob; SearchRange = (_bary-10.0, _bary+10.0), NumDimensions = 1)
		# xi_opt = optimize(sub_prob, _bary, 1,xNES)
		# model = Model(with_optimizer(Ipopt.Optimizer))
		# @variable(model, xi)
		# f(xi) = sum( (y->y^2).(scale_epv(tmp_epv, (i -> _x[i]-xi).([1:length(comp_list);]))) )
		# register(model, :f, 1, f; autodiff = true)
		# @NLobjective(model, Min, f(xi) )
		# optimize!(model)
		# xi_opt = value(xi)

		return xi_opt		
	end 

	export first_parallel, first_parallel!

end 