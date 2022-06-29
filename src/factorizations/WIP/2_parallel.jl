module M_2_parallel

  using LinearAlgebra, Statistics
  using ..M_abstract_element_struct, ..M_abstract_part_struct, ..M_elt_mat, ..M_part_mat, ..ModElemental_pm, ..ModElemental_em
  using ..M_elt_vec, ..M_part_v, ..ModElemental_ev, ..ModElemental_pv
  using ..Link

  export second_parallel, second_parallel!

  function second_parallel(epm_A::Elemental_pm{T}, epv_b::Elemental_pv{T}) where T
    epv_x = similar(epv_b)
    res = Vector{T}(undef, get_n(epm_A))
    second_parallel!(epm_A, epv_b, epv_x, res)
    return res
  end

  function second_parallel!(epm_A::Elemental_pm{T}, epv_b::Elemental_pv{T}, epv_x::Elemental_pv{T}, res::Vector{T}) where T
    check_epv_epm(epm_A, epv_b)
    N = get_N(epm_A)
    n = get_n(epm_A)
    length(res)==n || @error("wrong size res first_parallel!")
    vector_bool = zeros(Bool,n)

    #résolution de chaque system linéaire élément
    for i in [1:N;]
      set_eev!(epv_x, i, get_eem_set_Bie(epm_A,i)\get_eev_value(epv_b,i))
    end
    #trouver les indices du gradient
    grad = Vector(epv_b)
    # order = reverse(sortperm(grad))
    order = sortperm(grad)

    #procédure pour chaque coordonnée par ordre d'importance du gradient
    for index in order
      subproblem!(epm_A, epv_b, epv_x, index, vector_bool, res)
    end

    return res
  end

  """
      subproblem(epm_A, epv_b, epv_x, comp_list, i)
  define the subproblem which must be solve for the i-th variable
  """
  function subproblem!(epm_A::Elemental_pm{T}, epv_b::Elemental_pv{T}, epv_x::Elemental_pv{T}, index::Int, vector_bool:: Vector{Bool}, res::Vector{T}) where T
    comp_list = ModElemental_pm.get_component_list(epm_A,index) # element list using tha i-th variable
    _x = Vector{T}(undef,length(comp_list))
    ss_epm_A = get_eem_sub_set(epm_A, comp_list)
    ss_epv_x = get_eev_subset(epv_x, comp_list)
    _indices = get_indices.(ss_epv_x)
    __indices = mapreduce((x->x),((indice,indicei) -> vcat(indice, indicei)), _indices)
    every_indices = unique(__indices)
    other_indices = filter((idx -> vector_bool[idx]),every_indices)

    _indices = Vector{Int}(undef,length(comp_list))
    _columns1 = Vector{Elemental_elt_vec{T}}(undef,length(comp_list))
    for (idx,val) in enumerate(comp_list)
      eev = get_eev_set(epv_x, val) # retrieve elemental element-vector
      _indices[idx] = findfirst((id->id==index), eev.indices) # find the corresponding index
      _x[idx] = get_vec(eev,_indices[idx]) # store the result
      eem = get_eem_set(epm_A,val)
      vec = get_Bie(eem)[:,_indices[idx]] #la colonne de A associé à la variable xᵢ
      indices = M_elt_mat.get_indices(eem)
      nie = M_elt_mat.get_nie(eem)
      _columns1[idx] = Elemental_elt_vec{T}(vec,indices,nie)
    end
    tmp_epv = create_epv(_columns1)

    _columns2 = Vector{Elemental_elt_vec{T}}([])
    other_scalars = Vector{T}([]) # constant vecteur
    for i in other_indices
      for (idx,val) in enumerate(comp_list)
        eev = get_eev_set(epv_x, val) # retrieve elemental element-vector
        tmp = findfirst((id->id==i), eev.indices) # find the corresponding index
        if tmp != nothing
          push!(other_scalars, get_vec(eev,tmp) - res[i]) # the difference between the element linear system solution and the current solution
          eem = get_eem_set(epm_A,val)
          vec = get_Bie(eem)[:,tmp] #la colonne de A associé à la variable xᵢ
          indices = get_indices(eem)
          nie = get_nie(eem)
          push!(_columns2, Elemental_elt_vec{T}(vec,indices,nie))
        end
      end
    end
    # other_epv = create_epv(_columns2)

    _columns = vcat(_columns1,_columns2)
    sub_prob_epv = create_epv(_columns)

    first_scalars(xi) = (i -> _x[i]-xi).([1:length(comp_list);])
    every_scalars(xi) = vcat(first_scalars(xi), other_scalars)

    f(xi) = sum( (y->y^2).(scale_epv(sub_prob_epv, every_scalars(xi))))

    multiplicator = (λᵢ -> 2*λᵢ).(scale_epv(tmp_epv, ones(T,length(comp_list))))
    val(xi) = scale_epv(sub_prob_epv, every_scalars(xi))
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

    vector_bool[index]= true
    res[index] = xi_opt
    return xi_opt
  end

end