module M_frontale

  using ..M_part_mat, ..ModElemental_pm

  """
      frontale!(epm)
  Produce the Cholesky frontal factorization of the elemental partitioned matrix `epm`.
  The sparse factor `L` is stored in `epm.L`.
  """
  function frontale!(epm :: Elemental_pm{T}; perm::Vector{Int}=[1:get_n(epm);]) where T	
    if perm != [1:get_n(epm);]
      permute!(epm, perm) # apply the permutation
    end
    set_spm!(epm) #(re)-build the sparse matrix of epm
    set_L_to_spm!(epm) # copy on spm on L
    

    N = get_N(epm)
    n = get_n(epm)

    not_treated = ones(Bool,n)
    not_added = ones(Bool,n)
    front = Vector{Int}(undef,0)

    for _current_var in 1:n
      # Front update
      push!(front,_current_var)
      crl_var = correlated_var(epm, _current_var) # getting correlated var from the blocs
      needed_var = select_var(crl_var, not_treated) # filtering these correlated var to maintain order
      front = vcat(front, needed_var)
      sort!(front) # maintaining order
      unique!(front) # without duplication
      actualise_not_added!(front, not_added) # update of boolean list

      for j in front
        if j == _current_var # pivot (1st iter)
          v = sqrt(get_L(epm,_current_var,_current_var))
          set_L!(epm,_current_var,_current_var,v)
        else # terms below the pivot
          f_cr_j = get_L(epm,j,_current_var) / get_L(epm,_current_var,_current_var)
          set_L!(epm,j,_current_var,f_cr_j)
          # iterative update of the front
          for up_var in _current_var+1:j-1
            v_up = get_L(epm,j,up_var) - ( f_cr_j * get_L(epm,up_var,_current_var))
            set_L!(epm,j,up_var,v_up)
          end 
          # update of the j-th pivot
          v_up = get_L(epm,j,j) - f_cr_j^2
          set_L!(epm,j,j,v_up)
        end 						
      end 			
      not_treated[_current_var] = false	# update of boolean list
      actualise_front!(front, not_treated) # deleting the _current_var of the front 
    end
    return get_L(epm)
  end

  select_var(crl_var :: Vector{Int}, not_treated :: Vector{Bool}) = filter!(var -> not_treated[var], crl_var)

  actualise_not_added!(front :: Vector{Int}, not_added :: Vector{Bool}) = map(i -> not_added[i] = false, front)
  actualise_front!(front :: Vector{Int}, not_treated :: Vector{Bool}) = filter!(var -> not_treated[var], front)
  
  export frontale!

end 