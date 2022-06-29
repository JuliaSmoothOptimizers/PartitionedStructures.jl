module M_okoubi_koko

using LinearAlgebra, Statistics

using ..M_abstract_part_struct, ..M_elt_vec, ..M_part_mat, ..M_part_v, ..ModElemental_pm, ..ModElemental_pv
using ..Link

"""
    okoubi(epm, epv)

Solve the partitioned linear problem from `epm` and `epv` using the [Okoubi and Koko method](https://doi.org/10.1007/s11590-016-1020-x).
"""
function okoubi(epm_A::Elemental_pm{T}, epv_b::Elemental_pv{T}) where T
  epv_x = similar(epv_b)
  res = Vector{T}(undef, get_n(epm_A))
  okoubi!(epm_A, epv_b, epv_x, res)
  return res
end

"""
    okoubi!(epm, epv, epv_tmp, res)

Solve the partitioned linear problem using the [Okoubi and Koko method](https://doi.org/10.1007/s11590-016-1020-x).
This method avoid the allocation of `epv_tmp` and `res`.
"""
function okoubi!(epm_A::Elemental_pm{T}, epv_b::Elemental_pv{T}, epv_x::Elemental_pv{T}, res::Vector{T}) where T
  check_epv_epm(epm_A, epv_b)
  N = get_N(epm_A)
  n = get_n(epm_A)
  length(res)==n || @error("wrong size res first_parallel!")

  # solve each element linear system
  for i in [1:N;]
    set_eev!(epv_x, i, get_eem_set_Bie(epm_A,i)\get_eev_value(epv_b,i))
  end
  # compute the final result of each coordinate
  for i in 1:n
    _comp_list = ModElemental_pm.get_component_list(epm_A,i) # element list using tha i-th variable
    if length(_comp_list)==1 # in case only one element uses it
      eev = get_eev_set(epv_x, _comp_list[1]) # retrieve elemental element-vector
      j = findfirst((index->index==i), eev.indices) # find the corresponding index
      res[i] = get_vec(eev,j) # store the result
    else
      s = Vector{T}(undef,length(_comp_list))
      for (idx,val) in enumerate(_comp_list)
        eev = get_eev_set(epv_x, val) # retrieve elemental element-vector
        j = findfirst((index->index==i), eev.indices) # find the corresponding index
        s[idx] = get_vec(eev,j) # store the results
      end
      res[i] = mean(s)
    end
  end
end

export okoubi, okoubi!

end