module M_part_mat
  using SparseArrays
  using ..M_abstract_part_struct, ..M_elt_mat

  export Part_mat, Part_LO_mat
  export get_N, get_n, get_permutation, get_spm
  export hard_reset_spm!, reset_spm!, set_spm!
  export hard_reset_L!, reset_L!
  export set_N!, set_n!, set_permutation!	
  export get_eelom_set, get_ee_struct_Bie

  "Abstract type representing partitioned matrix"
  abstract type Part_mat{T} <: Part_struct{T} end
  "Abstract type representing partitioned matrix using linear operators"
  abstract type Part_LO_mat{T} <: Part_mat{T} end

  @inline set_spm!(pm :: T) where T <: Part_mat = @error("should not be called")

  """
      get_spm(pm)

      get_spm(pm, i, j)

  Get either the sparse matrix associated to the partitioned matrix `pm` or ones ot its element at coordinate `[i,j]`.
  """
  @inline get_spm(pm :: T) where T <: Part_mat = pm.spm
  @inline get_spm(pm :: T, i :: Int, j :: Int) where T <: Part_mat = @inbounds get_spm(pm)[i,j]

  @inline get_permutation(pm :: T) where T <: Part_mat = pm.permutation
  @inline set_permutation!(pm :: T, perm :: Vector{Int}) where T <: Part_mat = pm.permutation .= perm

  """
      reset_spm!(pm)
  Set the elements of sparse matrix `pm.spm` to `0`.
  """
  @inline reset_spm!(pm :: T) where T <: Part_mat{Y} where Y <: Number  = pm.spm.nzval .= (Y)(0)

  """
      hard_reset_spm!(pm)
  Reset the sparse matrix `pm.spm`.
  """
  @inline hard_reset_spm!(pm :: T) where T <: Part_mat = pm.spm = spzeros(T, get_n(pm), get_n(pm))

  """
      reset_L!(pm)
  Set the elements of sparse matrix `pm.L` to `0`.
  """
  @inline reset_L!(pm :: T) where T <: Part_mat{Y} where Y <: Number = pm.L.nzval .= (Y)(0)

  """
      hard_reset_L!(pm)
  Reset the sparse matrix `pm.L`.
  """
  @inline hard_reset_L!(pm :: T) where T <: Part_mat = pm.L = spzeros(T, get_n(pm), get_n(pm))

  @inline get_eelom_set(plm :: T) where T <: Part_LO_mat = @error("should not be called")
  @inline get_ee_struct_Bie(pm :: T, i :: Int) where T <: Part_mat = get_Bie(get_ee_struct(pm, i))
end