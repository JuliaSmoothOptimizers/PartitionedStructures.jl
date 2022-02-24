module M_elt_mat

  using ..M_abstract_element_struct

  export Elt_mat
  export get_Bie

  "Abstract type representing element matrix"
  abstract type Elt_mat{T} <: Element_struct{T} end

  """
      get_Bie(elt_mat) 
  Return the element matrix elt_mat.Bie.
  """
  @inline get_Bie(elt_mat :: T) where T <: Elt_mat = elt_mat.Bie
  
end