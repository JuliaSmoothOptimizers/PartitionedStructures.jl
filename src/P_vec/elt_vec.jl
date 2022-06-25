module M_elt_vec

  using ..M_abstract_element_struct

  export Elt_vec
  export get_vec, set_vec!
  export set_add_vec!, set_minus_vec!

  """Abstract type representing element vectors."""
  abstract type Elt_vec{T} <: Element_struct{T} end

  #generic getter/setter
  @inline get_vec(ev :: T) where T <: Elt_vec = ev.vec
  @inline get_vec(ev :: T, i::Int) where T <: Elt_vec = ev.vec[i]

  @inline set_vec!(ev :: T, vec :: Vector{Y}) where T <: Elt_vec{Y} where Y = ev.vec .= vec
  @inline set_vec!(ev :: T, i :: Int, val :: Y) where T <: Elt_vec{Y} where Y <: Number = ev.vec[i] = val

  @inline set_minus_vec!(ev :: T) where T <: Elt_vec = set_vec!(ev, - get_vec(ev))
  @inline set_add_vec!(ev :: T, vec :: Vector{Y}) where {T <: Elt_vec, Y <:Number}= ev.vec .+= vec

end
