module M_abstract_part_struct

import Base.==

abstract type Part_struct{T} end 

get_n(ps :: T) where T <: Part_struct = ps.n
get_N(ps :: T) where T <: Part_struct = ps.N

(==)(ps1 :: T, ps2 :: T) where T <: Part_struct = get_n(ps1)==get_n(ps2) && get_N(ps1)==get_N(ps2)
export Part_struct

export get_n, get_N

end