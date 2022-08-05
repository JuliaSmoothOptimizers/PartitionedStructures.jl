module Instances

using ..Acronyms
using ..M_part_mat, ..M_part_v
using ..ModElemental_pv,
  ..ModElemental_plo_bfgs, ..ModElemental_plo_sr1, ..ModElemental_plo, ..ModElemental_pm
using ..ModElemental_ev
using ..M_abstract_element_struct, ..M_abstract_part_struct

export create_epv_eplo,
  create_epv_eplo_bfgs, create_epv_eplo_sr1, create_epv_epm, create_epv_epm_rand

"""
    (epm,epv) = create_epv_epm(;n=9,nie=5,overlapping=1,mul_m=5., mul_v=100.)

Create an $(_epm) and an $(_epv).
Both have the same partitioned structure defined by the size of the problem `n::Int`, the size of the element `nie::Int` and the overlapping between the consecutive elements `overlapping::Int`.
Each elemental element-matrix is fill with ones, except the terms of the diagonal of value `mul_v::Real`.
The value of each elemental element-vector is `rand(nie) .* mul_v::Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
The default values are correct.
"""
function create_epv_epm(; n = 9, nie = 5, overlapping = 1, mul_m = 5.0, mul_v = 100.0)
  epm = part_mat(; n = n, nie = nie, overlapping = overlapping, mul = mul_m)
  epv = part_vec(; n = n, nie = nie, overlapping = overlapping, mul = mul_v)
  return (epm, epv)
end

"""
    (epm,epv) = create_epv_epm_rand(;n=9,nie=5,overlapping=1,range_mul_m=nie:2*nie, mul_v=100.)

Create an elemeental partitioned quasi-Newton operator `epm` and an $(_epv).
Both have the same partitioned structure defined by the size of the problem `n::Int`, the size of the element `nie::Int` and the overlapping between the consecutive elements `overlapping::Int`.
Each elemental element-matrix is fill with ones, except the terms of the diagonal of value `rand(1:range_mul_v)`.
The value of each elemental element-vector is `rand(nie) .* mul_v::Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
The default values are correct.
"""
function create_epv_epm_rand(;
  n = 9,
  nie = 5,
  overlapping = 1,
  range_mul_m = 1:(2 * nie),
  mul_v = 100.0,
)
  epm = part_mat(; n = n, nie = nie, overlapping = overlapping, mul = rand(range_mul_m))
  epv = part_vec(; n = n, nie = nie, overlapping = overlapping, mul = mul_v)
  return (epm, epv)
end

"""
    (eplo,epv) = create_epv_eplo_bfgs(;n=9,nie=5,overlapping=1, mul_v=100.)

Create an elemental partitioned limited-memory quasi-Newton operator PLBFGS `eplo` and an $(_epv).
Both have the same partitioned structure defined by the size of the problem `n::Int`, the size of the element `nie::Int` and the overlapping between the consecutive elements `overlapping::Int`.
Each elemental element-matrix is a `LBFGSOperator`.
The value of each elemental element-vector is `rand(nie) .* mul_v::Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
The default values are correct.
"""
function create_epv_eplo_bfgs(; n = 9, nie = 5, overlapping = 1, mul_v = 100.0)
  eplo = PLBFGS_eplo(; n = n, nie = nie, overlapping = overlapping)
  epv = part_vec(; n = n, nie = nie, overlapping = overlapping, mul = mul_v)
  return (eplo, epv)
end

"""
    (eplo,epv) = create_epv_eplo_sr1(;n=9,nie=5,overlapping=1, mul_v=100.)

Create an elemental partitioned limited-memory quasi-Newton operator PLSR1 `eplo` and an $(_epv).
Both have the same partitioned structure defined by the size of the problem `n::Int`, the size of the element `nie::Int` and the overlapping between the consecutive elements `overlapping::Int`.
Each elemental element-matrix is a `LSR1Operator`.
The value of each elemental element-vector is `rand(nie) .* mul_v::Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
The default values are correct.
"""
function create_epv_eplo_sr1(; n = 9, nie = 5, overlapping = 1, mul_v = 100.0)
  eplo = PLSR1_eplo(; n = n, nie = nie, overlapping = overlapping)
  epv = part_vec(; n = n, nie = nie, overlapping = overlapping, mul = mul_v)
  return (eplo, epv)
end

"""
    (eplo,epv) = create_epv_eplo_sr1(;n=9,nie=5,overlapping=1, mul_v=100.)

Create an elemental partitioned limited-memory quasi-Newton operator `eplo` and $(_epv).
Both have the same partitioned structure defined by the size of the problem `n::Int`, the size of the element `nie::Int` and the overlapping between the consecutive elements `overlapping::Int`.
Each elemental element-matrix is instantiated as a `LBFGSOperator`, but it may change to a `LSR1Operator` later on.
The value of each elemental element-vector is `rand(nie) .* mul_v::Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
The default values are correct.
"""
function create_epv_eplo(; n = 9, nie = 5, overlapping = 1, mul_v = 100.0)
  eplo = PLBFGSR1_eplo(; n = n, nie = nie, overlapping = overlapping)
  epv = part_vec(; n = n, nie = nie, overlapping = overlapping, mul = mul_v)
  return (eplo, epv)
end

end
