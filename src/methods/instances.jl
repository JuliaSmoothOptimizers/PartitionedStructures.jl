module Instances

using ..M_part_mat, ..M_part_v
using ..ModElemental_pv, ..ModElemental_plom_bfgs, ..ModElemental_plom_sr1, ..ModElemental_plom, ..ModElemental_pm
using ..ModElemental_ev
using ..M_abstract_element_struct, ..M_abstract_part_struct

export create_epv_eplom, create_epv_eplom_bfgs, create_epv_eplom_sr1, create_epv_epm, create_epv_epm_rand

"""
    (epm,epv) = create_epv_epm(;n=9,nie=5,overlapping=1,mul_m=5., mul_v=100.)

Create an elemental partitioned-matrix `epm` and an elemental partitioned-vector `epv`.
Both have the same partitioned structure defined by the size of the problem `n :: Int`, the size of the element `nie :: Int` and the overlapping between the consecutive elements `overlapping :: Int`.
Each elemental element-matrix is fill with ones, except the terms of the diagonal which are of value `mul_v :: Real`.
The value of each elemental element-vector is made `rand(nie) .* mul_v :: Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
"""
function create_epv_epm(;n=9,nie=5,overlapping=1,mul_m=5., mul_v=100.)
  epm = part_mat(;n=n,nie=nie,overlapping=overlapping,mul=mul_m)
  epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
  return (epm,epv)
end

"""
    (epm,epv) = create_epv_epm_rand(;n=9,nie=5,overlapping=1,range_mul_m=nie:2*nie, mul_v=100.)

Create an elemeental partitioned quasi-Newton operator `epm` and an elemental partitioned-vector `epv`.
Both have the same partitioned structure defined by the size of the problem `n :: Int`, the size of the element `nie :: Int` and the overlapping between the consecutive elements `overlapping :: Int`.
Each elemental element-matrix is fill with ones, except the terms of the diagonal of `rand(range_mul_v)`.
The value of each elemental element-vector is `rand(nie) .* mul_v :: Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
"""
function create_epv_epm_rand(;n=9,nie=5,overlapping=1,range_mul_m=nie:2*nie, mul_v=100.)
  epm = part_mat(;n=n,nie=nie,overlapping=overlapping,mul=rand(range_mul_m))
  epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
  return (epm,epv)
end

"""
    (eplom,epv) = create_epv_eplom_bfgs(;n=9,nie=5,overlapping=1, mul_v=100.)

Create an elemental partitioned limited-memory quasi-Newton operator PLBFGS `eplom` and an elemental partitioned-vector `epv`.
Both have the same partitioned structure defined by the size of the problem `n :: Int`, the size of the element `nie :: Int` and the overlapping between the consecutive elements `overlapping :: Int`.
Each elemental element-matrix is a `LBFGSOperator`.
The value of each elemental element-vector is `rand(nie) .* mul_v :: Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
"""
function create_epv_eplom_bfgs(;n=9,nie=5,overlapping=1, mul_v=100.)
  eplom = PLBFGS_eplom(;n=n,nie=nie,overlapping=overlapping)
  epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
  return (eplom,epv)
end

"""
    (eplom,epv) = create_epv_eplom_sr1(;n=9,nie=5,overlapping=1, mul_v=100.)

Create an elemental partitioned limited-memory quasi-Newton operator PLSR1 `eplom` and an elemental partitioned-vector `epv`.
Both have the same partitioned structure defined by the size of the problem `n :: Int`, the size of the element `nie :: Int` and the overlapping between the consecutive elements `overlapping :: Int`.
Each elemental element-matrix is a `LSR1Operator`.
The value of each elemental element-vector is `rand(nie) .* mul_v :: Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
"""
function create_epv_eplom_sr1(;n=9,nie=5,overlapping=1, mul_v=100.)
eplom = PLSR1_eplom(;n=n,nie=nie,overlapping=overlapping)
epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
return (eplom,epv)
end

"""
    (eplom,epv) = create_epv_eplom_sr1(;n=9,nie=5,overlapping=1, mul_v=100.)

Create an elemental partitioned limited-memory quasi-Newton operator `eplom` and elemental partitioned-vector `epv`.
Both have the same partitioned structure defined by the size of the problem `n :: Int`, the size of the element `nie :: Int` and the overlapping between the consecutive elements `overlapping :: Int`.
Each elemental element-matrix is instantiated as a `LBFGSOperator`, but it may change to a `LSR1Operator` later on.
The value of each elemental element-vector is `rand(nie) .* mul_v :: Real`.
Warning: You have to choose carefully the values `n`, `nie` and `overlap`, otherwise the method may fail.
"""
function create_epv_eplom(;n=9,nie=5,overlapping=1, mul_v=100.)
  eplom = PLBFGSR1_eplom(;n=n,nie=nie,overlapping=overlapping)
  epv = part_vec(;n=n,nie=nie,overlapping=overlapping,mul=mul_v)
  return (eplom,epv)
end

end