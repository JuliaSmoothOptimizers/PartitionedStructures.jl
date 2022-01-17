using PartitionedStructures
using PartitionedStructures.Link, PartitionedStructures.M_part_v, PartitionedStructures.PartitionedQuasiNewton

n=9
epm_B1,epv_y1 = create_epv_epm(;n=n,nie=5,overlapping=1,mul_m=5., mul_v=100.)
epm_B2,epv_y2 = create_epv_epm(;n=n,nie=3,overlapping=0,mul_m=5., mul_v=100.)

s = ones(n)

epm_B11 = PBFGS(epm_B1,epv_y1,s) 
epm_B12 = PSR1(epm_B1,epv_y1,s) 


@test Matrix(epm_B1) == transpose(Matrix(epm_B1))
@test Matrix(epm_B11) != Matrix(epm_B1)
@test Matrix(epm_B11) != Matrix(epm_B12)

@test Matrix(epm_B11) == transpose(Matrix(epm_B11))
@test Matrix(epm_B12) == transpose(Matrix(epm_B12))

@test_throws DimensionMismatch PBFGS(epm_B1,epv_y2,s)
@test_throws DimensionMismatch PSR1(epm_B1,epv_y2,s) 