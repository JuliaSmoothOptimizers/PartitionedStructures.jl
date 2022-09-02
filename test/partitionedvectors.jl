epv = create_epv(element_variables; type=Float32)
_pv32 = PartitionedVector(epv)
pv32 = PartitionedVector(element_variables; T=Float32)

pv64 = PartitionedVector(element_variables)

pv_sim = similar(pv32)
pv_copy = copy(pv32)

x = [1.:1:n;]
pv32 .= x



pv64 .= x
pv = broadcast!(-, pv_copy, x)
broadcast!(+, pv_copy, pv64)

y = [2.:1:n+1;]
setindex!(pv64, y, 1:n)

pv_sim .= pv_copy .+ pv
pv_res = pv_copy .+ pv