using PartitionedStructures
using PartitionedStructures.ModElemental_pv
using PartitionedStructures.M_part_mat
using PartitionedStructures.Instances
using LinearAlgebra, LinearOperators, SparseArrays


@testset "Test PLBFGS damped or not" begin
	n = 150
	N = 100
	ni = 9
	sp_vec_elt(ni::Int, n; p=ni/n ) = sprand(Float64, n, p)
	vec_sp_eev = map(i -> sp_vec_elt(ni, n), 1:N)
	# map!(x -> x .= 100*x .- 50, vec_sp_eev)
	# map(spx -> map!(val -> val = val*100 - 50, spx.nzval), vec_sp_eev)
	map(spx -> begin spx.nzval .*= 100; spx.nzval .-= 50 end, vec_sp_eev)
	epv = create_epv(vec_sp_eev)
	build_v!(epv)
	y = get_v(epv)
	epm = epm_from_epv(epv)

	eplom_lbfgs = eplom_lbfgs_from_epv(epv; damped=false)
	eplom_lbfgs_damped = eplom_lbfgs_from_epv(epv; damped=true)

	s = ones(n)

	PLBFGS_update!(eplom_lbfgs, epv, s)
	PLBFGS_update!(eplom_lbfgs_damped, epv, s)

end