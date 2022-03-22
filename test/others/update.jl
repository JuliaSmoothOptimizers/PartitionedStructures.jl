using PartitionedStructures.Instances
@testset "Test update/update!" begin

	@testset " partitioned matrix update" begin
		n=9
		epm_bfgs,epv_y = create_epv_epm(;n=n,nie=5,overlapping=1,mul_m=5., mul_v=100.)
		epm_sr1 = copy(epm_bfgs)
		s = ones(n)
		mat_PBFGS = update(epm_bfgs, epv_y, s)
		mat_PSR1 = update(epm_sr1, epv_y, s; name=:psr1)
	end
	
	@testset "partitioned linear operators update" begin
		n=10
		nie=4
		over=2
		s = ones(n)
		eplom_bfgs,epv_y = create_epv_eplom_bfgs(; n=n, nie=nie, overlapping=over)
		mat_PLBFGS = update(eplom_bfgs, epv_y, s)
		eplom_sr1,epv_y = create_epv_eplom_sr1(; n=n, nie=nie, overlapping=over)
		mat_PSR1 = update(eplom_sr1, epv_y, s)
		eplom_bfgs_sr1,epv_y = create_epv_eplom(;n=n,nie=nie,overlapping=over)		
		mat_PLBFGS_SR1 = update(eplom_bfgs_sr1, epv_y, s)
	end
end