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

  @testset "partitioned linear operators update!" begin
    n=10
    nie=4
    over=2
    s = ones(n)
    eplom_bfgs,epv_y = create_epv_eplom_bfgs(; n=n, nie=nie, overlapping=over)
    mat_PLBFGS = update!(eplom_bfgs, epv_y, s)
    eplom_sr1,epv_y = create_epv_eplom_sr1(; n=n, nie=nie, overlapping=over)
    mat_PSR1 = update!(eplom_sr1, epv_y, s)
    eplom_bfgs_sr1,epv_y = create_epv_eplom(;n=n,nie=nie,overlapping=over)
    mat_PLBFGS_SR1 = update!(eplom_bfgs_sr1, epv_y, s)
  end
end

@testset "General update" begin
  n = 150
  N = 100
  ni = 9
  sp_vec_elt(ni::Int, n; p=ni/n ) = sprand(Float64, n, p)
  vec_sp_eev = map(i -> sp_vec_elt(ni, n), 1:N)
  map(spx -> begin spx.nzval .*= 100; spx.nzval .-= 50 end, vec_sp_eev)
  epv = create_epv(vec_sp_eev)
  build_v!(epv)
  y = get_v(epv)
  epm = epm_from_epv(epv)
  eplom_lbfgs = eplom_lbfgs_from_epv(epv)
  eplom_sr1 = eplom_lsr1_from_epv(epv)
  eplom_lose = eplom_lose_from_epv(epv)

  s = ones(n)

  epm_bfgs = PBFGS_update(epm, epv, s)
  epm_sr1 = PSR1_update(epm, epv, s)
  epm_se = PSE_update(epm, epv, s)
  epm_plbfgs = PLBFGS_update(eplom_lbfgs, epv, s)
  epm_plsr1 = PLSR1_update(eplom_sr1, epv, s)
  epm_plse = PLSE_update(eplom_lose, epv, s)
end