module ModElemental_plom
using SparseArrays, LinearOperators
using ..M_part_mat, ..M_abstract_part_struct
using ..M_elt_mat, ..M_abstract_element_struct, ..ModElemental_elom_bfgs, ..ModElemental_elom_sr1

import Base.==, Base.copy, Base.similar
import ..M_part_mat: set_spm!, get_eelom_set, set_eelom_set!
import Base.Matrix, SparseArrays.SparseMatrixCSC
import ..M_abstract_part_struct: initialize_component_list!, get_ee_struct

export Elemental_plom
export get_eelom_set, get_spm, get_L, get_eelom_set_Bie, get_eelom_sub_set
export set_L!, set_L_to_spm!	

export identity_eplom_LOSE, PLBFGSR1_eplom, PLBFGSR1_eplom_rand

elom_type{T} = Union{Elemental_elom_sr1{T}, Elemental_elom_bfgs{T}}

"""
Elemental_plom_sr1{T} <: Part_LO_mat{T}

    Type that represents an elemental limited-memory partitioned quasi-Newton linear operator, each Bᵢ may use a LBFGS or LSR1 linear operator.
"""
mutable struct Elemental_plom{T} <: Part_LO_mat{T}
  N :: Int
  n :: Int
  eelom_set :: Vector{elom_type{T}}
  spm :: SparseMatrixCSC{T,Int}
  L :: SparseMatrixCSC{T,Int}
  component_list :: Vector{Vector{Int}}
  permutation :: Vector{Int} # n-size vector 
end

"""
    eelom_set = get_eelom_set(eplom)

Returns the vector of every elemental element linear operator `eplom.eelom_set`.
"""
@inline get_eelom_set(eplom :: Elemental_plom{T}) where T = eplom.eelom_set

"""
    eelom = get_eelom_set(eplom :: Elemental_plom_bfgs{T}, i :: Int)

Returns the `i`-th elemental element linear operator `eplom.eelom_set[i]`.
"""
@inline get_eelom_set(eplom :: Elemental_plom{T}, i :: Int) where T = @inbounds eplom.eelom_set[i]

"""
    set_eelom_set!(eplom, i, eelom)

Sets the `i`-th elemental elemental linear operator of `eplom` to `eelom`.
"""
@inline set_eelom_set!(eplom :: Elemental_plom{T}, i::Int, eelom :: Y) where Y <: LOEltMat{T} where T = @inbounds eplom.eelom_set[i] = eelom

"""
    eelom_set = get_ee_struct(eplom)

Returns the vector of every elemental element linear operator `eplom.eelom_set`.
"""
@inline get_ee_struct(eplom :: Elemental_plom{T}) where T = get_eelom_set(eplom)

"""
    eelom = get_ee_struct(eplom, i)

Returns the `i`-th elemental element linear operator `eplom.eelom_set[i]`.
"""
@inline get_ee_struct(eplom :: Elemental_plom{T}, i :: Int) where T = get_eelom_set(eplom, i)

"""
    eelom_subset = get_eelom_set_Bie(eplom, indices)

Returns a subset of the elemental element linear operators composing `eplom`.
`indices` selects the differents elemental element linear operators needed.
"""
@inline get_eelom_sub_set(eplom :: Elemental_plom{T}, indices :: Vector{Int}) where T = eplom.eelom_set[indices]

"""
    Bie = get_eelom_set_Bie(eplom, i)

Get the linear operator of the `i`-th elemental element linear operator of `eplom`.
"""
@inline get_eelom_set_Bie(eplom :: Elemental_plom{T}, i :: Int) where T = get_Bie(get_eelom_set(eplom, i))	

"""
    L = get_L(eplom)

Returns the sparse matrix `eplom.L`, who aims to store a Cholesky factor.
By default `eplom.L` is not instantiate.
"""
@inline get_L(eplom :: Elemental_plom{T}) where T = eplom.L

"""
    get_L(eplom, i, j)

Returns the value `eplom.L[i,j]`, from the sparse matrix `eplom.L`.
"""
@inline get_L(eplom :: Elemental_plom{T}, i :: Int, j :: Int) where T = @inbounds eplom.L[i, j]

"""
    set_L!(eplom, i, j, value)

Sets the value of `eplom.L[i,j] = value`.
"""
@inline set_L!(eplom :: Elemental_plom{T}, i :: Int, j :: Int, value :: T) where T = @inbounds eplom.L[i, j] = value

"""
    set_L_to_spm!(eplom, i, j, value)

Sets the sparse matrix `eplom.L` to the sparse matrix `eplom.spm`.
"""
@inline set_L_to_spm!(eplom :: Elemental_plom{T}) where T = eplom.L .= eplom.spm

@inline (==)(eplom1 :: Elemental_plom{T}, eplom2 :: Elemental_plom{T}) where T = (get_N(eplom1) == get_N(eplom2)) && (get_n(eplom1) == get_n(eplom2)) && (get_eelom_set(eplom1) .== get_eelom_set(eplom2)) && (get_permutation(eplom1) == get_permutation(eplom2))
@inline copy(eplom :: Elemental_plom{T}) where T = Elemental_plom{T}(copy(get_N(eplom)),copy(get_n(eplom)),copy.(get_eelom_set(eplom)),copy(get_spm(eplom)), copy(get_L(eplom)),copy(get_component_list(eplom)),copy(get_permutation(eplom)))
@inline similar(eplom :: Elemental_plom{T}) where T = Elemental_plom{T}(copy(get_N(eplom)),copy(get_n(eplom)),similar.(get_eelom_set(eplom)),similar(get_spm(eplom)), similar(get_L(eplom)),copy(get_component_list(eplom)),copy(get_permutation(eplom)))
  
"""
    identity_eplom_LOSE(vec_indices, N, n; T=T)

Creates an elemental partitionned limited-memory operator of `N` elemental element linear operators initialized with LBFGS operators.
The positions are given by the vector of the element variables `element_variables`.
"""
function identity_eplom_LOSE(element_variables :: Vector{Vector{Int}}, N :: Int, n :: Int; T=Float64)
  eelom_set = map( (elt_var -> init_eelom_LBFGS(elt_var; T=T)), element_variables)
  spm = spzeros(T, n, n)
  L = spzeros(T, n, n)
  component_list = map(i -> Vector{Int}(undef, 0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom{T}(N, n, eelom_set, spm, L, component_list, no_perm)
  initialize_component_list!(eplom)
  return eplom
end

"""
    PLBFGSR1_eplom(; n, type, nie, overlapping, prob)

Creates an elemental partitionned limited-memory operator PLSE of `N` (deduced from `n` and `nie`) elemental element linear operators.
Each element overlaps the coordinates of the next element by `overlapping` components.
Each element is randomly (rand() > p) choose between an elemental element LBFGS operator or an elemental element LSR1 operator.
"""
function PLBFGSR1_eplom(;n::Int=9, T=Float64, nie::Int=5, overlapping::Int=1, prob=0.5)		
  overlapping < nie || error("l'overlapping doit être plus faible que nie")
  mod(n-(nie-overlapping), nie-overlapping) == mod(overlapping, nie-overlapping) || error("wrong structure: mod(n-(nie-over), nie-over) == mod(over, nie-over) must holds")

  indices = filter(x -> x <= n-nie+1, vcat(1,(x -> x + (nie-overlapping)).([1:nie-overlapping:n-(nie-overlapping);])))
  eelom_set = map(i -> rand() > prob ? LBFGS_eelom(nie;T=T,index=i) : LSR1_eelom(nie;T=T,index=i), indices)	
  N = length(indices)
  spm = spzeros(T,n,n)
  L = spzeros(T,n,n)
  component_list = map(i -> Vector{Int}(undef,0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom{T}(N,n,eelom_set,spm,L,component_list,no_perm)
  initialize_component_list!(eplom)
  return eplom
end 

"""
    PLBFGS_eplom_rand(N,n; type, nie)

Create an elemental partitionned limited-memory operator PLSE of `N` elemental element linear operators.
The size of each element is `nie`, whose positions are random in the range `1:n`.
Each element is randomly (rand() > p) choose between an elemental element LBFGS operator or an elemental element LSR1 operator.
"""
function PLBFGSR1_eplom_rand(N :: Int, n ::Int; T=Float64, nie::Int=5, prob=0.5)		
  eelom_set = map(i -> rand() > prob ? LBFGS_eelom_rand(nie;T=T,n=n) : LSR1_eelom_rand(nie;T=T,n=n), [1:N;])
  spm = spzeros(T,n,n)
  L = spzeros(T,n,n)
  component_list = map(i -> Vector{Int}(undef,0), [1:n;])
  no_perm = [1:n;]
  eplom = Elemental_plom{T}(N,n,eelom_set,spm,L,component_list,no_perm)
  initialize_component_list!(eplom)	
  return eplom
end 

"""
    initialize_component_list!(eplom)

Builds for each index i (∈ {1, ..., n}) a list of the elements using the i-th variable.
"""
function initialize_component_list!(eplom::Elemental_plom)
  N = get_N(eplom)
  for i in 1:N
    eelomᵢ = get_eelom_set(eplom,i)
    _indices = get_indices(eelomᵢ)
    for j in _indices 
      push!(get_component_list(eplom,j),i)
    end 
  end 
end 

"""
    set_spm!(eplom)

Builds the sparse matrix of `eplom` in `eplom.spm` from all the elemental element linear operator.
The sparse matrix is built with respect to the indices of each elemental element linear operator.
"""
function set_spm!(eplom :: Elemental_plom{T}) where T
  reset_spm!(eplom) # eplom.spm .= 0
  N = get_N(eplom)	
  n = get_n(eplom)
  spm = get_spm(eplom)
  for i in 1:N
    eplomᵢ = get_eelom_set(eplom,i)
    nie = get_nie(eplomᵢ)
    Bie = get_Bie(eplomᵢ)
    indicesᵢ = get_indices(eplomᵢ)
    value_Bie = zeros(T,nie,nie)
    map( (i -> value_Bie[:,i] .= Bie*SparseVector(nie,[i],[1])), 1:nie)
    spm[indicesᵢ,indicesᵢ] .+= value_Bie			
  end 
end

function Base.Matrix(eplom :: Elemental_plom{T}) where T
  set_spm!(eplom)
  sp_eplom = get_spm(eplom)
  m = Matrix(sp_eplom)
  return m
end 

SparseArrays.SparseMatrixCSC(eplom :: Elemental_plom{T}) where T = begin set_spm!(eplom); get_spm(eplom) end 

end 