# PartitionedStructures.jl: Partitioned derivatives storage and partitioned quasi-Newton updates

## Philosophy
The methods exploiting the derivatives of partially separable functions require specific structures to store their partitioned derivatives.
In the case of the partitioned quasi-Newton methods, it requires the storage of partitioned gradients and the partitioned matrix.
[PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl) facilitates the definition of those partitioned structures and define methods to ease their use.

## Compatibility
Julia 1 up to 1.7.

## How to install
```julia
julia> ]
pkg> add https://github.com/paraynaud/PartitionedStructures.jl
pkg> test PartitionedStructures
```

## How to use
Check the [tutorial](https://paraynaud.github.io/PartitionedStructures.jl/dev/tutorial/).

## Partitioned structures available
Structure              | Description
-----------------------|------------
`Part_struct`          | The (abstract) supertype of all partitioned structures
`Elemental_pm`         | An elemental partitioned matrix, each element-matrix is dense
`Elemental_plo_bfgs`  | A limited-memory elemental partitioned matrix, each elemental element-matrix is a `LBFGSOperator`
`Elemental_plo_sr1`   | A limited-memory elemental partitioned matrix, each elemental element-matrix is a `LSR1Operator`
`Elemental_plo`       | A limited-memory elemental partitioned matrix, each elemental element-matrix is a `LBFGSOperator` or a `LSR1Operator`
`Elemental_pv`         | An elemental partitioned-vector

## Methods available
Method                 | Description
-----------------------|------------
`identity_epm`         | Creates a partitioned-matrix with identity element-matrix
`identity_eplo_LBFGS` | Creates a limited-memory (LBFGS) partitioned matrix
`identity_eplo_LSR1`  | Creates a limited-memory (LSR1) partitioned matrix
`identity_eplo_LOSE`  | Creates a limited-memory (with both LBFGS and LSR1) partitioned matrix
`update`               | Performs a partitioned quasi-Newton update onto a partitioned matrix
`eplo_lbfgs_from_epv` | Creates an `Elemental_plo_bfgs` from the partitioned structure of an `Elemental_pv`
`eplo_lsr1_from_epv`  | Creates an `Elemental_plo_sr1` from the partitioned structure of an `Elemental_pv`
`eplo_lose_from_epv`  | Creates an `Elemental_plo` from the partitioned structure of an `Elemental_pv`
`epm_from_epv`         | Creates an `Elemental_pm` from the partitioned structure of an `Elemental_pv`
`epv_from_epm`         | Creates an `Elemental_pv` from the partitioned structure of an `Elemental_pm`
`epv_from_eplo`       | Creates an `Elemental_pv` from the partitioned structure of: an `Elemental_plo`, an `Elemental_plo_bfgs` or an `Elemental_plo_sr1`
`mul_epm_epv`          | Return a partitioned-vector from an elementwise product between a partitioned-matrix and a partitioned-vector
`mul_epm_vector`       | Return the vector resulting of a product partitioned-matrix vector
`build_v!`             | Builds the vector associated to a partitioned-vector
`get_v`                | Return the vector associated to a partitioned-vector  **Warning: it doesn't build the vector**
`set_epv!`             | Set the value of every element-vectors
`minus_epv!`           | Applies a unary minus on every element-vector of a partitioned-vector
`add_epv!`             | Performs an elementwise addition between two partitioned-vectors

## Modules applying [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl)
These structures are applied in the module
[PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) inside a trust-region using partitioned quasi-Newton operators and in [PartitionedKnetNLPModel.jl](https://github.com/paraynaud/PartitionedKnetNLPModels.jl) to train a neural network of classification with a limited-memory partitioned quasi-Newton stochastic method.

## Features
For now, PartitionedStructures.jl supports only the elemental Uᵢ, i.e. a linear operator where the lines of Uᵢ are vectors from the euclidean basis.
Concretely, each Uᵢ is a vector of size nᵢ who's indicates the indices of the variables used by the i-th element function.